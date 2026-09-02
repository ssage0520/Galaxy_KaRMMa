"""Construct a feasible starting position for sampling, from the observed counts.

`ForwardModel.log_prob` is `-inf` unless `Ng_obs < n + 1` at every masked
pixel, and `n` grows with the local density, so the highest-count pixels demand
local overdensity. A prior draw fails that condition at ~0.6% of pixels, and so
does `xlm = 0`, which leaves nothing to shrink toward. Both halves of the
position therefore have to be built from the data.

`init_xlm` inverts `dg_obs` through the point transform for a seed, then Wiener
filters that into an approximate posterior draw by matrix-free CG. The
refinement is not cosmetic: the raw inversion overfits the shot noise by ~3e4
nats, and `refine_theta` run against it infers too little scatter and pushes
`mu0` onto the support boundary. Only the refined draw is offered as a starting
point.

`refine_theta` then maximizes `log_prob` over `theta` at fixed `xlm`. Its
purpose is consistency rather than accuracy — the Hessian that sets the
sampler's whitening is only representative at a self-consistent pair.
"""

import warnings

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jax.scipy.sparse.linalg import cg
from scipy.special import legendre_p_all, roots_legendre
from scipy.stats import kurtosis, skew

from karmma.forward_model import ForwardModel
from karmma.structs import KarmmaPosition, ThetaParams, XlmParams
from karmma.transforms import alm2map, map2alm

# Keeps `dg_obs.min() / b` strictly inside the transform's domain rather than
# exactly on its boundary, where G2's log diverges.
_FLOOR_MARGIN = 1e-3

# Floor on the binomial variance used to build the CG noise weighting, guarding
# the reciprocal where a pixel's predicted variance underflows.
_VAR_FLOOR = 1e-30

# Guard on band-power divisions in `fit_transfer_and_noise`, where an empty or
# pathological band would otherwise divide by zero.
_CL_FLOOR = 1e-30


class InfeasibleInitError(RuntimeError):
    """Raised when the constructed initial `xlm` falls outside the likelihood's support."""


def _seed_xlm(model: ForwardModel) -> XlmParams:
    """Build the linearization point for `init_xlm` by inverting `dg_obs`.

    Parameters
    ----------
    model : ForwardModel
        Model supplying `dg_obs`, the point transform, and `L_G`.

    Returns
    -------
    XlmParams
        Seed coefficients, rescaled to unit rms.

    Notes
    -----
    Pipeline: `dm = dg_obs / b` -> `gn` -> `map2alm` at `gen_lmax` ->
    `unapply_CL_G` -> `pack_xlm` -> rescale to unit rms.

    `b` is only ever a domain guard, set to the smallest value keeping every
    pixel inside the point transform's range: an empty pixel has
    `dg_obs = -1`, and `gn` rejects `dm` at or below `-beta` for G2 or `-1`
    for G3. It is load-bearing -- `b = 1` raises outright -- but its precise
    value is not: varying it from the bound up to 2.8x left the refined
    draw's support margin, residual correlation and whitening `cond`
    unchanged, and its overfit within 4%. Earlier versions root-found `b` to
    land the seed at unit rms; the explicit rescale below now does that
    directly.

    The seed's *structure*, unlike `b`, does matter. Seeding from a prior
    draw (`make_random_xlm`) fails outright: the second Gauss-Newton pass
    returns NaN, because a full-amplitude prior field puts `deff` outside
    the range `deff_to_binom_params` can evaluate. Linearizing at `xlm = 0`
    does converge, and is only modestly worse on one mock (residual
    correlation `cond` 9.9 against 8.4, both measured at the true theta) --
    not enough evidence to justify dropping this stage, but not much of a
    margin either.

    This is deliberately *not* checked against the likelihood's support.
    It is only ever a point to linearize the forward model around, which
    needs no feasibility, and rejecting a seed here would abort runs whose
    refined draw would have been perfectly valid.

    On its own this field overfits the shot noise by ~3e4 nats, which is
    what biases a subsequent `refine_theta` into pushing `mu0` onto the
    support boundary; `init_xlm` exists to remove that. Only multipoles
    below roughly `2 * Nside` are genuinely recovered, and the higher modes
    come out at the right amplitude but nearly uncorrelated with the
    truth — the right behaviour for a starting point, since the data
    cannot constrain them either.
    """
    dg_obs = np.asarray(model.dg_obs)
    floor = (
        -np.asarray(model.lbda[1])
        if model.gn_order == 2
        else np.full(model.Nbins, -1.0)
    )
    b = np.abs(dg_obs.min(axis=1)) / np.abs(floor) * (1.0 + _FLOOR_MARGIN)
    ys = model.gn(dg_obs / b[:, np.newaxis], model.gn_order, model.lbda)
    ylm = map2alm(jnp.asarray(ys), model.gen_lmax)
    xlm = model.pack_xlm(jnp.asarray(model.unapply_CL_G(ylm)))

    rms = float(
        jnp.sqrt(
            jnp.mean(jnp.concatenate([xlm.real.ravel() ** 2, xlm.imag.ravel() ** 2]))
        )
    )
    print(
        f"xlm init: seed from data (domain guard b = {np.array2string(b, precision=4)})"
    )
    print(
        f"  Gaussianized field: skew "
        f"{np.array2string(skew(ys, axis=1), precision=3)}"
        f", excess kurtosis {np.array2string(kurtosis(ys, axis=1), precision=3)}"
    )
    print(
        f"    (both are 0 for an exact G{model.gn_order} fit and noiseless data; "
        "large values mean the point transform is misspecified)"
    )
    return XlmParams(real=xlm.real / rms, imag=xlm.imag / rms)


def init_xlm(
    model: ForwardModel,
    theta: ThetaParams,
    key: jax.Array,
    n_gauss_newton: int = 2,
    cg_maxiter: int = 120,
    cg_tol: float = 1e-8,
    max_tries: int = 3,
) -> XlmParams:
    """Build an initial `xlm`: an approximate posterior draw given `dg_obs`.

    Parameters
    ----------
    model : ForwardModel
        Model supplying `dg_obs`, the mask, and the forward map.
    theta : ThetaParams
        Bias/nuisance parameters, held fixed throughout.
    key : jax.Array
        PRNG key for the constrained realization's two random draws.
    n_gauss_newton : int, optional
        Gauss-Newton relinearizations before the final solve, by default 2.
        Total CG solves is this plus two.
    cg_maxiter : int, optional
        Maximum CG iterations per solve, by default 120.
    cg_tol : float, optional
        CG convergence tolerance, by default 1e-8.
    max_tries : int, optional
        Constrained realizations to attempt before giving up, by default 3.
        Each retry redraws with a fresh key; the Wiener solves are reused.

    Returns
    -------
    XlmParams
        Refined `xlm`, feasible under `model.log_prob`.

    Raises
    ------
    InfeasibleInitError
        If every attempted realization violates the binomial support.

    Notes
    -----
    Seeds with `_seed_xlm` (which inverts `dg_obs` through the point
    transform), then solves the Gaussian-approximation posterior for `xlm` given `dg_obs`,
    `(I + A^T N^-1 A) x = A^T N^-1 d`, matrix-free: `A` and `A^T` come from
    `jax.linearize`/`jax.vjp` of the forward map, and `N` is the diagonal
    binomial variance `n p (1 - p) / N_bar^2` restricted to the mask. The
    `xlm` prior is exactly `N(0, I)`, which is what makes the identity the
    only prior term. `A` depends on `xlm` through the point transform, so
    the solve is repeated at successive linearization points.

    The returned field is a *constrained realization*, not the Wiener mean:

        x_draw = x_hat(d) + [omega - x_hat(A omega + eta)]

    with `omega` a prior draw and `eta` a noise draw. This is exact for the
    linearized model, since `W A = I - Sigma` and `W N W^T = Sigma -
    Sigma^2` give it covariance `Sigma`. The mean alone would be far too
    smooth to be a typical posterior sample: it suppresses every mode the
    data does not constrain, whereas sampling needs them at prior
    amplitude.

    Compared with the seed on its own, this cuts the shot-noise overfit by
    roughly two orders of magnitude, which is what removes the `mu0` bias a
    subsequent `refine_theta` would otherwise inherit. The seed is never a
    valid starting point by itself and is not offered as one.
    """
    xlm_seed = _seed_xlm(model)

    mask_row = jnp.asarray(model.mask)[None, :]
    n_bar = jnp.asarray(model.N_bar)[:, None]
    # Zero dg_obs off-mask up front: real data may carry NaN there, and those
    # pixels are dropped by the weighting anyway.
    dg_obs = jnp.where(mask_row, jnp.asarray(model.dg_obs), 0.0)

    def forward(xlm: XlmParams) -> jnp.ndarray:
        """Predict the full-sky mean `dg` from `xlm` at fixed `theta`."""
        n, p = model.deff_to_binom_params(
            model.xlm_to_deff(xlm, theta), theta, mask_output=False
        )
        return n * p / n_bar - 1.0

    def linearize_at(xlm: XlmParams) -> tuple:
        """Build the linear operators and noise weighting at a linearization point."""
        pred, jvp = jax.linearize(forward, xlm)
        _, vjp = jax.vjp(forward, xlm)
        n, p = model.deff_to_binom_params(
            model.xlm_to_deff(xlm, theta), theta, mask_output=False
        )
        var = jnp.maximum(n * p * (1.0 - p) / n_bar**2, _VAR_FLOOR)
        return pred, jax.jit(jvp), jax.jit(lambda u: vjp(u)[0]), var

    def solve(rhs: jnp.ndarray, ops: tuple) -> tuple[XlmParams, float]:
        """Solve `(I + A^T N^-1 A) x = A^T N^-1 rhs`, returning `x` and its residual."""
        _, jvp, vjp_fn, var = ops

        def weight(residual: jnp.ndarray) -> jnp.ndarray:
            """Apply `N^-1`, as a select so off-mask NaN cannot leak in."""
            return jnp.where(mask_row, residual / var, 0.0)

        def matvec(xlm: XlmParams) -> XlmParams:
            """Apply `I + A^T N^-1 A`."""
            data_term = vjp_fn(weight(jvp(xlm)))
            return XlmParams(
                real=xlm.real + data_term.real, imag=xlm.imag + data_term.imag
            )

        def norm(xlm: XlmParams) -> float:
            """Euclidean norm over both halves of an `XlmParams`."""
            return float(jnp.sqrt(jnp.sum(xlm.real**2) + jnp.sum(xlm.imag**2)))

        b = vjp_fn(weight(rhs))
        x, _ = cg(matvec, b, maxiter=cg_maxiter, tol=cg_tol)
        resid = jax.tree.map(lambda u, v: u - v, matvec(x), b)
        return x, norm(resid) / norm(b)

    print(
        f"  CG refinement: {n_gauss_newton} Gauss-Newton passes, "
        f"{n_gauss_newton + 2} solves, cg_maxiter={cg_maxiter}"
    )

    # Gauss-Newton on the Wiener mean. The linearized data vector is
    # d - f(x0) + A x0, which reduces to d when the model is already linear.
    xlm = xlm_seed
    residuals = []
    for _ in range(n_gauss_newton):
        ops = linearize_at(xlm)
        xlm, resid = solve(dg_obs - ops[0] + ops[1](xlm), ops)
        residuals.append(resid)

    ops = linearize_at(xlm)
    pred, jvp, _, var = ops
    mean, resid = solve(dg_obs - pred + jvp(xlm), ops)
    residuals.append(resid)
    noise_std = jnp.sqrt(var)

    for attempt in range(max_tries):
        key, omega_key, eta_key = jax.random.split(key, 3)
        omega = model.make_random_xlm(omega_key)
        eta = noise_std * jax.random.normal(eta_key, shape=dg_obs.shape)
        omega_wiener, resid = solve(jvp(omega) + eta, ops)
        draw = XlmParams(
            real=mean.real + omega.real - omega_wiener.real,
            imag=mean.imag + omega.imag - omega_wiener.imag,
        )

        n, _ = model.deff_to_binom_params(
            model.xlm_to_deff(draw, theta), theta, mask_output=True
        )
        margin = np.asarray(n) + 1 - model.Ng_obs
        # `~(margin > 0)` rather than `margin <= 0`: a non-finite margin must count as a
        # failure, and `NaN <= 0` is False. Without this a NaN field is returned as
        # feasible, and the caller only finds out when refine_theta reports log_prob is
        # not finite -- several steps from the actual cause.
        n_nonfinite = int((~np.isfinite(margin)).sum())
        n_bad = int((~(margin > 0)).sum())

        rms = float(
            jnp.sqrt(
                jnp.mean(
                    jnp.concatenate([draw.real.ravel() ** 2, draw.imag.ravel() ** 2])
                )
            )
        )
        print(
            f"  attempt {attempt + 1}: rms {rms:.4f}, support margin min "
            f"{np.nanmin(margin):+.2f}, median {np.nanmedian(margin):.1f}"
            + (f", NON-FINITE at {n_nonfinite} terms" if n_nonfinite else "")
        )
        if not n_bad:
            print(
                f"  CG relative residuals: "
                f"{np.array2string(np.array(residuals + [resid]), precision=1)}"
            )
            worst = max(residuals + [resid])
            if worst > 1e-3:
                warnings.warn(
                    f"init_xlm: worst CG relative residual {worst:.2e} exceeds "
                    "1e-3; raise cg_maxiter or loosen cg_tol expectations.",
                    stacklevel=2,
                )
            return draw

    raise InfeasibleInitError(
        f"CG constrained realization fell outside the likelihood's support on all "
        f"{max_tries} attempts: {n_bad} of {margin.size} masked pixel-bin terms "
        f"have Ng_obs >= n + 1 (or a non-finite margin, at {n_nonfinite} of them), "
        f"worst finite margin {np.nanmin(margin):+.2f}.\n"
        f"  The Wiener mean itself is reusable, so this is the prior draw pushing "
        f"the field outside the support. Check that theta is sensible and that the "
        f"seed was feasible to begin with."
    )


def refine_theta(
    model: ForwardModel,
    xlm: XlmParams,
    theta_start: ThetaParams,
    n_iter: int = 90,
    step_init: float = 0.05,
    max_backtracks: int = 45,
    max_stalls: int = 3,
) -> ThetaParams:
    """Maximize `log_prob` over `theta` at fixed `xlm` by backtracking ascent.

    Parameters
    ----------
    model : ForwardModel
        Model supplying `log_prob`.
    xlm : XlmParams
        Field to condition on, held fixed throughout.
    theta_start : ThetaParams
        Starting point. Must already be inside the likelihood's support.
    n_iter : int, optional
        Maximum ascent steps, by default 90.
    step_init : float, optional
        Initial step length in rescaled units, by default 0.05.
    max_backtracks : int, optional
        Step halvings per iteration before declaring a stall, by default 45.
    max_stalls : int, optional
        Consecutive stalled iterations tolerated before stopping, by
        default 3. Each stall also shrinks the trial step tenfold.

    Returns
    -------
    ThetaParams
        Refined parameters, at least as good as `theta_start`.

    Raises
    ------
    InfeasibleInitError
        If `log_prob` is not finite at `theta_start`.

    Notes
    -----
    Hand-rolled rather than delegated to `scipy.optimize`: gradients here run
    to ~1e6 while the parameters themselves are ~0.01, and L-BFGS-B returns
    immediately with `nit=0` on that scaling. Steps are taken along the
    gradient rescaled by `max(|theta|, 0.01)` per component, and every trial
    is rejected unless `log_prob` is finite and strictly improved, so the
    iterate cannot leave the support.

    This makes `theta` *consistent with* `xlm` rather than more accurate. At
    a `xlm` that overfits the shot noise the residuals are too small, so the
    fit infers too little scatter and pushes `mu0` up; refining against a
    posterior draw from `init_xlm` largely removes that. The pairing
    matters more than the accuracy — the Hessian used for whitening is only
    representative at a self-consistent `(xlm, theta)`.
    """
    flat_start, unflatten = ravel_pytree(theta_start)

    def flat_log_prob(flat: jax.Array) -> jax.Array:
        """Evaluate `log_prob` on a flattened `theta` at fixed `xlm`."""
        return model.log_prob(KarmmaPosition(xlm=xlm, theta=unflatten(flat)))

    log_prob = jax.jit(flat_log_prob)
    grad_log_prob = jax.jit(jax.grad(flat_log_prob))

    x = np.asarray(flat_start, dtype=float)
    scale = np.maximum(np.abs(x), 0.01)
    f = float(log_prob(jnp.asarray(x)))
    if not np.isfinite(f):
        raise InfeasibleInitError(
            "refine_theta: log_prob is not finite at theta_start, so there is no "
            "ascent direction to follow. The starting theta must already be inside "
            "the binomial support at this xlm."
        )

    f_start = f
    step = step_init
    stalls = 0
    n_accepted = 0
    for _ in range(n_iter):
        g = np.asarray(grad_log_prob(jnp.asarray(x))) * scale
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-30:
            break
        direction = (g / g_norm) * scale

        trial = step
        accepted = False
        for _ in range(max_backtracks):
            x_trial = x + trial * direction
            f_trial = float(log_prob(jnp.asarray(x_trial)))
            if np.isfinite(f_trial) and f_trial > f:
                accepted = True
                break
            trial *= 0.5

        if not accepted:
            stalls += 1
            if stalls >= max_stalls:
                break
            step *= 0.1
            continue

        stalls = 0
        n_accepted += 1
        x, f = x_trial, f_trial
        step = trial * 2.0

    print(
        f"theta init: refined at fixed xlm ({n_accepted} steps accepted, "
        f"log_prob {f_start:.6e} -> {f:.6e})"
    )
    return unflatten(jnp.asarray(x))


def _cl_bands(lmax: int, l_min: int = 25, min_width: int = 8) -> list[tuple[int, int]]:
    """Log-spaced Cl bands, each at least `min_width` wide since the mask couples `dl ~ 5`.

    Starts at `l_min = 25` because the mask-mean subtraction in
    `fit_transfer_and_noise` suppresses power below that, by an amount the model
    does not reproduce.
    """
    candidates = np.unique(np.round(np.geomspace(l_min, lmax + 1, 11)).astype(int))
    edges = [int(candidates[0])]
    for edge in candidates[1:]:
        if edge - edges[-1] >= min_width:
            edges.append(int(edge))
    edges[-1] = lmax + 1
    return list(zip(edges[:-1], edges[1:], strict=True))


def fit_transfer_and_noise(
    model: ForwardModel, n_template: int = 5, key: jax.Array | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the effective `dm -> dg` transfer and the noise level from `dg_obs` alone.

    Parameters
    ----------
    model : ForwardModel
        Supplies `dg_obs`, `mask`, `N_bar` and the point transform.
    n_template : int, optional
        Prior draws averaged for `Cl_dm`, by default 5.
    key : jax.Array, optional
        PRNG key for those draws. Defaults to a fixed key: the average is a
        model quantity, so it needs no entropy from the caller.

    Returns
    -------
    b_eff : np.ndarray
        Effective transfer amplitude per bin, shape (Nbins,).
    kappa : np.ndarray
        `<1 - p>` weighted by mean count, per bin, shape (Nbins,).

    Notes
    -----
    Fits, per bin, in inverse-variance-weighted bands from `_cl_bands`:

        pseudoCl(dg_obs) = b_eff^2 * pseudoCl(Cl_dm) + f_sky * Omega_pix * kappa / N_bar

    Linear in `(b_eff^2, N_ell)`, so one `lstsq` per bin; the terms separate
    because the signal falls with `ell` while the noise is flat. Only one factor
    of `N_bar` survives because `<mean_Ng>_mask = N_bar` identically, cancelling
    one of the two from `dg = counts/N_bar - 1`.

    `Cl_dm` comes from `n_template` prior draws, and the mask enters through the
    identity `pseudo-Cl = Legendre[xi * xi_mask]`. The transfer's `ell`-shape is
    held constant, so `b_eff` is its band-averaged amplitude.
    """
    lmax, pad = model.lmax, model.pad_lmax
    mask = np.asarray(model.mask)
    n_bar = np.asarray(model.N_bar)
    f_sky = float(mask.mean())
    omega_pix = 4.0 * np.pi / mask.size

    mu, quad_w = roots_legendre(2 * pad)
    legendre = legendre_p_all(pad, mu).squeeze()

    def to_xi(cl: np.ndarray, lm: int) -> np.ndarray:
        return ((2 * np.arange(lm + 1) + 1) * cl) @ legendre[: lm + 1] / (4 * np.pi)

    def to_cl(xi: np.ndarray) -> np.ndarray:
        return 2 * np.pi * (xi * quad_w) @ legendre[: lmax + 1].T

    xi_mask = to_xi(hp.anafast(mask.astype(float), lmax=pad, use_pixel_weights=True), pad)

    key = jax.random.PRNGKey(0) if key is None else key
    cl_dm = np.zeros((model.Nbins, lmax + 1))
    for t in range(n_template):
        xlm = model.make_random_xlm(jax.random.fold_in(key, t))
        ys = alm2map(model.apply_CL_G(model.unpack_xlm(xlm)), model.Nside, model.gen_lmax)
        dm_lm = np.asarray(map2alm(model.gn_inv(ys, model.gn_order, model.lbda), lmax))
        for i in range(model.Nbins):
            cl_dm[i] += hp.alm2cl(np.ascontiguousarray(dm_lm[i])) / n_template

    bands = _cl_bands(lmax)
    n_mode = np.array([f_sky * np.sum(2 * np.arange(lo, hi) + 1) for lo, hi in bands])
    b_eff, kappa = np.empty(model.Nbins), np.empty(model.Nbins)
    for i in range(model.Nbins):
        dg = np.where(mask, np.asarray(model.dg_obs[i]), 0.0)
        cl_obs = hp.anafast(dg - dg[mask].mean() * mask, lmax=lmax)
        signal = to_cl(to_xi(cl_dm[i], lmax) * xi_mask)
        obs_b = np.array([cl_obs[lo:hi].mean() for lo, hi in bands])
        signal_b = np.array([signal[lo:hi].mean() for lo, hi in bands])

        weight = np.sqrt(n_mode / 2.0) / np.maximum(obs_b, _CL_FLOOR)
        design = np.stack([signal_b, np.ones_like(signal_b)], axis=1) * weight[:, None]
        (amplitude, n_ell), *_ = np.linalg.lstsq(design, obs_b * weight, rcond=None)

        b_eff[i] = np.sqrt(max(float(amplitude), _CL_FLOOR))
        kappa[i] = float(np.clip(n_ell * n_bar[i] / (f_sky * omega_pix), 1e-3, 1.0))
    print(
        f"  transfer fit: b_eff {np.array2string(b_eff, precision=3)}, "
        f"kappa {np.array2string(kappa, precision=3)}"
    )
    return b_eff, kappa


def init_xlm_theta_free(
    model: ForwardModel,
    key: jax.Array,
    n_gauss_newton: int = 2,
    cg_maxiter: int = 120,
    cg_tol: float = 1e-8,
) -> XlmParams:
    """Build an initial `xlm` from `dg_obs` **without knowing `theta`**.

    Parameters
    ----------
    model : ForwardModel
        Supplies `dg_obs`, `mask`, `N_bar` and the theta-free part of the
        forward map.
    key : jax.Array
        PRNG key for the constrained realization's prior and noise draws.
    n_gauss_newton : int, optional
        Gauss-Newton passes on the Wiener mean, by default 2.
    cg_maxiter, cg_tol : optional
        Passed to the CG solver.

    Returns
    -------
    XlmParams
        A constrained realization of the linearized posterior.

    Raises
    ------
    InfeasibleInitError
        If the draw comes back non-finite. Unlike `init_xlm` this cannot check
        the likelihood's support, since the margin `n + 1 - Ng_obs` needs
        `theta`; feasibility is the caller's to check once `refine_theta` has
        run.

    Notes
    -----
    Same structure as `init_xlm` — Gauss-Newton on the Wiener mean, then a
    constrained realization — with the operator replaced by `b_eff *
    xlm_to_dm` and the noise by per-pixel `N_obs * kappa / N_bar^2`, both from
    `fit_transfer_and_noise`. Neither takes `theta`.

    `A` still depends on the linearization point through `gn_inv`, hence the
    Gauss-Newton passes. What the operator omits is the bias sigmoid, so a
    subsequent `refine_theta` leaves `log_T` — the sigmoid's width — biased by
    several sigma.

    Staying in `xlm` space rather than `dm` space keeps the prior term exactly
    `I` and `Cl_dm` out of the solve.

    Follow with `refine_theta`, which is where `theta` comes from.
    """
    b_eff, kappa = fit_transfer_and_noise(model)

    mask_row = jnp.asarray(model.mask)[None, :]
    dg_obs = jnp.where(mask_row, jnp.asarray(model.dg_obs), 0.0)
    transfer = jnp.asarray(b_eff)[:, None]

    n_bar = np.asarray(model.N_bar)[:, None]
    ng_obs = np.round((np.asarray(model.dg_obs) + 1.0) * n_bar)
    # Off-mask pixels are floored to 1 rather than 0 only to keep the reciprocal
    # finite; `weight` zeroes them anyway.
    var = jnp.asarray(
        np.maximum(np.where(np.asarray(model.mask), ng_obs, 1.0), 1.0)
        * kappa[:, None]
        / n_bar**2
    )

    def forward(xlm: XlmParams) -> jnp.ndarray:
        """Predict `dg` from `xlm` with no `theta` anywhere."""
        return transfer * model.xlm_to_dm(xlm)

    def weight(residual: jnp.ndarray) -> jnp.ndarray:
        """Apply `N^-1`, as a select so off-mask values cannot leak in."""
        return jnp.where(mask_row, residual / var, 0.0)

    def linearize_at(xlm: XlmParams) -> tuple:
        """Build `A`, `A^T` and the solver at a linearization point."""
        pred, jvp = jax.linearize(forward, xlm)
        _, vjp = jax.vjp(forward, xlm)
        jvp = jax.jit(jvp)
        vjp_fn = jax.jit(lambda u: vjp(u)[0])

        def solve(rhs: jnp.ndarray) -> XlmParams:
            """Solve `(I + A^T N^-1 A) x = A^T N^-1 rhs`, matrix-free."""

            def matvec(x: XlmParams) -> XlmParams:
                data_term = vjp_fn(weight(jvp(x)))
                return XlmParams(
                    real=x.real + data_term.real, imag=x.imag + data_term.imag
                )

            out, _ = cg(matvec, vjp_fn(weight(rhs)), maxiter=cg_maxiter, tol=cg_tol)
            return out

        return pred, jvp, solve

    print(
        f"  theta-free CG: {n_gauss_newton} Gauss-Newton passes, "
        f"{n_gauss_newton + 2} solves, cg_maxiter={cg_maxiter}"
    )
    # Gauss-Newton on the Wiener mean. The linearized data vector is
    # d - f(x0) + A x0, which reduces to d when the model is already linear.
    xlm = _seed_xlm(model)
    for _ in range(n_gauss_newton):
        pred, jvp, solve = linearize_at(xlm)
        xlm = solve(dg_obs - pred + jvp(xlm))
    pred, jvp, solve = linearize_at(xlm)
    mean = solve(dg_obs - pred + jvp(xlm))

    # Constrained realization: the mean alone suppresses every mode the data
    # does not constrain, whereas sampling needs them at prior amplitude.
    omega_key, eta_key = jax.random.split(key)
    omega = model.make_random_xlm(omega_key)
    eta = jnp.sqrt(var) * jax.random.normal(eta_key, shape=dg_obs.shape)
    omega_wiener = solve(jvp(omega) + eta)
    draw = XlmParams(
        real=mean.real + omega.real - omega_wiener.real,
        imag=mean.imag + omega.imag - omega_wiener.imag,
    )

    rms = float(
        jnp.sqrt(
            jnp.mean(jnp.concatenate([draw.real.ravel() ** 2, draw.imag.ravel() ** 2]))
        )
    )
    print(f"  draw rms {rms:.4f} (prior: 1.0)")
    if not np.isfinite(rms):
        raise InfeasibleInitError(
            "init_xlm_theta_free: the draw is non-finite. The transfer fit gave "
            f"b_eff={np.array2string(b_eff, precision=3)}, "
            f"kappa={np.array2string(kappa, precision=3)}; a b_eff far from ~1.7, or a "
            "kappa pinned at a bound, points at the fit rather than the solve."
        )
    return draw
