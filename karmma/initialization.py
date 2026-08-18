"""Construct a feasible, prior-typical initial `xlm` directly from the observed counts.

`ForwardModel.log_prob` is `-inf` unless `Ng_obs < n + 1` at every masked
pixel, and `n` grows with the local density, so the highest-count pixels demand
local overdensity. A prior draw fails that condition at ~0.6% of pixels, and so
does `xlm = 0`, which leaves nothing to shrink toward. This module builds a
starting point by running the forward model backwards from `dg_obs` instead.
"""

import warnings

import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq
from scipy.stats import kurtosis, skew

from karmma.forward_model import ForwardModel
from karmma.structs import ThetaParams, XlmParams
from karmma.transforms import map2alm

# Keeps `dg_obs.min() / b` strictly inside the transform's domain rather than
# exactly on its boundary, where G2's log diverges.
_FLOOR_MARGIN = 1e-3


class InfeasibleInitError(RuntimeError):
    """Raised when the constructed initial `xlm` falls outside the likelihood's support."""


def _target_ys_var(model: ForwardModel) -> np.ndarray:
    """Compute the variance the latent Gaussian field should have, from `CL_G`.

    Parameters
    ----------
    model : ForwardModel
        Model whose `CL_G` sets the target.

    Returns
    -------
    np.ndarray
        Per-bin variance, shape (Nbins,), `sum((2l + 1) * CL_G[i, i, l]) / (4 pi)`.
    """
    ell = np.arange(model.gen_lmax + 1)
    return np.array(
        [
            np.sum((2 * ell + 1) * model.CL_G[i, i]) / (4.0 * np.pi)
            for i in range(model.Nbins)
        ]
    )


def fit_bias(
    model: ForwardModel, b_max: float = 20.0, xtol: float = 1e-6
) -> np.ndarray:
    """Fit the per-bin scaling that makes `dg_obs / b` Gaussianize to unit variance.

    Parameters
    ----------
    model : ForwardModel
        Model supplying `dg_obs`, the point transform, and `CL_G`.
    b_max : float, optional
        Upper bracket for the root find, by default 20.0.
    xtol : float, optional
        Absolute tolerance on `b`, by default 1e-6.

    Returns
    -------
    np.ndarray
        Fitted bias per bin, shape (Nbins,).

    Notes
    -----
    Solves `var(gn(dg_obs_i / b_i)) == _target_ys_var(model)[i]` for each bin.
    `dg_obs` is a noisy tracer of `dm`, so dividing by `b` both undoes the
    galaxy bias and shrinks the shot-noise contribution; matching the variance
    `CL_G` predicts is what lands the resulting `xlm` at unit rms, i.e. on the
    prior's typical set.

    Choosing `b` this way rather than by maximizing `log_prob` is deliberate:
    the `xlm` prior term dominates the posterior (~1.67e6 of 2.11e6 for a
    2-bin nside-256 run) and keeps improving as the field shrinks, so
    `log_prob` has a spurious optimum at a badly cold start. Matching the
    variance and maximizing the *likelihood* alone agree with each other, and
    with the truth in mocks.

    The search is bracketed below by the smallest `b` keeping every pixel
    inside the transform's domain. That bound binds for G2, whose floor is
    `-beta`, well above the `dg_obs = -1` of an empty pixel.
    """
    dg_obs = np.asarray(model.dg_obs)
    target = _target_ys_var(model)

    # Lowest dm the transform accepts: -beta for G2, from gn's range; -1 for
    # G3, where log1p(deff) in deff_to_binom_params breaks.
    floor = (
        -np.asarray(model.lbda[1])
        if model.gn_order == 2
        else np.full(model.Nbins, -1.0)
    )
    # dm_min = dg_min / b must sit above floor, hence b > |dg_min| / |floor|.
    b_min = np.abs(dg_obs.min(axis=1)) / np.abs(floor) * (1.0 + _FLOOR_MARGIN)

    def _excess_var(b_i: float, i: int) -> float:
        """Return `var(gn(dg_obs_i / b_i))` minus bin `i`'s target variance."""
        ys = model.gn(dg_obs[i : i + 1] / b_i, model.gn_order, model.lbda[:, i : i + 1])
        return float(ys.var()) - target[i]

    b = np.zeros(model.Nbins)
    for i in range(model.Nbins):
        if _excess_var(b_min[i], i) <= 0.0:
            warnings.warn(
                f"fit_bias: bin {i} hits the transform floor before matching its "
                f"target variance, so b is pinned at {b_min[i]:.4f} and the "
                "initial field will be colder than the prior expects.",
                stacklevel=2,
            )
            b[i] = b_min[i]
            continue
        b[i] = brentq(_excess_var, b_min[i], b_max, args=(i,), xtol=xtol)
    return b


def init_xlm_from_data(
    model: ForwardModel,
    theta: ThetaParams,
    b: np.ndarray | None = None,
    rescale: bool = True,
    verbose: bool = True,
) -> XlmParams:
    """Build an initial `xlm` by running the forward model backwards from `dg_obs`.

    Parameters
    ----------
    model : ForwardModel
        Model supplying `dg_obs`, the point transform, and `L_G`.
    theta : ThetaParams
        Bias/nuisance parameters the run will start from. Only used to
        confirm the result lands inside the likelihood's support, but the
        check depends on them, so they must be the values actually used.
    b : np.ndarray or None, optional
        Per-bin bias, shape (Nbins,). Fitted with `fit_bias` when None
        (the default); pass a precomputed value to avoid refitting.
    rescale : bool, optional
        Whether to rescale `xlm` to exactly unit rms, by default True.
        The variance match leaves it ~5% low, which is ~80 sigma off the
        prior's typical-set radius at ~1e6 parameters.
    verbose : bool, optional
        Whether to print the fitted `b`, the rms, the support margin, and
        the Gaussianized field's moments, by default True.

    Returns
    -------
    XlmParams
        Initial harmonic coefficients, feasible under `model.log_prob`.

    Raises
    ------
    InfeasibleInitError
        If the result still violates the binomial support at any masked
        pixel, reporting the worst offender.

    Notes
    -----
    Pipeline: `dm = dg_obs / b` -> `gn` -> `map2alm` at `gen_lmax` ->
    `unapply_CL_G` -> `pack_xlm` -> rescale.

    Only multipoles below roughly `2 * Nside` are genuinely recovered; the
    HEALPix analysis above that is not invertible, so the higher modes come
    out at the right amplitude but nearly uncorrelated with the truth. That
    is the desired behaviour for a starting point, since the data cannot
    constrain them either — they should look like a prior draw.
    """
    if b is None:
        b = fit_bias(model)

    dg_obs = np.asarray(model.dg_obs)
    ys = model.gn(dg_obs / b[:, np.newaxis], model.gn_order, model.lbda)
    ylm = map2alm(jnp.asarray(ys), model.gen_lmax)
    xlm = model.pack_xlm(jnp.asarray(model.unapply_CL_G(ylm)))

    rms = float(
        jnp.sqrt(
            jnp.mean(jnp.concatenate([xlm.real.ravel() ** 2, xlm.imag.ravel() ** 2]))
        )
    )
    if rescale:
        xlm = XlmParams(real=xlm.real / rms, imag=xlm.imag / rms)

    deff = model.xlm_to_deff(xlm, theta)
    n, _ = model.deff_to_binom_params(deff, theta, mask_output=True)
    margin = np.asarray(n) + 1 - model.Ng_obs

    if verbose:
        print(f"xlm init: from data (b = {np.array2string(b, precision=4)})")
        print(f"  rms {rms:.4f}" + (" -> rescaled to 1.0" if rescale else ""))
        print(
            f"  support margin: min {margin.min():+.2f}, median {np.median(margin):.1f}"
        )
        print(
            f"  Gaussianized field: skew {np.array2string(skew(ys, axis=1), precision=3)}"
            f", excess kurtosis {np.array2string(kurtosis(ys, axis=1), precision=3)}"
        )
        print(
            f"    (both are 0 for an exact G{model.gn_order} fit and noiseless data; "
            "large values mean the point transform is misspecified)"
        )

    n_bad = int((margin <= 0).sum())
    if n_bad:
        worst = np.unravel_index(int(np.argmin(margin)), margin.shape)
        deff_masked = np.asarray(deff)[:, model.mask]
        raise InfeasibleInitError(
            f"Initial xlm falls outside the likelihood's support: {n_bad} of "
            f"{margin.size} masked pixel-bin terms have Ng_obs >= n + 1, making "
            f"log_prob -inf.\n"
            f"  worst margin {margin[worst]:+.2f} at bin {worst[0]}, masked pixel "
            f"{worst[1]}: Ng_obs={model.Ng_obs[worst]}, n={float(n[worst]):.3f}, "
            f"deff={deff_masked[worst]:+.4f}\n"
            f"  fitted b = {np.array2string(b, precision=4)}, xlm rms = {rms:.4f}\n"
            "  There is no safe automatic fallback here: xlm = 0 is itself "
            "infeasible, so shrinking the field further makes this worse. Check "
            "that theta is sensible (mu0/a set n ~ 1/mu) and that CL and lbda "
            "match the data."
        )

    return xlm
