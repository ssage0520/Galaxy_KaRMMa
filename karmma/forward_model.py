import time
from datetime import timedelta

import blackjax
import healpy as hp
import jax
import jax.flatten_util

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.stats as jst
import numpy as np
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState
from jax.scipy.sparse.linalg import cg
from scipy.special import legendre_p_all, roots_legendre

from karmma.structs import (
    KarmmaPosition,
    MCLMCInfo,
    ThetaParams,
    WhitenedKarmmaPosition,
    XlmParams,
)
from karmma.transforms import alm2map, map2alm

_INVGAMMA_ALPHA_R = 1.0  # TODO: expose in McmcConfig
_INVGAMMA_BETA_R = (5.0 / 8.0) ** 2.0  # TODO: expose in McmcConfig


class ForwardModel:
    def __init__(
        self,
        dg_obs,
        mask,
        CL,
        alpha,
        beta,
        N_bar=None,
        infer_theta=False,
        theta_fixed=None,
        lmax=None,
        gen_lmax=None,
        pixwin=None,
    ):
        self.dg_obs = dg_obs
        self.mask = mask.astype(bool)

        self.alpha = alpha
        self.beta = beta

        self.CL = CL

        self.Nbins = dg_obs.shape[0]
        self.Nside = hp.get_nside(self.dg_obs[0])
        self.pixel_size = float(hp.nside2resol(self.Nside))
        self.map_shape = dg_obs.shape

        self.infer_theta = infer_theta

        self.N_bar = np.asarray(N_bar)
        self.Ng_obs = np.round(
            (dg_obs[:, self.mask] + 1.0) * self.N_bar[:, None]
        ).astype(np.int32)

        self.theta_fixed = theta_fixed if not infer_theta else None
        self.lmax = 2 * self.Nside if lmax is None else lmax
        self.gen_lmax = 3 * self.Nside - 1 if gen_lmax is None else gen_lmax

        self.ell, self.emm = hp.Alm.getlm(self.lmax)
        self.gen_ell, self.gen_emm = hp.Alm.getlm(self.gen_lmax)

        self.pixwin = pixwin

        # Eigenbasis whitening transform for theta, set by build_theta_reparam.
        self.V = None
        self.w = None
        self.theta0 = None

        self.compute_CL_G()

        # Precomputed so that get_xlm can be jit compiled — numpy, not jnp,
        # so these static index constants are never confused with JAX tracers.
        self._real_idx = np.where(self.gen_ell > 1)[0]
        self._imag_idx = np.where((self.gen_ell > 1) & (self.gen_emm > 0))[0]
        self.n_modes = len(self._real_idx) + len(self._imag_idx)

    def _compute_CL_G_binpair(self, i, j, ell_array, P_ell, w):
        weighted_CL = (2 * ell_array + 1) * self.CL[i, j]
        xi_NG = weighted_CL @ P_ell / (4 * np.pi)
        xi_G = np.log(1 + xi_NG / (self.beta[i] * self.beta[j])) / (
            self.alpha[i] * self.alpha[j]
        )
        weighted_xi_G = w * xi_G
        CL_G_ij = 2 * np.pi * (P_ell @ weighted_xi_G)
        CL_G_ij[:2] = 1e-20 if i == j else 0.0
        return CL_G_ij

    def compute_CL_G(self, order=2):
        mu, w = roots_legendre(order * self.gen_lmax)
        ell_array = np.arange(self.gen_lmax + 1)
        P_ell = legendre_p_all(self.gen_lmax, mu).squeeze()
        self.CL_G = np.zeros_like(self.CL)
        for i in range(self.Nbins):
            for j in range(i + 1):
                self.CL_G[i, j, :] = self._compute_CL_G_binpair(
                    i, j, ell_array, P_ell, w
                )
                if i != j:
                    self.CL_G[j, i] = self.CL_G[i, j]
        CL_T = np.moveaxis(self.CL_G, 2, 0)
        L_T = np.linalg.cholesky(CL_T)
        self.L_G = np.moveaxis(L_T, 0, 2)

    def get_xlm(self, xlm: XlmParams):
        _real = jnp.zeros((self.Nbins, len(self.gen_ell)), dtype=jnp.float64)
        _imag = jnp.zeros_like(_real)
        _real = _real.at[:, self._real_idx].set(xlm.real)
        _imag = _imag.at[:, self._imag_idx].set(xlm.imag)
        return _real + 1j * _imag

    def apply_CL_G(self, xlm):
        L_expanded = self.L_G[:, :, self.gen_ell]
        ylm_real = jnp.einsum("ijm,jm->im", L_expanded, xlm.real) / jnp.sqrt(2)
        ylm_imag = jnp.einsum("ijm,jm->im", L_expanded, xlm.imag) / jnp.sqrt(2)
        ylm_real = jnp.where(self.gen_emm == 0, ylm_real * jnp.sqrt(2), ylm_real)
        ylm_imag = jnp.where(self.gen_emm == 0, 0.0, ylm_imag)
        return ylm_real + 1j * ylm_imag

    def x2deff(self, xlm: XlmParams, theta: ThetaParams):
        xlm_full = self.get_xlm(xlm)
        ylm = self.apply_CL_G(xlm_full)

        ys = alm2map(ylm, self.Nside, self.gen_lmax)
        dm = self.beta[:, None] * (
            jnp.exp(self.alpha[:, None] * ys - 0.5 * self.alpha[:, None] ** 2) - 1
        )
        dm_lm = map2alm(dm, self.lmax)
        b_ell = jnp.exp(
            -0.5
            * self.ell
            * (self.ell + 1)
            * (jnp.exp(theta.log_R[:, None]) * self.pixel_size) ** 2
        )
        filt = (1.0 + theta.c[:, None] * b_ell) * (
            self.pixwin[self.ell] if self.pixwin is not None else 1.0
        )
        return alm2map(dm_lm * filt, self.Nside, self.lmax)

    def x2dm(self, xlm: XlmParams):
        xlm_full = self.get_xlm(xlm)
        ylm = self.apply_CL_G(xlm_full)

        ys = alm2map(ylm, self.Nside, self.gen_lmax)
        dm = self.beta[:, None] * (
            jnp.exp(self.alpha[:, None] * ys - 0.5 * self.alpha[:, None] ** 2) - 1
        )
        dm_lm = map2alm(dm, self.lmax)
        if self.pixwin is not None:
            dm_lm = dm_lm * self.pixwin[self.ell]
        return alm2map(dm_lm, self.Nside, self.lmax)

    def dm_to_binom_params(self, deff, theta: ThetaParams):
        A_t = theta.A_t[:, np.newaxis]
        T = jnp.exp(theta.log_T)[:, np.newaxis]
        mu0 = theta.mu0[:, np.newaxis]
        a = theta.a[:, np.newaxis]
        N_bar = self.N_bar[:, np.newaxis]

        A = jnp.log1p(deff)
        sig = jax.nn.sigmoid((A - A_t) / T)

        deff_b = deff[:, self.mask]
        sig_b = sig[:, self.mask]
        b = 1.0 / jnp.mean((1 + deff_b) * sig_b, axis=1)

        mean_Ng = b[:, np.newaxis] * (1 + deff) * sig * N_bar

        mu = mu0 + a * deff
        A_prime = A - mu
        deff_prime = jnp.expm1(A_prime)
        sig_prime = jax.nn.sigmoid((A_prime - A_t) / T)
        mean_Ng_prime = b[:, np.newaxis] * (1 + deff_prime) * sig_prime * N_bar

        p = jnp.clip(mean_Ng - mean_Ng_prime, 1e-6, 1 - 1e-6)
        n = mean_Ng / p

        return n, p

    def make_random_xlm(self, key):
        rk, ik = jax.random.split(key)
        return XlmParams(
            real=jax.random.normal(
                rk, shape=(self.Nbins, len(self._real_idx)), dtype=jnp.float64
            ),
            imag=jax.random.normal(
                ik, shape=(self.Nbins, len(self._imag_idx)), dtype=jnp.float64
            ),
        )

    def log_prob(self, params: KarmmaPosition):
        theta = params.theta if self.infer_theta else self.theta_fixed

        deff = self.x2deff(params.xlm, theta)

        n, p = self.dm_to_binom_params(deff, theta)

        n_m = n[:, self.mask]
        p_m = p[:, self.mask]

        log_lik = jnp.sum(jax.scipy.stats.binom.logpmf(self.Ng_obs, n_m, p_m))

        log_prior_real = jnp.sum(jst.norm.logpdf(params.xlm.real, loc=0.0, scale=1.0))
        log_prior_imag = jnp.sum(jst.norm.logpdf(params.xlm.imag, loc=0.0, scale=1.0))

        log_jacobian_theta = 0.0
        log_prior_theta = 0.0
        if self.infer_theta:
            log_jacobian_theta = (
                jnp.sum(theta.log_T)  # log_T -> T
                + jnp.sum(theta.log_R)  # log_R -> R
            )
            log_prior_theta = (
                # InvGamma(alpha, beta) prior on R^2, sampled as log_R.
                +jnp.sum(theta.log_R)  # Jacobian: R^2 -> R
                - 2.0
                * (1.0 + _INVGAMMA_ALPHA_R)
                * jnp.sum(theta.log_R)  # InvGamma log-prior
                - _INVGAMMA_BETA_R
                * jnp.sum(jnp.exp(-2.0 * theta.log_R))  # InvGamma log-prior
            )

        return (
            log_prior_real
            + log_prior_imag
            + log_jacobian_theta
            + log_prior_theta
            + log_lik
        )

    def dense_theta_imm(
        self,
        position: KarmmaPosition,
        tol: float = 1e-3,
        maxiter: int = 300,
        kappa_max: float = 1e9,
        verbose: bool = True,
    ) -> np.ndarray:
        """Dense theta-only covariance-like matrix via Schur complement + CG.

        Marginalises over the xlm block with n_theta CG solves against H_xx,
        then fixes the resulting indefinite n_theta×n_theta Schur complement
        to PD via |λ| eigenvalue correction. Used directly for eigenbasis
        reparametrization (see build_theta_reparam), or reduced to a
        diagonal by initialize_imm for a diagonal-only IMM seed instead.

        Requires `infer_theta=True` (theta must be part of the sampled
        position for the Schur complement over the theta block to apply).

        Returns
        -------
        np.ndarray of shape (n_theta, n_theta), field-major/bin-minor layout
        matching jax.flatten_util.ravel_pytree(ThetaParams(...)).
        """
        n_theta = len(ThetaParams._fields) * self.Nbins

        # ravel_pytree matches BlackJax's pytree flattening: xlm-first, theta-last.
        flat_pos, unravel_fn = jax.flatten_util.ravel_pytree(position)
        N_full = flat_pos.shape[0]
        n_x = N_full - n_theta

        def _flat_log_prob(flat):
            return self.log_prob(unravel_fn(flat))

        @jax.jit
        def _hvp(v):
            _, g = jax.jvp(jax.grad(_flat_log_prob), (flat_pos,), (v,))
            return -g

        @jax.jit
        def _hvp_xx(vx):
            v_full = jnp.zeros(N_full).at[:n_x].set(vx)
            return _hvp(v_full)[:n_x]

        if verbose:
            print(
                f"dense_theta_imm: step 1 — {n_theta} b-indicator HVPs ...", flush=True
            )
        rows_b = jnp.stack(
            [_hvp(jnp.zeros(N_full).at[n_x + i].set(1.0)) for i in range(n_theta)]
        )
        H_bb_est = rows_b[:, n_x:]
        H_bx_est = rows_b[:, :n_x]

        if verbose:
            n_finite = int(jnp.sum(jnp.all(jnp.isfinite(rows_b), axis=1)))
            abs_rows_b = jnp.abs(rows_b)
            print(
                f"  HVP finiteness: {n_finite}/{n_theta} finite  |  "
                f"|HVP| range: min={abs_rows_b.min():.2e} max={abs_rows_b.max():.2e}"
            )
            print(
                f"dense_theta_imm: step 2 — {n_theta} CG solves "
                f"(tol={tol}, maxiter={maxiter}) ...",
                flush=True,
            )
        X = jnp.stack(
            [
                cg(_hvp_xx, H_bx_est[j], tol=tol, maxiter=maxiter)[0]
                for j in range(n_theta)
            ]
        )

        precision_bb = H_bb_est - H_bx_est @ X.T

        if verbose:
            evals = np.array(jnp.linalg.eigvalsh(precision_bb))
            resid = np.array(
                jax.vmap(
                    lambda x, r: jnp.linalg.norm(_hvp_xx(x) - r) / jnp.linalg.norm(r)
                )(X, H_bx_est)
            )
            print(f"  CG rel residuals: max={resid.max():.2e}  mean={resid.mean():.2e}")
            print(
                f"  Schur eigenvalues: min={evals.min():.4e}  max={evals.max():.4e}  "
                f"negative={np.sum(evals < 0)}"
            )

        S = 0.5 * (precision_bb + precision_bb.T)
        w, U = jnp.linalg.eigh(S)
        w_fixed = jnp.clip(jnp.abs(w), min=float(jnp.max(jnp.abs(w))) / kappa_max)

        return np.array((U / w_fixed) @ U.T)

    def initialize_imm(
        self,
        position: KarmmaPosition,
        tol: float = 1e-3,
        maxiter: int = 300,
        kappa_max: float = 1e9,
        verbose: bool = True,
    ) -> np.ndarray:
        """Diagonal IMM for NUTS warm-start via Schur complement + CG.

        xlm block  → 1.0  (consistent with the N(0,1) prior)
        theta block → diagonal of dense_theta_imm's dense matrix

        Requires `infer_theta=True` (theta must be part of the sampled
        position for the Schur complement over the theta block to apply).

        Returns
        -------
        np.ndarray of shape (n_x + n_theta,) in BlackJax pytree-flat layout:
            [xlm.real.ravel(), xlm.imag.ravel(), theta fields in ThetaParams order]
        """
        n_theta = len(ThetaParams._fields) * self.Nbins
        n_x = jax.flatten_util.ravel_pytree(position)[0].shape[0] - n_theta
        dense = self.dense_theta_imm(position, tol, maxiter, kappa_max, verbose)
        return np.concatenate([np.ones(n_x), np.diag(dense)])

    def build_theta_reparam(
        self, dense_theta_matrix: np.ndarray, theta0: ThetaParams
    ) -> None:
        """Eigendecomposes a dense theta covariance estimate (e.g.
        self.dense_theta_imm(...)'s output) and stores the whitening
        transform (self.V, self.w, self.theta0) for theta_to_phi/
        phi_to_theta and sample(), which requires this to have been called
        first."""
        self.w, self.V = jnp.linalg.eigh(jnp.asarray(dense_theta_matrix))
        self.theta0 = theta0

    def theta_to_phi(self, theta: ThetaParams) -> jnp.ndarray:
        """Physical theta -> whitened phi, via the eigenbasis transform set
        by build_theta_reparam."""
        theta_flat, _ = jax.flatten_util.ravel_pytree(theta)
        theta0_flat, _ = jax.flatten_util.ravel_pytree(self.theta0)
        return (self.V.T @ (theta_flat - theta0_flat)) / jnp.sqrt(self.w)

    def phi_to_theta(self, phi: jnp.ndarray) -> ThetaParams:
        """Whitened phi -> physical theta, via the eigenbasis transform set
        by build_theta_reparam."""
        theta0_flat, unravel = jax.flatten_util.ravel_pytree(self.theta0)
        theta_flat = theta0_flat + self.V @ (phi * jnp.sqrt(self.w))
        return unravel(theta_flat)

    def sample(
        self,
        key,
        num_samples,
        initial_position: WhitenedKarmmaPosition,
        initial_imm: np.ndarray,
        frac_tune1: float = 0.1,
        frac_tune2: float = 0.3,
        frac_tune3: float = 0.1,
        l_factor: float = 0.4,
        desired_energy_var: float = 5e-4,
        thinning_warmup: int = 5,
        thinning_sampling: int = 5,
    ):
        """Runs MCLMC, seeding its diagonal preconditioner from `initial_imm`.

        Requires the jax_karmma_dev blackjax build (mclmc's call-time kernel
        args, mclmc_find_L_and_step_size's logdensity_fn/l_factor kwargs).
        `initial_imm` must be a 1-D diagonal IMM in BlackJax pytree-flat
        layout, e.g. as returned by `initialize_imm`. `num_samples` is the
        number of samples actually saved (post-thinning), not a raw
        integrator-step budget.

        Warmup runs as two chained `mclmc_find_L_and_step_size` calls, since
        phases 1+2 and phase 3 want opposite thinning:

        - Phases 1+2 (step size + diagonal IMM) always run unthinned — they
          only carry O(dim) running accumulators, so thinning gains nothing
          and only coarsens the step-size feedback.
        - Phase 3 (L via effective-sample-size) is thinned by
          `thinning_warmup`, since its FFT-based ESS calculation over the
          full position is what actually blows up memory at scale, while its
          estimate is fairly insensitive to draw spacing.

        Call 2 is seeded from call 1's tuned `state`/`params` (not
        `init_state`/`initial_params`) so phase 3 continues the chain instead
        of restarting cold.
        """
        if self.V is None:
            raise ValueError("Call build_theta_reparam(...) before sample().")

        def log_prob(params: WhitenedKarmmaPosition):
            theta = self.phi_to_theta(params.phi)
            return self.log_prob(KarmmaPosition(xlm=params.xlm, theta=theta))

        # jax.jit matters here since mclmc.init isn't itself jit-decorated;
        # elsewhere the enclosing lax.scan compiles everything regardless.
        log_prob = jax.jit(log_prob)
        dim = blackjax.util.pytree_size(initial_position)

        t0 = time.perf_counter()

        key, key_init, key_warmup1, key_warmup2, key_sample = jax.random.split(key, 5)

        def sample_info(info):
            """Aggregates raw per-step MCLMC info over a thinning block.

            logdensity takes the last raw step's value — exactly matching
            the block's final (saved) position — rather than an aggregate;
            energy_change/kinetic_change are genuinely mean-zero step-error
            diagnostics where an RMS magnitude is meaningful, but RMS-ing
            logdensity (uniformly large-magnitude and negative) collapses to
            |logdensity|, silently flipping its sign. nonans is a 0/1
            indicator, so it needs a mean (fraction clean), not RMS — RMS of
            a 0/1 array is sqrt(fraction), not the fraction itself.
            """
            return info._replace(
                logdensity=info.logdensity[-1],
                energy_change=(info.energy_change**2).mean() ** 0.5,
                kinetic_change=(info.kinetic_change**2).mean() ** 0.5,
                nonans=info.nonans.mean(),
            )

        init_state = blackjax.mcmc.mclmc.init(
            position=initial_position, logdensity_fn=log_prob, rng_key=key_init
        )
        initial_params = MCLMCAdaptationState(
            L=jnp.sqrt(dim),
            step_size=jnp.sqrt(dim) * 0.25,
            inverse_mass_matrix=initial_imm,
        )

        # desired_energy_var only matters via mclmc_find_L_and_step_size's own
        # kwarg (the step-size dual-averaging target) — build_kernel's use of
        # it is dead unless desired_energy_var_max_ratio is also set, which
        # we don't do, so it's passed only to the calls below.

        # Call 1: phases 1+2 only, always unthinned — raw kernel, since
        # thinning=1 needs no thin_kernel wrapper.
        print()
        with blackjax.progress_bar(label="Phases 1+2 (step size + IMM)"):
            state_12, params_12, warmup_calls_12 = blackjax.mclmc_find_L_and_step_size(
                mclmc_kernel=blackjax.mcmc.mclmc.build_kernel(
                    integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
                ),
                logdensity_fn=log_prob,
                num_steps=round(num_samples * thinning_sampling),
                state=init_state,
                rng_key=key_warmup1,
                diagonal_preconditioning=True,
                frac_tune1=frac_tune1,
                frac_tune2=frac_tune2,
                frac_tune3=0.0,
                desired_energy_var=desired_energy_var,
                params=initial_params,
                l_factor=l_factor,
            )
            # Forces real synchronization before the with-block exits — otherwise
            # JAX's async dispatch lets this return (and the progress bar close,
            # stamped at 100%) long before the tuning scan has actually finished.
            jax.block_until_ready((state_12, params_12))

        imm_12 = np.array(params_12.inverse_mass_matrix)
        step_size_12_finite = bool(np.isfinite(params_12.step_size))
        imm_12_finite = bool(np.all(np.isfinite(imm_12)))
        imm_12_positive = bool(np.all(imm_12 > 0))
        flat_init, _ = jax.flatten_util.ravel_pytree(initial_position)
        flat_state_12, _ = jax.flatten_util.ravel_pytree(state_12.position)
        max_delta_12 = float(jnp.max(jnp.abs(flat_state_12 - flat_init)))

        print(
            f"[Phases 1+2] Tuned step size: {params_12.step_size:.5f}  "
            f"(finite={step_size_12_finite})"
        )
        print(
            f"[Phases 1+2] Inv. mass matrix: min={imm_12.min():.3e}  "
            f"mean={imm_12.mean():.3e}  max={imm_12.max():.3e}  "
            f"(finite={imm_12_finite}, all_positive={imm_12_positive})"
        )
        print(f"[Phases 1+2] Max |Δ position| from init: {max_delta_12:.3e}")

        # Call 2: phase 3 only, thinned by thinning_warmup, seeded from
        # call 1's state/params so it continues the tuned chain.
        print()
        with blackjax.progress_bar(label="Phase 3 (L via ESS)"):
            tuned_state, tuned_params, warmup_calls_3 = blackjax.mclmc_find_L_and_step_size(
                mclmc_kernel=blackjax.util.thin_kernel(
                    blackjax.mcmc.mclmc.build_kernel(
                        integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
                    ),
                    thinning=thinning_warmup,
                    info_transform=sample_info,
                ),
                logdensity_fn=log_prob,
                num_steps=round(num_samples * thinning_sampling / thinning_warmup),
                state=state_12,
                rng_key=key_warmup2,
                diagonal_preconditioning=True,
                frac_tune1=0.0,
                frac_tune2=0.0,
                frac_tune3=frac_tune3,
                desired_energy_var=desired_energy_var,
                params=params_12,
                l_factor=l_factor * thinning_warmup,
            )
            # See the analogous comment on Call 1 — same reason.
            jax.block_until_ready((tuned_state, tuned_params))

        t1 = time.perf_counter()
        print()

        # Confirms phase 3 didn't reset step_size/IMM (printed, not
        # asserted, so it can't crash a run).
        step_size_preserved = bool(
            np.array(tuned_params.step_size) == np.array(params_12.step_size)
        )
        imm_preserved = bool(
            np.array_equal(
                np.array(tuned_params.inverse_mass_matrix),
                np.array(params_12.inverse_mass_matrix),
            )
        )
        print(
            f"step_size/IMM unchanged by phase 3: "
            f"step_size={step_size_preserved}  inverse_mass_matrix={imm_preserved}"
        )

        warmup_calls = warmup_calls_12 + warmup_calls_3
        warmup_integration_steps = (
            warmup_calls_12 * 1 + warmup_calls_3 * thinning_warmup
        )
        imm = np.array(tuned_params.inverse_mass_matrix)

        L_finite = bool(np.isfinite(tuned_params.L))
        print(f"Warmup time: {timedelta(seconds=int(t1 - t0))}")
        print(f"Tuned L: {tuned_params.L:.4f}  (finite={L_finite})")
        print(f"Tuned step size: {tuned_params.step_size:.5f}")
        print(
            f"Steps per trajectory (L / step_size): "
            f"{tuned_params.L / tuned_params.step_size:.2f}"
        )
        print(
            f"Warmup calls (thinned): {warmup_calls}  "
            f"(phases 1+2: {warmup_calls_12}, phase 3: {warmup_calls_3})  |  "
            f"raw integration steps: {warmup_integration_steps}"
        )
        print(
            f"Inv. mass matrix: min={imm.min():.3e}  mean={imm.mean():.3e}  max={imm.max():.3e}"
        )

        # Bakes the tuned L/step_size/IMM into a fixed SamplingAlgorithm;
        # thin_algorithm (not thin_kernel) wraps that.
        mclmc_sampler = blackjax.mclmc(
            logdensity_fn=log_prob,
            L=tuned_params.L,
            step_size=tuned_params.step_size,
            inverse_mass_matrix=tuned_params.inverse_mass_matrix,
        )
        thinned_sampling_alg = blackjax.util.thin_algorithm(
            mclmc_sampler, thinning=thinning_sampling, info_transform=sample_info
        )

        print()
        with blackjax.progress_bar(label="Sampling"):
            _, (states, infos) = blackjax.util.run_inference_algorithm(
                rng_key=key_sample,
                inference_algorithm=thinned_sampling_alg,
                num_steps=num_samples,
                initial_state=tuned_state,
                transform=lambda state, info: (
                    state.position,
                    MCLMCInfo(
                        logdensity=info.logdensity,
                        energy_change=info.energy_change,
                        kinetic_change=info.kinetic_change,
                        nonans=info.nonans,
                    ),
                ),
            )
            # See the analogous comment on Call 1 — same reason.
            jax.block_until_ready((states, infos))

        t2 = time.perf_counter()
        print()

        print(f"Sampling time:    {timedelta(seconds=int(t2 - t1))}")
        print(f"Total time (w+s): {timedelta(seconds=int(t2 - t0))}")
        print(
            f"Samples saved:    {num_samples}  "
            f"(thinned by {thinning_sampling} for sampling, {thinning_warmup} for warmup phase 3 only)"
        )
        print(
            f"Mean |energy change| (RMS-thinned): {np.array(infos.energy_change).mean():.4e}"
        )
        print(f"Fraction of non-NaN steps: {np.array(infos.nonans).mean():.4f}")

        theta = jax.vmap(self.phi_to_theta)(states.phi)
        states = KarmmaPosition(xlm=states.xlm, theta=theta)

        return states, infos, tuned_params, warmup_calls
