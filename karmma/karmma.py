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
from blackjax.adaptation.base import get_filter_adapt_info_fn
from jax.scipy.sparse.linalg import cg
from scipy.special import legendre_p_all, roots_legendre

from karmma.structs import (
    KarmmaPosition,
    NUTSInfo,
    ThetaParams,
    WhitenedKarmmaPosition,
    XlmParams,
)
from karmma.transforms import alm2map, map2alm

_INVGAMMA_ALPHA_R = 1.0  # TODO: expose in McmcConfig
_INVGAMMA_BETA_R = (5.0 / 8.0) ** 2.0  # TODO: expose in McmcConfig


class KarmmaSampler:
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
        w_fixed = jnp.clip(jnp.abs(w), a_min=float(jnp.max(jnp.abs(w))) / kappa_max)

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
        num_warmup,
        num_samples,
        initial_position: WhitenedKarmmaPosition,
        initial_imm: np.ndarray,
        imm_shrinkage_to_previous: float = 0.0,
        step_size=0.05,
        target_acceptance_rate=0.65,
    ):
        """Runs NUTS, seeding window_adaptation's inverse mass matrix.

        Requires a blackjax build with `initial_inverse_mass_matrix` /
        `imm_shrinkage_to_previous` support in `window_adaptation` (the
        jax_karmma_dev environment) — stock blackjax (jax_karmma) raises a
        TypeError.

        `initial_imm` must be a 1-D diagonal IMM in BlackJax pytree-flat
        layout, e.g. as returned by `initialize_imm`.

        Always samples theta in the whitened eigenbasis (see
        build_theta_reparam, which must be called first): `initial_position`
        is a WhitenedKarmmaPosition (xlm + phi), and the returned `states` is
        converted back to a physical-coordinate KarmmaPosition (theta, not
        phi) before returning, so callers never see phi-space values.
        `mcmc_parameters["inverse_mass_matrix"]`'s theta block is phi-space-
        scaled (expected, not a bug — needed to interpret it as a diagnostic
        later), but `infos.logdensity`/`winfo`'s log-density values remain in
        true physical units regardless, since the wrapped log_prob below
        evaluates `self.log_prob` at the untransformed point, adding no
        constant (the phi -> theta map is linear with a phi-independent
        Jacobian, which doesn't affect NUTS/HMC dynamics or acceptance).
        """
        if self.V is None:
            raise ValueError("Call build_theta_reparam(...) before sample().")

        def log_prob(params: WhitenedKarmmaPosition):
            theta = self.phi_to_theta(params.phi)
            return self.log_prob(KarmmaPosition(xlm=params.xlm, theta=theta))

        log_prob = jax.jit(log_prob)

        t0 = time.perf_counter()

        filter_fn = get_filter_adapt_info_fn(
            info_keys={"acceptance_rate", "is_divergent", "num_integration_steps"}
        )

        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            logdensity_fn=log_prob,
            initial_step_size=step_size,
            initial_inverse_mass_matrix=initial_imm,
            imm_shrinkage_to_previous=imm_shrinkage_to_previous,
            target_acceptance_rate=target_acceptance_rate,
            is_mass_matrix_diagonal=True,
            progress_bar=True,
            adaptation_info_fn=filter_fn,
        )
        key, warmup_key = jax.random.split(key)
        print()
        (wstate, parameters), winfo = warmup.run(
            warmup_key, initial_position, num_steps=num_warmup
        )

        wstate.position.xlm.real.block_until_ready()
        t1 = time.perf_counter()
        print()

        warmup_steps = np.array(winfo.info.num_integration_steps)
        time_per_leapfrog = (t1 - t0) / warmup_steps.sum()
        mean_steps_end = warmup_steps[-20:].mean()
        time_per_sample = time_per_leapfrog * mean_steps_end
        est_sampling_time = num_samples * time_per_sample

        print(f"Warmup time: {timedelta(seconds=int(t1 - t0))}")
        print(f"Adapted step size: {parameters['step_size']:.4f}")
        print(
            f"Mean integration steps (warmup): {warmup_steps.mean():.1f}  |  last 20: {mean_steps_end:.1f}"
        )
        print(
            f"Mean acceptance rate (warmup): {jnp.mean(winfo.info.acceptance_rate):.4f}"
        )
        print(f"Number of divergences (warmup): {jnp.sum(winfo.info.is_divergent)}")
        print(
            f"Estimated sampling time: ~{timedelta(seconds=int(est_sampling_time))}  ({num_samples} samples × ~{time_per_sample:.1f}s/sample)"
        )

        nuts = blackjax.nuts(log_prob, **parameters)

        key, sample_key = jax.random.split(key)
        print()
        _, (states, infos) = blackjax.util.run_inference_algorithm(
            rng_key=sample_key,
            inference_algorithm=nuts,
            num_steps=num_samples,
            initial_state=wstate,
            progress_bar=True,
            transform=lambda state, info: (
                state.position,
                NUTSInfo(
                    is_divergent=info.is_divergent,
                    num_integration_steps=info.num_integration_steps,
                    acceptance_rate=info.acceptance_rate,
                    energy=info.energy,
                    logdensity=state.logdensity,
                ),
            ),
        )
        states.xlm.real.block_until_ready()
        t2 = time.perf_counter()
        print()

        print(f"Sampling time:    {timedelta(seconds=int(t2 - t1))}")
        print(f"Total time (w+s): {timedelta(seconds=int(t2 - t0))}")
        print(
            f"Mean integration steps: {np.array(infos.num_integration_steps).mean():.1f}"
        )
        print(f"Mean acceptance rate: {jnp.mean(infos.acceptance_rate):.4f}")
        print(f"Number of divergences: {jnp.sum(infos.is_divergent)}")

        theta = jax.vmap(self.phi_to_theta)(states.phi)
        states = KarmmaPosition(xlm=states.xlm, theta=theta)

        return states, infos, parameters, winfo
