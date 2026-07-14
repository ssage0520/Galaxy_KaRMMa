import healpy as hp
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.stats as jst
import numpy as np
from scipy.special import legendre_p_all, roots_legendre

from karmma.structs import (
    KarmmaPosition,
    ThetaParams,
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

