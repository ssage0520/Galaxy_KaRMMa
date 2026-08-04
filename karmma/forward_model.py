"""Forward model: xlm harmonic coefficients to galaxy counts, and the posterior log-density."""

import functools

import healpy as hp
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.stats as jst
import numpy as np
from scipy.special import legendre_p_all, roots_legendre

from karmma.structs import KarmmaPosition, ThetaParams, XlmParams
from karmma.transforms import alm2map, map2alm

_INVGAMMA_ALPHA_R = 1.0  # TODO: expose in McmcConfig
_INVGAMMA_BETA_R = (5.0 / 8.0) ** 2.0  # TODO: expose in McmcConfig


class _KmaxExceeded(Exception):
    """Raised by generate_mock_dg_obs; caught by make_random_mock to redraw xlm."""


class ForwardModel:
    """Forward model from harmonic-space xlm coefficients to the galaxy-count log-density.

    Attributes
    ----------
    dg_obs : np.ndarray
        Observed galaxy overdensity maps, shape (Nbins, npix).
    mask : np.ndarray
        Survey mask, shape (npix,); cast to bool.
    lbda : np.ndarray
        Point-transform parameters, shape (gn_order, Nbins): rows (alpha,
        beta) for gn_order=2 (shifted lognormal), or (a, b, c) for
        gn_order=3. One column per tomographic bin.
    gn_order : int
        Point-transform order in use for this model — 2 or 3 — applied
        uniformly across all bins (never mixed per-bin).
    CL : np.ndarray
        Target (physical, non-Gaussian) angular power spectra, shape
        (Nbins, Nbins, pad_lmax + 1); off-diagonal entries `CL[i, j]`
        (i != j) are cross-power spectra between bins i and j.
    Nbins : int
        Number of tomographic bins (`dg_obs.shape[0]`).
    Nside : int
        HEALPix resolution parameter, from `dg_obs[0]`.
    pixel_size : float
        Pixel angular size in radians (`hp.nside2resol(Nside)`).
    map_shape : tuple of int
        Shape of `dg_obs`.
    N_bar : np.ndarray
        Average galaxy count per pixel, per bin — averaged across the
        full sky, not just the observed/masked region.
    Ng_obs : np.ndarray
        Observed galaxy counts per masked pixel, per bin —
        `(dg_obs[:, mask] + 1) * N_bar`, rounded to the nearest integer.
    lmax : int
        Maximum multipole of the output maps, by default `2 * Nside`.
    gen_lmax : int
        Maximum multipole used internally for Gaussian field generation,
        by default `3 * Nside - 1` (higher than `lmax`, to avoid
        aliasing from the nonlinear lognormal transform).
    pad_lmax : int
        Maximum multipole of the input `CL` and of the `CL -> CL_G`
        Legendre round-trip, `3.5 * Nside - 1` (higher than `gen_lmax`).
        Truncating that round-trip's quadrature at `gen_lmax` directly
        leaves ringing/mode-mixing artifacts (worse for `gn_order=3`)
        inside the multipole range that's actually kept; computing it
        out to `pad_lmax` and discarding the `(gen_lmax, pad_lmax]`
        buffer pushes those artifacts out of the retained range instead.
    ell, emm : np.ndarray
        Harmonic (l, m) index arrays at `lmax` resolution, from
        `hp.Alm.getlm(lmax)`.
    gen_ell, gen_emm : np.ndarray
        Harmonic (l, m) index arrays at `gen_lmax` resolution.
    pixwin : np.ndarray or None
        Pixel window function, indexed by multipole (length >= lmax + 1).
    CL_G : np.ndarray
        Gaussianized angular power spectra: the covariance of the
        underlying Gaussian field whose `gn_order`-transform reproduces
        `CL`. Set by `compute_CL_G`.
    L_G : np.ndarray
        Per-multipole Cholesky factor of `CL_G`, used by `apply_CL_G` to
        correlate independent per-bin Gaussian `xlm` draws into the
        physically-correlated Gaussian field. Set by `compute_CL_G`.
    _real_idx, _imag_idx : np.ndarray
        Indices selecting the free (non-redundant) real/imaginary
        harmonic modes of a real field, for packing/unpacking
        `XlmParams` (see `get_xlm`).
    n_modes : int
        Total number of free `xlm` parameters (`len(_real_idx) +
        len(_imag_idx)`).
    n_real, n_imag : int
        Per-bin count of free real/imaginary harmonic modes (`len(_real_idx)`,
        `len(_imag_idx)`).
    """

    def __init__(
        self,
        dg_obs: np.ndarray,
        mask: np.ndarray,
        CL: np.ndarray,
        lbda: np.ndarray,
        gn_order: int,
        N_bar: np.ndarray | None = None,
        lmax: int | None = None,
        gen_lmax: int | None = None,
        pixwin: np.ndarray | None = None,
    ) -> None:
        self.dg_obs = dg_obs
        self.mask = mask.astype(bool)

        if lbda.shape[0] != gn_order:
            raise ValueError(
                f"lbda has {lbda.shape[0]} rows, expected gn_order={gn_order}"
            )
        self.lbda = lbda
        self.gn_order = gn_order

        self.CL = CL

        self.Nbins = dg_obs.shape[0]
        if lbda.shape[1] != self.Nbins:
            raise ValueError(
                f"lbda has {lbda.shape[1]} columns, expected Nbins={self.Nbins}"
            )
        self.Nside = hp.get_nside(self.dg_obs[0])
        self.pixel_size = float(hp.nside2resol(self.Nside))
        self.map_shape = dg_obs.shape

        self.N_bar = np.asarray(N_bar)
        self.Ng_obs = np.round(
            (dg_obs[:, self.mask] + 1.0) * self.N_bar[:, None]
        ).astype(np.int32)

        self.lmax = 2 * self.Nside if lmax is None else lmax
        self.gen_lmax = 3 * self.Nside - 1 if gen_lmax is None else gen_lmax
        self.pad_lmax = int(3.5 * self.Nside) - 1

        if self.CL.shape[-1] != self.pad_lmax + 1:
            raise ValueError(
                f"CL has {self.CL.shape[-1]} multipoles, expected "
                f"pad_lmax + 1 = {self.pad_lmax + 1}"
            )

        self.ell, self.emm = hp.Alm.getlm(self.lmax)
        self.gen_ell, self.gen_emm = hp.Alm.getlm(self.gen_lmax)

        self.pixwin = pixwin

        self.compute_CL_G()

        # Precomputed so that get_xlm can be jit compiled — numpy, not jnp,
        # so these static index constants are never confused with JAX tracers.
        self._real_idx = np.where(self.gen_ell > 1)[0]
        self._imag_idx = np.where((self.gen_ell > 1) & (self.gen_emm > 0))[0]
        self.n_real = len(self._real_idx)
        self.n_imag = len(self._imag_idx)
        self.n_modes = self.n_real + self.n_imag

    @staticmethod
    def _xi_NG_to_xi_G_g3(
        params_i: np.ndarray,
        params_j: np.ndarray,
        xi_NG: np.ndarray,
        n_iter: int = 50,
        tol: float = 1e-13,
    ) -> np.ndarray:
        """Invert G3's closed-form `xi_NG(xi_G)` relation via vectorized Newton's method.

        `xi_NG(xi_G) = norm_i*norm_j*(exp(K*xi_G) + L*xi_G + c_i+c_j+c_i*c_j) - 1`,
        with `K = a_i*a_j`, `L = a_i*b_j + a_j*b_i + b_i*b_j`, `norm = 1/(1+c)`.

        Parameters
        ----------
        params_i, params_j : np.ndarray
            This bin pair's (a, b, c) transform parameters, shape (3,) each.
        xi_NG : np.ndarray
            Target non-Gaussian correlation values (one per multipole/quadrature node).
        n_iter : int, optional
            Maximum Newton iterations, by default 50.
        tol : float, optional
            Convergence tolerance on the max per-iteration update, by default 1e-13.

        Returns
        -------
        np.ndarray
            The Gaussianized correlation `xi_G`, same shape as `xi_NG`.

        Raises
        ------
        ValueError
            If `params_i`/`params_j` don't guarantee `xi_NG(xi_G)` is monotonic
            over `xi_G in (-1, 1)` (`K*exp(-K) + L > 0`) — otherwise the
            Newton solve could converge to the wrong root silently.
        """
        a_i, b_i, c_i = params_i
        a_j, b_j, c_j = params_j
        K = a_i * a_j
        L = a_i * b_j + a_j * b_i + b_i * b_j
        norm = (1.0 / (1.0 + c_i)) * (1.0 / (1.0 + c_j))
        const = c_i + c_j + c_i * c_j

        if not (K * np.exp(-K) + L > 0):
            raise ValueError(
                f"G3 params {params_i}, {params_j} do not guarantee a monotonic "
                "xi_NG -> xi_G relation over xi_G in (-1, 1)."
            )

        xi_G = np.zeros_like(xi_NG, dtype=float)
        for _ in range(n_iter):
            F = norm * (np.exp(K * xi_G) + L * xi_G + const) - 1.0
            dF = norm * (K * np.exp(K * xi_G) + L)
            xi_G_new = np.clip(xi_G - (F - xi_NG) / dF, -0.999999, 0.999999)
            if np.max(np.abs(xi_G_new - xi_G)) < tol:
                xi_G = xi_G_new
                break
            xi_G = xi_G_new
        return xi_G

    def _compute_CL_G_binpair(
        self,
        i: int,
        j: int,
        ell_array: np.ndarray,
        P_ell: np.ndarray,
        w: np.ndarray,
        newton_iter: int = 50,
        newton_tol: float = 1e-13,
    ) -> np.ndarray:
        """Gaussianize the (i, j) bin-pair power spectrum `CL[i, j]` into `CL_G[i, j]`.

        Parameters
        ----------
        i, j : int
            Tomographic bin indices.
        ell_array : np.ndarray
            Multipoles 0..pad_lmax.
        P_ell : np.ndarray
            Legendre polynomials `P_ell(mu)` at the quadrature nodes
            `mu`, shape (pad_lmax + 1, n_quad).
        w : np.ndarray
            Gauss-Legendre quadrature weights, shape (n_quad,).
        newton_iter, newton_tol : int, float, optional
            Forwarded to `_xi_NG_to_xi_G_g3` when `gn_order=3`; ignored
            when `gn_order=2`. See `compute_CL_G`.

        Returns
        -------
        np.ndarray
            Gaussianized power spectrum for bin pair (i, j), at the
            padded resolution, shape (pad_lmax + 1,). `compute_CL_G`
            truncates this to `gen_lmax + 1` before storing it, to drop
            the high-multipole buffer used to keep the retained range
            free of quadrature/mode-mixing artifacts.

        Notes
        -----
        Three steps, via Gauss-Legendre quadrature:

        1. Forward Legendre transform: `CL[i, j]` to `xi_NG`, its
           real-space angular correlation function, evaluated at the
           quadrature nodes.
        2. Gaussianize pointwise: for `gn_order=2`, G2's exact closed-form
           log relation; for `gn_order=3`, `_xi_NG_to_xi_G_g3` (closed-form
           relation + Newton solve — see that method's docstring for why
           not the Lambert-W route).
        3. Inverse Legendre transform: `xi_G` back to `CL_G[i, j]`.

        The monopole/dipole (l=0,1) are forced to ~0 (exactly 0
        off-diagonal, a tiny 1e-20 floor on-diagonal to avoid a
        zero-variance mode breaking the later per-multipole Cholesky
        decomposition in `compute_CL_G`) — a zero-mean fluctuation field
        has no physically meaningful monopole/dipole.
        """
        weighted_CL = (2 * ell_array + 1) * self.CL[i, j]
        xi_NG = weighted_CL @ P_ell / (4 * np.pi)

        params_i = self.lbda[:, i]
        params_j = self.lbda[:, j]
        if self.gn_order == 2:
            alpha_i, beta_i = params_i
            alpha_j, beta_j = params_j
            xi_G = np.log(1 + xi_NG / (beta_i * beta_j)) / (alpha_i * alpha_j)
        elif self.gn_order == 3:
            xi_G = self._xi_NG_to_xi_G_g3(
                params_i, params_j, xi_NG, n_iter=newton_iter, tol=newton_tol
            )
        else:
            raise ValueError(f"Unknown gn_order: {self.gn_order}")

        weighted_xi_G = w * xi_G
        CL_G_ij = 2 * np.pi * (P_ell @ weighted_xi_G)
        CL_G_ij[:2] = 1e-20 if i == j else 0.0
        return CL_G_ij

    def compute_CL_G(
        self, quad_order: int = 2, newton_iter: int = 50, newton_tol: float = 1e-13
    ) -> None:
        """Gaussianize `CL` into `CL_G`, and Cholesky-factorize it into `L_G`.

        Parameters
        ----------
        quad_order : int, optional
            Gauss-Legendre quadrature order multiplier — uses
            `quad_order * pad_lmax` quadrature points, by default 2.
        newton_iter : int, optional
            Maximum Newton iterations for the `gn_order=3` case, by
            default 50. Ignored when `gn_order=2`.
        newton_tol : float, optional
            Newton convergence tolerance for the `gn_order=3` case, by
            default 1e-13. Ignored when `gn_order=2`.

        Notes
        -----
        Calls `_compute_CL_G_binpair` once per bin pair (i, j) with
        i >= j, mirroring the result across the diagonal since `CL_G` is
        symmetric in the bin indices. The Legendre round-trip itself
        runs at `pad_lmax` resolution; each bin pair's result is then
        truncated to `gen_lmax + 1` before being stored, discarding the
        `(gen_lmax, pad_lmax]` buffer (see `pad_lmax`'s docstring). Sets
        `self.CL_G` and `self.L_G` (`CL_G`'s per-multipole Cholesky
        factor, used by `apply_CL_G`), both at `gen_lmax` resolution.
        """
        mu, w = roots_legendre(quad_order * self.pad_lmax)
        ell_array = np.arange(self.pad_lmax + 1)
        P_ell = legendre_p_all(self.pad_lmax, mu).squeeze()
        self.CL_G = np.zeros((self.Nbins, self.Nbins, self.gen_lmax + 1))
        for i in range(self.Nbins):
            for j in range(i + 1):
                self.CL_G[i, j, :] = self._compute_CL_G_binpair(
                    i,
                    j,
                    ell_array,
                    P_ell,
                    w,
                    newton_iter=newton_iter,
                    newton_tol=newton_tol,
                )[: self.gen_lmax + 1]
                if i != j:
                    self.CL_G[j, i] = self.CL_G[i, j]
        CL_T = np.moveaxis(self.CL_G, 2, 0)
        L_T = np.linalg.cholesky(CL_T)
        self.L_G = np.moveaxis(L_T, 0, 2)

    def get_xlm(self, xlm: XlmParams) -> jnp.ndarray:
        """Unpack `xlm`'s free real/imaginary parameters into the full complex harmonic array.

        Parameters
        ----------
        xlm : XlmParams
            Free (non-redundant) real/imaginary harmonic coefficients, at
            the indices given by `_real_idx`/`_imag_idx`.

        Returns
        -------
        jnp.ndarray
            Full complex harmonic-coefficient array, shape
            (Nbins, len(gen_ell)). Modes not in `_real_idx`/`_imag_idx`
            are left at zero: the monopole/dipole (l<=1, excluded per
            `_compute_CL_G_binpair`), and the imaginary part of m=0
            modes (forced to zero by the reality condition of a
            real-valued map).
        """
        _real = jnp.zeros((self.Nbins, len(self.gen_ell)), dtype=jnp.float64)
        _imag = jnp.zeros_like(_real)
        _real = _real.at[:, self._real_idx].set(xlm.real)
        _imag = _imag.at[:, self._imag_idx].set(xlm.imag)
        return _real + 1j * _imag

    def apply_CL_G(self, xlm_array: jnp.ndarray) -> jnp.ndarray:
        """Correlate independent per-bin harmonic coefficients into the physical Gaussian field.

        Parameters
        ----------
        xlm_array : jnp.ndarray
            Full complex harmonic-coefficient array (independent per
            bin), as returned by `get_xlm`, shape (Nbins, len(gen_ell)).

        Returns
        -------
        jnp.ndarray
            Correlated complex harmonic coefficients of the underlying
            Gaussian field, same shape as `xlm_array`.

        Notes
        -----
        Applies `L_G` (the per-multipole Cholesky factor of `CL_G`, set
        by `compute_CL_G`) as a per-multipole linear mixing across bins
        (the same matrix for every m at a given ell), turning
        independent unit-variance `xlm_array` draws into correlated
        `ylm` with the target cross-bin covariance `CL_G`. The
        `1/sqrt(2)` factor (undone for m=0 via the `gen_emm == 0`
        branches) splits the variance evenly between the real and
        imaginary parts for m>0 modes; at m=0 the coefficient is purely
        real (no imaginary part, per the reality condition), so it alone
        must carry the full variance instead.
        """
        L_expanded = self.L_G[:, :, self.gen_ell]
        ylm_real = jnp.einsum("ijm,jm->im", L_expanded, xlm_array.real) / jnp.sqrt(2)
        ylm_imag = jnp.einsum("ijm,jm->im", L_expanded, xlm_array.imag) / jnp.sqrt(2)
        ylm_real = jnp.where(self.gen_emm == 0, ylm_real * jnp.sqrt(2), ylm_real)
        ylm_imag = jnp.where(self.gen_emm == 0, 0.0, ylm_imag)
        return ylm_real + 1j * ylm_imag

    @staticmethod
    def gn(x: jnp.ndarray, N: int, lbda: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the G_N point-transformation for all tomographic bins at once.

        Parameters
        ----------
        x : jnp.ndarray
            Standard-normal latent Gaussian field values, shape (Nbins, ...)
            (e.g. one HEALPix map per bin).
        N : int
            Transformation order. Currently ``2``
            (shifted lognormal) or ``3`` are supported.
        lbda : jnp.ndarray
            Transformation parameters, shape ``(N, Nbins)``: rows ``(alpha,
            beta)`` for ``N=2``, or ``(a, b, c)`` for ``N=3``, one column per
            bin.

        Returns
        -------
        jnp.ndarray
            Transformed field, same shape as `x`.

        Raises
        ------
        ValueError
            If `N` is not `2` or `3`.
        """
        if N == 2:
            alpha, beta = lbda[..., jnp.newaxis]
            return beta * jnp.exp(alpha * x - 0.5 * alpha**2) - beta

        elif N == 3:
            a, b, c = lbda[..., jnp.newaxis]
            arg = jnp.exp(a * x - 0.5 * a**2) + b * x + c
            norm = 1.0 / (1.0 + c)
            return norm * arg - 1.0

        else:
            raise ValueError(f"Unknown model type: {N}")

    def x2deff(self, xlm: XlmParams, theta: ThetaParams) -> jnp.ndarray:
        """Forward-model `xlm` into the effective density field used for the likelihood.

        Follows `x2dm`'s pipeline up through the harmonic-space density
        contrast `dm_lm`, then additionally applies a theta-dependent
        smoothing filter before transforming back to a map.

        Parameters
        ----------
        xlm : XlmParams
            Free harmonic coefficients (see `get_xlm`).
        theta : ThetaParams
            Bias/nuisance parameters; only `c` (smoothing amplitude) and
            `log_R` (smoothing scale) are used here — the rest are used
            later, in `dm_to_binom_params`.

        Returns
        -------
        jnp.ndarray
            Effective density contrast map, shape (Nbins, npix).

        Notes
        -----
        After `dm_lm` (see `x2dm`), applies
        `filt = (1 + c * b_ell) * pixwin`, where `b_ell` is a Gaussian
        smoothing kernel of scale `R = exp(log_R) * pixel_size`, then
        transforms back to a map.
        """
        xlm_full = self.get_xlm(xlm)
        ylm = self.apply_CL_G(xlm_full)

        ys = alm2map(ylm, self.Nside, self.gen_lmax)
        dm = self.gn(ys, self.gn_order, self.lbda)
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

    def x2dm(self, xlm: XlmParams) -> jnp.ndarray:
        """Forward-model `xlm` into the raw density contrast map.

        Parameters
        ----------
        xlm : XlmParams
            Free harmonic coefficients (see `get_xlm`).

        Returns
        -------
        jnp.ndarray
            Raw density contrast map, shape (Nbins, npix).

        Notes
        -----
        Pipeline: `xlm` -> full harmonic array (`get_xlm`) -> correlated
        Gaussian field harmonics (`apply_CL_G`) -> Gaussian field map
        `ys` (`alm2map`) -> the configured `G_N` point transform (`gn`,
        `self.gn_order`/`self.lbda`) to the density contrast `dm` -> back
        to harmonic space (`map2alm`), pixel-window-filtered if set ->
        back to a map.
        """
        xlm_full = self.get_xlm(xlm)
        ylm = self.apply_CL_G(xlm_full)

        ys = alm2map(ylm, self.Nside, self.gen_lmax)
        dm = self.gn(ys, self.gn_order, self.lbda)
        dm_lm = map2alm(dm, self.lmax)
        if self.pixwin is not None:
            dm_lm = dm_lm * self.pixwin[self.ell]
        return alm2map(dm_lm, self.Nside, self.lmax)

    def dm_to_binom_params(
        self, deff: jnp.ndarray, theta: ThetaParams, *, mask_output: bool = False
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Convert the effective density field into binomial (n, p) count parameters.

        Parameters
        ----------
        deff : jnp.ndarray
            Effective density contrast over the full sky, shape
            (Nbins, npix), as returned by `x2deff` — full-sky for either
            `mask_output` setting.
        theta : ThetaParams
            Bias/nuisance parameters; uses `A_t`, `log_T` (detection
            threshold/sharpness) and `mu0`, `a` (variance-depletion
            offset/slope).
        mask_output : bool, optional
            Whether to return only the `mask` columns (True) or the full
            sky (False), by default False. `log_prob`'s binomial
            likelihood consumes the observed footprint alone, so
            evaluating the full sky there spends ~8x the elementwise work
            and ~8x the reverse-mode residuals on results it discards.

        Returns
        -------
        n : jnp.ndarray
            Binomial number-of-trials parameter, per bin per pixel; shape
            (Nbins, npix), or (Nbins, mask.sum()) if `mask_output`.
        p : jnp.ndarray
            Binomial success-probability parameter, same shape as `n`.

        Notes
        -----
        `A = log1p(deff)` is a log-density variable, and
        `sig = sigmoid((A - A_t) / T)` is a detection/completeness
        function of it: `sig -> 1` well above the threshold `A_t`,
        `sig -> 0` well below it, with `T = exp(log_T)` controlling the
        transition sharpness. `b` renormalizes the calibrated mean count
        (`mean_Ng = b * (1 + deff) * sig * N_bar`) so its sky average
        over the observed mask matches `N_bar`, given the current
        `deff` realization. That average runs over the observed mask
        under either `mask_output` setting, so `b` comes out the same
        scalar per bin either way; given `b`, `n` and `p` are elementwise
        in `deff`, making the `mask_output=True` result exactly the
        `mask` columns of the `mask_output=False` one.

        A second mean count, `mean_Ng_prime`, is then computed at a
        density perturbed by `mu = mu0 + a * deff`. The difference
        `mean_Ng - mean_Ng_prime` becomes the binomial `p`, and
        `n = mean_Ng / p` is backed out so `n * p` still equals the
        calibrated mean exactly. This decouples the binomial's mean
        from its variance: a direct fit to `mean_Ng` alone would force
        a fixed mean-variance relationship, but real galaxy counts need
        that variance independently tunable (sub- or super-Poissonian)
        — `mu0`/`a` dial the effective variance via how far
        `mean_Ng_prime` sits from `mean_Ng`, while `n` keeps the mean
        itself pinned to the calibrated target.

        TODO: this is the heart of the forward model and deserves a
        more thorough writeup later.
        """
        A_t = theta.A_t[:, np.newaxis]
        T = jnp.exp(theta.log_T)[:, np.newaxis]
        mu0 = theta.mu0[:, np.newaxis]
        a = theta.a[:, np.newaxis]
        N_bar = self.N_bar[:, np.newaxis]

        deff = deff[:, self.mask] if mask_output else deff

        A = jnp.log1p(deff)
        sig = jax.nn.sigmoid((A - A_t) / T)

        if mask_output:
            b = 1.0 / jnp.mean((1 + deff) * sig, axis=1)
        else:
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

    def log_prob(self, params: KarmmaPosition) -> jnp.ndarray:
        """Compute the (unnormalized) posterior log-density at `params`.

        Parameters
        ----------
        params : KarmmaPosition
            Position to evaluate; both `params.xlm` and `params.theta`
            must be set.

        Returns
        -------
        jnp.ndarray
            Log-density, summing the binomial likelihood, the `xlm`
            prior, and `theta`'s prior and change-of-variables Jacobian.

        Notes
        -----
        `xlm` has an i.i.d. `N(0, 1)` prior on both its real and
        imaginary free parameters.

        The binomial term is evaluated on the observed footprint only
        (`dm_to_binom_params(..., mask_output=True)`), matching
        `Ng_obs`'s masked layout.

        Two additional terms cover `theta`:

        - `log_jacobian_theta`: `log_T`/`log_R` are sampled in
          log-space, but `T`/`R` themselves have flat priors, so
          sampling in log-space adds a `+log_T`/`+log_R` Jacobian term
          each (`d(T)/d(log T) = T`, and likewise for `R`).
        - `log_prior_theta`: an `InvGamma(alpha, beta)` prior on `R^2`
          (not `R` itself), with fixed hyperparameters
          `_INVGAMMA_ALPHA_R`/`_INVGAMMA_BETA_R` (not yet exposed via
          config — see the module-level TODOs). Combines the
          `R -> R^2` Jacobian (another `+log_R`, dropping the constant
          `log(2)` term) with the InvGamma log-density in terms of
          `log_R`.
        """
        theta = params.theta

        deff = self.x2deff(params.xlm, theta)

        n, p = self.dm_to_binom_params(deff, theta, mask_output=True)

        log_lik = jnp.sum(jst.binom.logpmf(self.Ng_obs, n, p))

        log_prior_real = jnp.sum(jst.norm.logpdf(params.xlm.real, loc=0.0, scale=1.0))
        log_prior_imag = jnp.sum(jst.norm.logpdf(params.xlm.imag, loc=0.0, scale=1.0))

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

    def make_random_xlm(self, key: jax.Array) -> XlmParams:
        """Draw a random `xlm` from its prior (i.i.d. standard normal per component).

        Parameters
        ----------
        key : jax.Array
            JAX PRNG key.

        Returns
        -------
        XlmParams
            Random draw, matching `log_prob`'s `N(0, 1)` prior on `xlm`.
        """
        rk, ik = jax.random.split(key)
        return XlmParams(
            real=jax.random.normal(
                rk, shape=(self.Nbins, len(self._real_idx)), dtype=jnp.float64
            ),
            imag=jax.random.normal(
                ik, shape=(self.Nbins, len(self._imag_idx)), dtype=jnp.float64
            ),
        )

    @staticmethod
    @functools.partial(jax.jit, static_argnums=(3,))
    def _inverse_cdf_scan(
        n: jnp.ndarray, p: jnp.ndarray, u: jnp.ndarray, kmax: int
    ) -> jnp.ndarray:
        """Inverse-CDF sample the continuous-`n` binomial pmf, streamed via `fori_loop`.

        Parameters
        ----------
        n, p : jnp.ndarray
            Continuous binomial parameters, same shape (e.g. (Nbins, npix)).
        u : jnp.ndarray
            `Uniform(0, 1)` draws, same shape as `n`.
        kmax : int
            Largest count evaluated; static, since it sets the scan length.

        Returns
        -------
        jnp.ndarray
            Sampled counts, same shape as `n`.

        Notes
        -----
        Builds the pmf via its exact ratio recursion rather than
        evaluating `jax.scipy.stats.binom.logpmf` at every `k`, so a full
        `(Nbins, npix, kmax)` table is never materialized.

        For non-integer `n`, `k = ceil(n)` is provably the last point with
        nonzero probability (the generalized binomial coefficient goes
        negative just past it), independent of `p` — so `kmax` only needs
        a margin for floating-point safety, not real tail allowance.

        Known limitation: `(1-p)**n` can underflow to exactly `0.0` for
        `p` near 1 and `n` upward of ~50, after which the recursion can't
        recover. Not an issue in this model's actual regime (checked:
        `n * -log1p(-p)` stays far below float64's ~745 underflow limit).
        """
        odds = p / (1.0 - p)
        w = jnp.power(1.0 - p, n)  # pmf(0)
        S = w
        counts = (u > S).astype(jnp.int32)

        def body(
            k: jnp.ndarray, carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            """One recursion step: advance the pmf weight/running sum, and the count comparison."""
            w, S, counts = carry
            num = n - k + 1.0
            ratio = jnp.where(num >= 0.0, num / k * odds, 0.0)
            w = w * ratio
            S = S + w
            return (w, S, counts + (u > S).astype(jnp.int32))

        _, _, counts = jax.lax.fori_loop(1, kmax + 1, body, (w, S, counts))
        return counts.astype(jnp.float64)

    def generate_mock_dg_obs(
        self, xlm: XlmParams, theta: ThetaParams, key: jax.Array
    ) -> jnp.ndarray:
        """Generate a synthetic `dg_obs` realization from a given `xlm`/`theta`.

        Parameters
        ----------
        xlm : XlmParams
            Free harmonic coefficients (see `get_xlm`) — the latent truth
            to generate a mock observation from.
        theta : ThetaParams
            Bias/nuisance parameters.
        key : jax.Array
            JAX PRNG key for the count draw.

        Returns
        -------
        jnp.ndarray
            Synthetic galaxy overdensity map, shape (Nbins, npix), in the
            same `dg_obs = counts/N_bar - 1` convention `log_prob` expects.

        Notes
        -----
        Counts are drawn by exact inverse-CDF sampling from the same
        continuous-`n` pmf `log_prob` uses (`_inverse_cdf_scan`), not by
        rounding `n` first — rounding would sample from a distribution
        other than the one the likelihood assumes, biasing recovery.

        `(n, p)` are full-sky (`mask_output=False`): the mock is a
        complete HEALPix map, written to disk and read back for `Nside`.

        Raises
        ------
        _KmaxExceeded
            If `kmax` comes out above 5000 — an extreme `deff` outlier
            can send `n` (and so `kmax`) into the millions, making
            `_inverse_cdf_scan` hang. Caught by `make_random_mock`.
        """
        deff = self.x2deff(xlm, theta)
        n, p = self.dm_to_binom_params(deff, theta, mask_output=False)
        kmax = int(np.ceil(float(jnp.max(n)))) + 1
        if kmax > 5000:
            raise _KmaxExceeded(kmax)
        u = jax.random.uniform(key, shape=n.shape, dtype=jnp.float64)
        counts = self._inverse_cdf_scan(n, p, u, kmax)
        return counts / self.N_bar[:, None] - 1.0

    def make_random_mock(
        self, key: jax.Array, theta: ThetaParams
    ) -> tuple[XlmParams, jnp.ndarray]:
        """Draw a random `xlm` and generate its corresponding synthetic `dg_obs`.

        Parameters
        ----------
        key : jax.Array
            JAX PRNG key; re-split on each attempt into a fresh
            `xlm`-draw key (see `make_random_xlm`) and binomial-draw key
            (see `generate_mock_dg_obs`).
        theta : ThetaParams
            Bias/nuisance parameters.

        Returns
        -------
        xlm : XlmParams
            The randomly-drawn latent truth.
        dg_obs : jnp.ndarray
            Its corresponding synthetic galaxy overdensity map, shape
            (Nbins, npix).

        Raises
        ------
        RuntimeError
            If 20 consecutive draws all raise `_KmaxExceeded` (see
            `generate_mock_dg_obs`) — rejects an `xlm` draw and retries
            with a fresh split of `key` on each occurrence, so exhausting
            the budget means something systematic rather than one rare
            outlier.
        """
        for _ in range(20):
            key, xlm_key, obs_key = jax.random.split(key, 3)
            xlm = self.make_random_xlm(xlm_key)
            try:
                dg_obs = self.generate_mock_dg_obs(xlm, theta, obs_key)
            except _KmaxExceeded:
                continue
            return xlm, dg_obs
        raise RuntimeError("make_random_mock: 20 consecutive draws all exceeded kmax=5000")
