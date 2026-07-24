"""Summary statistics (correlation function, pseudo-Cl, 1-point PDF) for KaRMMa fields."""

import healpy as hp
import numpy as np
import pymaster as nmt
import treecorr


def get_corrfunc(
    field_maps: np.ndarray,
    mask: np.ndarray,
    min_sep: float | None = None,
    max_sep: float = 300.0,
    nbins: int = 15,
    npatch: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the real-space 2-point correlation function, per tomographic-bin pair.

    Uses treecorr's scalar-scalar (KK) correlation with jackknife
    covariance estimation.

    Parameters
    ----------
    field_maps : np.ndarray
        Field values per tomographic bin, shape (n_zbins, npix)
        (full-sky HEALPix maps).
    mask : np.ndarray
        Boolean survey mask, shape (npix,).
    min_sep : float or None, optional
        Minimum angular separation in arcmin, by default the map's pixel
        resolution.
    max_sep : float, optional
        Maximum angular separation in arcmin, by default 300.0.
    nbins : int, optional
        Number of separation bins, by default 15.
    npatch : int, optional
        Number of jackknife patches for covariance estimation, by
        default 50.

    Returns
    -------
    corr : np.ndarray
        Correlation function, shape (n_zbins, n_zbins, nbins).
    errors : np.ndarray
        Jackknife standard errors, same shape as `corr`.
    bin_centres : np.ndarray
        Geometric-mean separation of each bin, shape (nbins,).
    bin_edges : np.ndarray
        Separation bin edges, shape (nbins + 1,).
    """
    nside = hp.npix2nside(field_maps.shape[1])

    if min_sep is None:
        min_sep = hp.pixelfunc.nside2resol(nside, arcmin=True)

    ipix = np.where(mask)[0]
    ra, dec = hp.pix2ang(nside, ipix, lonlat=True)
    n_zbins = field_maps.shape[0]

    cats = [
        treecorr.Catalog(
            ra=ra,
            dec=dec,
            k=field_maps[i][ipix],
            ra_units="deg",
            dec_units="deg",
            npatch=npatch,
        )
        for i in range(n_zbins)
    ]

    kk = treecorr.KKCorrelation(
        min_sep=min_sep,
        max_sep=max_sep,
        nbins=nbins,
        sep_units="arcmin",
        bin_slop=0.1,
        cross_patch_weight="match",
    )

    corr = np.zeros((n_zbins, n_zbins, nbins))
    errors = np.zeros((n_zbins, n_zbins, nbins))

    for i in range(n_zbins):
        for j in range(i + 1):
            kk.process(cats[i], cats[j])
            cov = kk.estimate_cov("jackknife")
            corr[i, j] = kk.xi
            corr[j, i] = kk.xi
            errors[i, j] = np.sqrt(np.diag(cov))
            errors[j, i] = np.sqrt(np.diag(cov))

    bin_centres = np.exp(kk.meanlogr)
    bin_edges = np.append(np.exp(kk.left_edges), np.exp(kk.right_edges[-1]))

    return corr, errors, bin_centres, bin_edges


def get_field_bins(
    field: np.ndarray, mask: np.ndarray, n_bins: int = 46, n_sigma_linear: float = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Compute two sets of histogram bin edges per tomographic bin, for the 1-point PDF.

    Despite the name, both sets of edges are linearly spaced — `linear_bins`
    and `log_bins` differ in range and intended plot y-axis scale, not bin
    spacing: `linear_bins` is truncated to `n_sigma_linear` standard
    deviations (a linear-y-axis view of the bulk of the distribution, see
    `plot_1pt_linear`), while `log_bins` spans the field's full min-max
    range (a log-y-axis view of the tails, see `plot_1pt_log`).

    Parameters
    ----------
    field : np.ndarray
        Field values per tomographic bin, shape (n_zbins, npix).
    mask : np.ndarray
        Boolean survey mask, shape (npix,).
    n_bins : int, optional
        Number of bin edges (n_bins - 1 histogram bins), by default 46.
    n_sigma_linear : float, optional
        Number of standard deviations `linear_bins` extends to, by
        default 4.

    Returns
    -------
    linear_bins : np.ndarray
        Bin edges truncated to `n_sigma_linear` std, shape (n_zbins, n_bins).
    log_bins : np.ndarray
        Bin edges spanning the field's full range, shape (n_zbins, n_bins).
    """
    n_zbins = field.shape[0]
    linear_bins = []
    log_bins = []

    for i in range(n_zbins):
        field_masked = field[i][mask]
        std_i = field_masked.std()
        field_min, field_max = field_masked.min(), field_masked.max()

        linear_range = n_sigma_linear * std_i - field_min
        linear_width = linear_range / (n_bins - 2)
        linear_bins_i = np.linspace(
            field_min - linear_width, n_sigma_linear * std_i, n_bins
        )

        log_range = field_max - field_min
        log_width = log_range / (n_bins - 3)
        log_bins_i = np.linspace(field_min - log_width, field_max + log_width, n_bins)

        linear_bins.append(linear_bins_i)
        log_bins.append(log_bins_i)

    return np.array(linear_bins), np.array(log_bins)


def get_1ptfunc(
    field_maps: np.ndarray,
    mask: np.ndarray,
    linear_bins: np.ndarray,
    log_bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 1-point histograms of a field, per tomographic bin, in both binning schemes.

    Parameters
    ----------
    field_maps : np.ndarray
        Field values per tomographic bin, shape (n_zbins, npix).
    mask : np.ndarray
        Boolean survey mask, shape (npix,).
    linear_bins : np.ndarray
        Per-bin histogram edges, as returned by `get_field_bins`, shape
        (n_zbins, n_bins).
    log_bins : np.ndarray
        Per-bin histogram edges, as returned by `get_field_bins`, shape
        (n_zbins, n_bins).

    Returns
    -------
    pdf_linear : np.ndarray
        Histogram counts using `linear_bins`, shape (n_zbins, n_bins - 1).
    pdf_log : np.ndarray
        Histogram counts using `log_bins`, shape (n_zbins, n_bins - 1).
    """
    n_zbins = field_maps.shape[0]
    n_linear = linear_bins.shape[1] - 1
    n_log = log_bins.shape[1] - 1

    pdf_linear = np.zeros((n_zbins, n_linear))
    pdf_log = np.zeros((n_zbins, n_log))

    for i in range(n_zbins):
        field_masked = field_maps[i][mask]
        pdf_linear[i], _ = np.histogram(field_masked, linear_bins[i])
        pdf_log[i], _ = np.histogram(field_masked, log_bins[i])

    return pdf_linear, pdf_log


def setup_pseudo_cls(
    mask: np.ndarray, n_ell_bins: int = 17
) -> tuple[nmt.NmtWorkspace, nmt.NmtBin, np.ndarray, np.ndarray]:
    """Set up NaMaster bandpowers and the mode-coupling matrix for pseudo-Cl estimation.

    Parameters
    ----------
    mask : np.ndarray
        Survey mask, shape (npix,).
    n_ell_bins : int, optional
        Number of log-spaced multipole bandpower edges, by default 17.

    Returns
    -------
    workspace : nmt.NmtWorkspace
        Precomputed mode-coupling matrix for this mask, reusable across
        `get_pseudo_cls` calls.
    nmt_ell_bins : nmt.NmtBin
        Bandpower binning scheme.
    eff_ell : np.ndarray
        Effective multipole of each bandpower.
    ell_edges : np.ndarray
        Multipole bandpower edges (genuinely log-spaced, via `np.logspace`).
    """
    nside = hp.npix2nside(mask.shape[0])
    lmax = 2 * nside

    ell_edges = np.ceil(np.logspace(np.log10(3), np.log10(lmax), n_ell_bins)).astype(
        int
    )
    ells = np.arange(lmax + 1)
    bpws = np.searchsorted(ell_edges[1:], ells, side="left")
    bpws[ells < ell_edges[0]] = -1
    bpws[ells >= ell_edges[-1]] = -1
    nmt_ell_bins = nmt.NmtBin(bpws=bpws, ells=ells, lmax=lmax)
    eff_ell = nmt_ell_bins.get_effective_ells()

    mask_field = nmt.NmtField(mask.astype(float), None, spin=0, lmax=lmax)
    workspace = nmt.NmtWorkspace()
    workspace.compute_coupling_matrix(mask_field, mask_field, nmt_ell_bins)

    return workspace, nmt_ell_bins, eff_ell, ell_edges


def get_pseudo_cls(
    field_maps: np.ndarray,
    mask: np.ndarray,
    nmt_ell_bins: nmt.NmtBin,
    workspace: nmt.NmtWorkspace,
) -> np.ndarray:
    """Compute mode-decoupled pseudo-Cl angular power spectra, per tomographic-bin pair.

    Parameters
    ----------
    field_maps : np.ndarray
        Field values per tomographic bin, shape (n_zbins, npix).
    mask : np.ndarray
        Survey mask, shape (npix,).
    nmt_ell_bins : nmt.NmtBin
        Bandpower binning scheme, as returned by `setup_pseudo_cls`.
    workspace : nmt.NmtWorkspace
        Precomputed mode-coupling matrix, as returned by `setup_pseudo_cls`.

    Returns
    -------
    np.ndarray
        Pseudo-Cl power spectra, shape (n_zbins, n_zbins, n_ell).
    """
    nside = hp.npix2nside(mask.shape[0])
    lmax = 2 * nside
    n_zbins = field_maps.shape[0]
    n_ell = nmt_ell_bins.get_n_bands()

    fields = [
        nmt.NmtField(mask.astype(float), [field_maps[i]], spin=0, lmax=lmax)
        for i in range(n_zbins)
    ]

    cls = np.zeros((n_zbins, n_zbins, n_ell))
    for i in range(n_zbins):
        for j in range(i + 1):
            cl_ij = nmt.compute_full_master(
                fields[i], fields[j], nmt_ell_bins, workspace=workspace
            )
            cls[i, j] = cl_ij[0]
            cls[j, i] = cl_ij[0]

    return cls
