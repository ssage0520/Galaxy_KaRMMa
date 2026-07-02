import healpy as hp
import numpy as np
import pymaster as nmt
import treecorr


def get_corrfunc(field_maps, mask, min_sep=None, max_sep=300.0, nbins=15, npatch=50):
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


def get_field_bins(field, mask, n_bins=46, n_sigma_linear=4):
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


def get_1ptfunc(field_maps, mask, linear_bins, log_bins):
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


def setup_pseudo_cls(mask, n_ell_bins=17):
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


def get_pseudo_cls(field_maps, mask, nmt_ell_bins, workspace):
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
