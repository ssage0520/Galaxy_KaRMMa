"""Plotting utilities for KaRMMa maps, correlation functions, pseudo-Cl, and 1-point PDFs."""

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import skyproj
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def plot_map(
    dm_map: np.ndarray,
    mask: np.ndarray,
    minmax: tuple[float, float],
    cmap: str = "viridis",
    cb_label: str = r"$\delta_m$",
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> skyproj.DESSkyproj:
    r"""Plot a HEALPix map on a DES sky projection, masking unseen pixels.

    Parameters
    ----------
    dm_map : np.ndarray
        HEALPix map values.
    mask : np.ndarray
        Survey mask; pixels outside it are drawn as unseen.
    minmax : tuple of float
        (vmin, vmax) colorbar range.
    cmap : str, optional
        Colormap name, by default "viridis".
    cb_label : str, optional
        Colorbar label, by default r"$\delta_m$".
    title : str or None, optional
        Plot title, by default None.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw into, by default None (creates a new one).

    Returns
    -------
    skyproj.DESSkyproj
        The sky projection object drawn into.
    """
    masked_map = dm_map.copy()
    masked_map[~mask.astype(bool)] = hp.UNSEEN
    vmin, vmax = minmax
    sp = skyproj.DESSkyproj(ax=ax)
    sp.draw_hpxmap(masked_map, vmin=vmin, vmax=vmax, cmap=cmap)
    sp.draw_inset_colorbar(label=cb_label)
    if title is not None:
        sp.ax.set_title(title, pad=25)
    return sp


def plot_dm_comparison(
    dm_true: np.ndarray,
    dm_mean: np.ndarray,
    mask: np.ndarray,
    n_samples: int | None = None,
    cmap: str = "viridis",
) -> None:
    """Plot the true dm map beside the sample-mean dm map, per tomographic bin.

    Parameters
    ----------
    dm_true : np.ndarray
        True dm map, shape (nbins, npix).
    dm_mean : np.ndarray
        Sample-mean dm map (pre-averaged over samples), shape (nbins, npix).
    mask : np.ndarray
        Survey mask.
    n_samples : int or None, optional
        Number of samples averaged into `dm_mean`, shown in the title,
        by default None.
    cmap : str, optional
        Colormap name, by default "viridis".
    """
    nbins = dm_true.shape[0]
    count_str = f"{n_samples} samples" if n_samples is not None else "samples"

    fig, axes = plt.subplots(nbins, 2, figsize=(14, 5 * nbins))
    if nbins == 1:
        axes = axes[np.newaxis, :]

    for i in range(nbins):
        minmax = np.percentile(dm_true[i][mask.astype(bool)], [1, 99])
        plot_map(
            dm_true[i],
            mask,
            minmax=minmax,
            cmap=cmap,
            cb_label=r"$\delta_m$",
            title=f"Bin {i + 1} — True $\\delta_m$",
            ax=axes[i, 0],
        )
        plot_map(
            dm_mean[i],
            mask,
            minmax=minmax,
            cmap=cmap,
            cb_label=r"$\delta_m$",
            title=f"Bin {i + 1} — $\\langle\\delta_m\\rangle$ ({count_str})",
            ax=axes[i, 1],
        )

    fig.subplots_adjust(hspace=0.15, top=0.95)
    plt.show()


def plot_corr(
    corr_samples: np.ndarray,
    corr_true: np.ndarray,
    bin_centres: np.ndarray,
    ylim: tuple[float, float] | None = None,
    interval: float = 68,
) -> None:
    r"""Plot sample/true correlation-function ratios, per tomographic-bin pair.

    Parameters
    ----------
    corr_samples : np.ndarray
        Per-sample correlation function, shape (n_samples, nbins, nbins, nsep).
    corr_true : np.ndarray
        True correlation function, shape (nbins, nbins, nsep).
    bin_centres : np.ndarray
        Separation bin centres, shape (nsep,).
    ylim : tuple of float or None, optional
        Y-axis limits, by default (0.95, 1.05).
    interval : float, optional
        Percentile interval to shade around the sample mean, by default 68.
    """
    if ylim is None:
        ylim = (0.95, 1.05)

    lo_p = (100 - interval) / 2
    hi_p = (100 + interval) / 2

    nbins = corr_true.shape[0]
    fig, axes = plt.subplots(nbins, nbins, figsize=(3 * nbins, 3 * nbins))

    for i in range(nbins):
        for j in range(nbins):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue

            ratio = corr_samples[:, i, j, :] / corr_true[i, j, :]
            ratio_mean = ratio.mean(0)
            ratio_lo = np.percentile(ratio, lo_p, axis=0)
            ratio_hi = np.percentile(ratio, hi_p, axis=0)

            ax.axhline(1.0, color="k", linestyle="--", linewidth=1.0)
            (l1,) = ax.semilogx(bin_centres, ratio_mean, "b-", linewidth=1.5)
            l2 = ax.fill_between(bin_centres, ratio_lo, ratio_hi, color="b", alpha=0.3)
            (l3,) = ax.semilogx(
                bin_centres, np.ones_like(bin_centres), "k--", linewidth=1.0
            )

            ax.set_ylim(ylim)
            ax.set_xlim(bin_centres[0], bin_centres[-1])
            ax.text(
                0.05,
                0.85,
                rf"$\xi_{{{i + 1}{j + 1}}}$",
                transform=ax.transAxes,
                fontsize=11,
            )

            if j != 0:
                ax.set_yticklabels([])

            if i == nbins - 1:
                ax.set_xlabel(r"$\theta$ (arcmin)")
            else:
                ax.set_xticklabels([])

    fig.supylabel(r"$\xi / \xi^\mathrm{true}$")
    axes[0, 1].legend(
        handles=[l1, l2, l3],
        labels=["Sample mean", f"Samples ({interval}th percentile)", "Truth"],
        loc="center",
        fontsize=11,
        framealpha=0.9,
    )
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    plt.show()


def plot_pseudo_cl(
    cl_samples: np.ndarray,
    cl_true: np.ndarray,
    eff_ell: np.ndarray,
    nside: int,
    ylim: tuple[float, float] | None = None,
    interval: float = 68,
) -> None:
    """Plot sample/true pseudo-Cl ratios, per tomographic-bin pair.

    Parameters
    ----------
    cl_samples : np.ndarray
        Per-sample pseudo-Cl, shape (n_samples, nbins, nbins, n_ell).
    cl_true : np.ndarray
        True pseudo-Cl, shape (nbins, nbins, n_ell).
    eff_ell : np.ndarray
        Effective multipole of each bandpower, shape (n_ell,).
    nside : int
        HEALPix resolution parameter, used to set the x-axis range.
    ylim : tuple of float or None, optional
        Y-axis limits, by default (0.95, 1.05).
    interval : float, optional
        Percentile interval to shade around the sample mean, by default 68.
    """
    if ylim is None:
        ylim = (0.95, 1.05)

    lo_p = (100 - interval) / 2
    hi_p = (100 + interval) / 2

    nbins = cl_true.shape[0]
    fig, axes = plt.subplots(nbins, nbins, figsize=(3 * nbins, 3 * nbins))

    for i in range(nbins):
        for j in range(nbins):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue

            ratio = cl_samples[:, i, j, :] / cl_true[i, j, :]
            ratio_mean = ratio.mean(0)
            ratio_lo = np.percentile(ratio, lo_p, axis=0)
            ratio_hi = np.percentile(ratio, hi_p, axis=0)

            ax.axhline(1.0, color="k", linestyle="--", linewidth=1.0)
            (l1,) = ax.semilogx(eff_ell, ratio_mean, "b-", linewidth=1.5)
            l2 = ax.fill_between(eff_ell, ratio_lo, ratio_hi, color="b", alpha=0.3)
            (l3,) = ax.semilogx(eff_ell, np.ones_like(eff_ell), "k--", linewidth=1.0)

            ax.set_ylim(ylim)
            ax.set_xlim(2, 2 * nside)
            ax.text(
                0.95,
                0.85,
                rf"$C^{{{i + 1}{j + 1}}}_{{\ell}}$",
                ha="right",
                transform=ax.transAxes,
                fontsize=11,
            )

            if j != 0:
                ax.set_yticklabels([])

            if i == nbins - 1:
                ax.set_xlabel(r"$\ell$")
            else:
                ax.set_xticklabels([])

    fig.supylabel(r"$C_{\ell} / C^{\mathrm{true}}_{\ell}$")
    axes[0, 1].legend(
        handles=[l1, l2, l3],
        labels=["Sample mean", f"Samples ({interval}th percentile)", "Truth"],
        loc="center",
        fontsize=11,
        framealpha=0.9,
    )
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    plt.show()


def plot_1pt_linear(
    pdf_linear_samples: np.ndarray,
    pdf_linear_true: np.ndarray,
    linear_bins: np.ndarray,
    interval: float = 68,
) -> None:
    r"""Plot sample/true 1-point PDF histograms, per tomographic bin, on a linear y-axis.

    Parameters
    ----------
    pdf_linear_samples : np.ndarray
        Per-sample histogram counts, shape (n_samples, nbins, n_bins - 1).
    pdf_linear_true : np.ndarray
        True histogram counts, shape (nbins, n_bins - 1).
    linear_bins : np.ndarray
        Per-bin histogram edges, as returned by `get_field_bins`.
    interval : float, optional
        Percentile interval to shade around the sample mean, by default 68.
    """
    lo_p = (100 - interval) / 2
    hi_p = (100 + interval) / 2

    nbins = pdf_linear_true.shape[0]
    fig, axes = plt.subplots(
        1,
        nbins + 1,
        figsize=(4 * nbins + 2, 4),
        gridspec_kw={"width_ratios": [4] * nbins + [0.5]},
    )

    for i in range(nbins):
        ax = axes[i]
        edges = linear_bins[i]
        mean = pdf_linear_samples[:, i, :].mean(0)
        lo = np.percentile(pdf_linear_samples[:, i, :], lo_p, axis=0)
        hi = np.percentile(pdf_linear_samples[:, i, :], hi_p, axis=0)

        ax.stairs(mean, edges, color="b", linewidth=1.5)
        ax.stairs(pdf_linear_true[i], edges, color="k", linewidth=1.5)
        ax.fill_between(
            np.repeat(edges, 2)[1:-1],
            np.repeat(np.clip(lo, 0, None), 2),
            np.repeat(hi, 2),
            color="b",
            alpha=0.3,
        )

        ax.set_title(f"Bin {i + 1}")
        ax.set_xlabel(r"$\delta_m$")
        ax.set_ylabel("Counts")
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=11)

    legend_ax = axes[-1]
    legend_ax.axis("off")
    legend_ax.legend(
        handles=[
            Line2D([0], [0], color="b", linewidth=1.5),
            Line2D([0], [0], color="k", linewidth=1.5),
            Patch(facecolor="b", alpha=0.3),
        ],
        labels=["Sample mean", "Truth", f"Samples\n({interval}th pct.)"],
        loc="center",
        fontsize=9,
        framealpha=0.9,
    )
    plt.tight_layout()
    plt.show()


def plot_1pt_log(
    pdf_log_samples: np.ndarray,
    pdf_log_true: np.ndarray,
    log_bins: np.ndarray,
    interval: float = 68,
) -> None:
    r"""Plot sample/true 1-point PDF histograms, per tomographic bin, on a log y-axis.

    Parameters
    ----------
    pdf_log_samples : np.ndarray
        Per-sample histogram counts, shape (n_samples, nbins, n_bins - 1).
    pdf_log_true : np.ndarray
        True histogram counts, shape (nbins, n_bins - 1).
    log_bins : np.ndarray
        Per-bin histogram edges, as returned by `get_field_bins`.
    interval : float, optional
        Percentile interval to shade around the sample mean, by default 68.
    """
    lo_p = (100 - interval) / 2
    hi_p = (100 + interval) / 2

    nbins = pdf_log_true.shape[0]
    fig, axes = plt.subplots(
        1,
        nbins + 1,
        figsize=(4 * nbins + 2, 4),
        gridspec_kw={"width_ratios": [4] * nbins + [0.5]},
    )

    for i in range(nbins):
        ax = axes[i]
        edges = log_bins[i]
        mean = pdf_log_samples[:, i, :].mean(0)
        lo = np.percentile(pdf_log_samples[:, i, :], lo_p, axis=0)
        hi = np.percentile(pdf_log_samples[:, i, :], hi_p, axis=0)

        ax.stairs(mean, edges, color="b", linewidth=1.5)
        ax.stairs(pdf_log_true[i], edges, color="k", linewidth=1.5)
        ax.fill_between(
            np.repeat(edges, 2)[1:-1],
            np.repeat(np.clip(lo, 0, None), 2),
            np.repeat(hi, 2),
            color="b",
            alpha=0.3,
        )

        ax.set_yscale("log")
        ax.set_title(f"Bin {i + 1}")
        ax.set_xlabel(r"$\delta_m$")
        ax.set_ylabel("Counts")
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=11)

    legend_ax = axes[-1]
    legend_ax.axis("off")
    legend_ax.legend(
        handles=[
            Line2D([0], [0], color="b", linewidth=1.5),
            Line2D([0], [0], color="k", linewidth=1.5),
            Patch(facecolor="b", alpha=0.3),
        ],
        labels=["Sample mean", "Truth", f"Samples\n({interval}th pct.)"],
        loc="center",
        fontsize=9,
        framealpha=0.9,
    )
    plt.tight_layout()
    plt.show()
