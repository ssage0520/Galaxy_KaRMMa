import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import skyproj
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def plot_map(
    dm_map, mask, minmax, cmap="viridis", cb_label=r"$\delta_m$", title=None, ax=None
):
    masked_map = dm_map.copy()
    masked_map[~mask.astype(bool)] = hp.UNSEEN
    vmin, vmax = minmax
    sp = skyproj.DESSkyproj(ax=ax)
    sp.draw_hpxmap(masked_map, vmin=vmin, vmax=vmax, cmap=cmap)
    sp.draw_inset_colorbar(label=cb_label)
    if title is not None:
        sp.ax.set_title(title, pad=25)
    return sp


def plot_dm_comparison(dm_true, dm_mean, mask, n_samples=None, cmap="viridis"):
    """Compare true dm map with sample-mean dm map.

    dm_true : (nbins, npix)
    dm_mean : (nbins, npix)  — pre-averaged over samples
    n_samples : int, optional — shown in the title
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


def plot_corr(corr_samples, corr_true, bin_centres, ylim=None, interval=68):
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


def plot_pseudo_cl(cl_samples, cl_true, eff_ell, nside, ylim=None, interval=68):
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


def plot_1pt_linear(pdf_linear_samples, pdf_linear_true, linear_bins, interval=68):
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


def plot_1pt_log(pdf_log_samples, pdf_log_true, log_bins, interval=68):
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
