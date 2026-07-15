"""Loads KaRMMa output directories for metadata.ipynb, normalizing the NUTS-era
and MCLMC-era `mcmc_metadata.h5` schemas into one dict shape per run so the
notebook can iterate over a mix of both without branching on sampler type.
"""

import os

import h5py as h5
import numpy as np
from blackjax.diagnostics import effective_sample_size

THETA_FIELDS = ("A_t", "log_T", "c", "log_R", "mu0", "a")

NUTS_ONLY_KEYS = (
    "acceptance_rate",
    "is_divergent",
    "num_integration_steps",
    "energy",
    "warmup_acceptance_rate",
    "warmup_is_divergent",
    "warmup_num_integration_steps",
)

MCLMC_ONLY_KEYS = (
    "L",
    "energy_change",
    "nonans",
)


def detect_run_type(metadata_path):
    """Classifies a mcmc_metadata.h5 file as "nuts" or "mclmc" by key presence
    (neither schema carries an explicit sampler-type field)."""
    with h5.File(metadata_path, "r") as f:
        if "acceptance_rate" in f:
            return "nuts"
        if "L" in f:
            return "mclmc"
        raise ValueError(
            f"{metadata_path}: unrecognized mcmc_metadata.h5 schema "
            "(neither 'acceptance_rate' nor 'L' present)"
        )


def _read_theta_group(f, group):
    # Stacks the 6 named datasets under `group` into one (..., 6) array.
    return np.stack([f[f"{group}/{field}"][:] for field in THETA_FIELDS], axis=-1)


def _read_scalar_or_array(dataset):
    return dataset[()] if dataset.shape == () else dataset[:]


def load_run(output_dir, mock_dg_path, label, color):
    """Loads one output directory into a run dict for the metadata notebook.

    Keys present regardless of sampler type: label, color, output_dir, type,
    seed, step_size, inverse_mass_matrix, log_prob,
    theta_reparam, xlm_real, xlm_imag, theta_samples, nbins, n_real, n_imag,
    n_samples, true_theta, ess_xlm_real, ess_xlm_imag, ess_theta, mcmc_config.
    `extra` holds whatever's specific to the detected type (NUTS_ONLY_KEYS or
    MCLMC_ONLY_KEYS). `mcmc_config` holds the full mcmc config dump (empty
    dict for pre-refactor runs that predate that group existing).
    """
    metadata_path = os.path.join(output_dir, "mcmc_metadata.h5")
    samples_path = os.path.join(output_dir, "samples.h5")
    run_type = detect_run_type(metadata_path)

    with h5.File(samples_path, "r") as f:
        xlm_real = f["xlm/real"][:]
        xlm_imag = f["xlm/imag"][:]
        theta_samples = _read_theta_group(f, "theta") if "theta" in f else None

    n_samples, nbins, n_real = xlm_real.shape
    n_imag = xlm_imag.shape[-1]

    with h5.File(metadata_path, "r") as f:
        seed = f["seed"][()]
        step_size = f["step_size"][()]
        inverse_mass_matrix = f["inverse_mass_matrix"][:]
        log_prob = f["log_prob"][:]
        theta_reparam = {
            "V": f["theta_reparam/V"][:],
            "w": f["theta_reparam/w"][:],
            "theta0": _read_theta_group(f, "theta_reparam/theta0"),  # (nbins, 6)
        }
        extra_keys = NUTS_ONLY_KEYS if run_type == "nuts" else MCLMC_ONLY_KEYS
        extra = {
            key: _read_scalar_or_array(f[key]) for key in extra_keys if key in f
        }
        mcmc_config = (
            {key: _read_scalar_or_array(f["mcmc_config"][key]) for key in f["mcmc_config"]}
            if "mcmc_config" in f
            else {}
        )

    with h5.File(mock_dg_path, "r") as f:
        true_theta = _read_theta_group(f, "true_theta")  # (nbins, 6)

    ess_xlm_real = np.array(effective_sample_size(xlm_real[np.newaxis]))
    ess_xlm_imag = np.array(effective_sample_size(xlm_imag[np.newaxis]))
    ess_theta = (
        np.array(effective_sample_size(theta_samples[np.newaxis]))
        if theta_samples is not None
        else None
    )

    return {
        "label": label,
        "color": color,
        "output_dir": output_dir,
        "type": run_type,
        "seed": seed,
        "step_size": step_size,
        "inverse_mass_matrix": inverse_mass_matrix,
        "log_prob": log_prob,
        "theta_reparam": theta_reparam,
        "xlm_real": xlm_real,
        "xlm_imag": xlm_imag,
        "theta_samples": theta_samples,
        "nbins": nbins,
        "n_real": n_real,
        "n_imag": n_imag,
        "n_samples": n_samples,
        "true_theta": true_theta,
        "ess_xlm_real": ess_xlm_real,
        "ess_xlm_imag": ess_xlm_imag,
        "ess_theta": ess_theta,
        "extra": extra,
        "mcmc_config": mcmc_config,
    }


def imm_blocks(run, key="inverse_mass_matrix"):
    """Slices a flat inverse-mass-matrix-shaped vector (`key` in `run`) into
    (xlm_real, xlm_imag, phi) blocks. The phi block is left flat (n_phi =
    nbins * 6) rather than reshaped by bin: phi is a whitened linear
    combination of all bins/parameters together, so unlike the xlm blocks it
    has no natural per-bin structure to reshape into."""
    nbins, n_real, n_imag = run["nbins"], run["n_real"], run["n_imag"]
    n_phi = nbins * len(THETA_FIELDS)
    flat = run[key]
    real_end = nbins * n_real
    imag_end = real_end + nbins * n_imag
    real_block = flat[:real_end].reshape(nbins, n_real)
    imag_block = flat[real_end:imag_end].reshape(nbins, n_imag)
    phi_block = flat[imag_end : imag_end + n_phi]
    return real_block, imag_block, phi_block
