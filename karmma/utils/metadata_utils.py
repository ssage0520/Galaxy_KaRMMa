"""Loads KaRMMa output directories for metadata.ipynb.

Normalizes the NUTS-era and MCLMC-era `mcmc_metadata.h5` schemas into one
dict shape per run so the notebook can iterate over a mix of both without
branching on sampler type.
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


def detect_run_type(metadata_path: str) -> str:
    """Classify an mcmc_metadata.h5 file as "nuts" or "mclmc" by key presence.

    Neither schema carries an explicit sampler-type field, so this checks
    for a key unique to each.

    Parameters
    ----------
    metadata_path : str
        Path to an mcmc_metadata.h5 file.

    Returns
    -------
    str
        "nuts" or "mclmc".

    Raises
    ------
    ValueError
        If neither schema's marker key is present.
    """
    with h5.File(metadata_path, "r") as f:
        if "acceptance_rate" in f:
            return "nuts"
        if "L" in f:
            return "mclmc"
        raise ValueError(
            f"{metadata_path}: unrecognized mcmc_metadata.h5 schema "
            "(neither 'acceptance_rate' nor 'L' present)"
        )


def _read_theta_group(f: h5.File, group: str) -> np.ndarray:
    """Stack the 6 named theta datasets under `group` into one array.

    Parameters
    ----------
    f : h5py.File
        Open HDF5 file to read from.
    group : str
        Path within `f` holding one dataset per `THETA_FIELDS` entry.

    Returns
    -------
    np.ndarray
        Stacked array, shape (..., 6).
    """
    return np.stack([f[f"{group}/{field}"][:] for field in THETA_FIELDS], axis=-1)


def _read_scalar_or_array(dataset: h5.Dataset) -> int | float | bool | np.ndarray:
    """Read an HDF5 dataset, converting scalars to native Python types.

    Parameters
    ----------
    dataset : h5py.Dataset
        Dataset to read; may be scalar-shaped or array-shaped.

    Returns
    -------
    int, float, bool, or np.ndarray
        `dataset.item()` for scalars (native Python type, not a numpy
        scalar), `dataset[:]` for arrays.
    """
    return dataset[()].item() if dataset.shape == () else dataset[:]


def load_run(output_dir: str, mock_dg_path: str, label: str, color: str) -> dict:
    """Load one output directory into a run dict for the metadata notebook.

    Parameters
    ----------
    output_dir : str
        Directory containing `samples.h5` and `mcmc_metadata.h5`.
    mock_dg_path : str
        Path to the mock datafile holding `true_theta`.
    label : str
        Plot label for this run.
    color : str
        Plot color for this run.

    Returns
    -------
    dict
        Run data. Keys present regardless of sampler type: `label`,
        `color`, `output_dir`, `type` ("nuts" or "mclmc"), `seed`,
        `step_size`, `inverse_mass_matrix`, `log_prob`, `theta_reparam`,
        `theta_samples`, `nbins`, `n_real`, `n_imag`, `n_samples`,
        `true_theta`, `ess_theta`. `xlm_real`, `xlm_imag`, `ess_xlm_real`,
        `ess_xlm_imag` are `None` for runs saved with `save_maps=False`
        (no `xlm` group in `samples.h5`). `extra` holds whatever's
        specific to the detected type (`NUTS_ONLY_KEYS` or
        `MCLMC_ONLY_KEYS`). `mcmc_config` holds the full mcmc config dump
        (empty dict for pre-refactor runs that predate that group
        existing).
    """
    metadata_path = os.path.join(output_dir, "mcmc_metadata.h5")
    samples_path = os.path.join(output_dir, "samples.h5")
    run_type = detect_run_type(metadata_path)

    with h5.File(samples_path, "r") as f:
        has_xlm = "xlm" in f
        xlm_real = f["xlm/real"][:] if has_xlm else None
        xlm_imag = f["xlm/imag"][:] if has_xlm else None
        theta_samples = _read_theta_group(f, "theta")

    n_samples = theta_samples.shape[0]

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
        nbins = int(f["model_shape/nbins"][()])
        n_real = int(f["model_shape/n_real"][()])
        n_imag = int(f["model_shape/n_imag"][()])

    with h5.File(mock_dg_path, "r") as f:
        true_theta = _read_theta_group(f, "true_theta")  # (nbins, 6)

    ess_xlm_real = (
        np.array(effective_sample_size(xlm_real[np.newaxis])) if has_xlm else None
    )
    ess_xlm_imag = (
        np.array(effective_sample_size(xlm_imag[np.newaxis])) if has_xlm else None
    )
    ess_theta = np.array(effective_sample_size(theta_samples[np.newaxis]))

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


def imm_blocks(
    run: dict, key: str = "inverse_mass_matrix"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice a flat IMM-shaped vector into (xlm_real, xlm_imag, phi) blocks.

    Parameters
    ----------
    run : dict
        Run dict, as returned by `load_run`.
    key : str, optional
        Key in `run` holding the flat vector to slice, by default
        "inverse_mass_matrix".

    Returns
    -------
    real_block : np.ndarray
        xlm real-part block, reshaped to (nbins, n_real).
    imag_block : np.ndarray
        xlm imaginary-part block, reshaped to (nbins, n_imag).
    phi_block : np.ndarray
        Whitened theta (phi) block, left flat (length nbins * 6) since
        phi is a linear combination across all bins/parameters together
        and has no natural per-bin structure to reshape into.
    """
    nbins, n_real, n_imag = run["nbins"], run["n_real"], run["n_imag"]
    n_phi = nbins * len(THETA_FIELDS)
    flat = run[key]
    real_end = nbins * n_real
    imag_end = real_end + nbins * n_imag
    real_block = flat[:real_end].reshape(nbins, n_real)
    imag_block = flat[real_end:imag_end].reshape(nbins, n_imag)
    phi_block = flat[imag_end : imag_end + n_phi]
    return real_block, imag_block, phi_block
