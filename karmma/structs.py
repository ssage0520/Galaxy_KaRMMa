"""NamedTuple data structures shared across the model, samplers, and config."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class XlmParams(NamedTuple):
    """Real/imaginary harmonic coefficients of the underlying whitened Gaussian field.

    Attributes
    ----------
    real : jnp.ndarray
        Real part, shape (Nbins, n_real).
    imag : jnp.ndarray
        Imaginary part, shape (Nbins, n_imag).
    """

    real: jnp.ndarray
    imag: jnp.ndarray


class ThetaParams(NamedTuple):
    """Per-tomographic-bin bias/nuisance parameters of the forward model.

    Attributes
    ----------
    A_t : jnp.ndarray
        Log-density threshold, shape (Nbins,).
    log_T : jnp.ndarray
        Log temperature (sigmoid sharpness).
    c : jnp.ndarray
        Gaussian smooth coupling amplitude (unconstrained).
    log_R : jnp.ndarray
        log(R / pixel_size), dimensionless pixel-scale units (`pixel_size`
        being `ForwardModel.pixel_size`).
    mu0 : jnp.ndarray
        Variance depletion offset (intercept, unconstrained).
    a : jnp.ndarray
        Variance depletion slope vs. delta_eff (unconstrained).
    """

    A_t: jnp.ndarray
    log_T: jnp.ndarray
    c: jnp.ndarray
    log_R: jnp.ndarray
    mu0: jnp.ndarray
    a: jnp.ndarray


class KarmmaPosition(NamedTuple):
    """A sampler position: harmonic-space field coefficients plus (optional) bias parameters.

    Attributes
    ----------
    xlm : XlmParams or None
        Field coefficients; `None` when awaiting random initialization
        (resolved in `run_karmma.py`).
    theta : ThetaParams
        Bias/nuisance parameters, sampled jointly with `xlm`.
    """

    xlm: XlmParams | None
    theta: ThetaParams


class WhitenedKarmmaPosition(NamedTuple):
    """A sampler position with `theta` replaced by its whitened, eigenbasis phi representation.

    Attributes
    ----------
    xlm : XlmParams
        Field coefficients (unchanged from `KarmmaPosition`).
    phi : jnp.ndarray
        Flat whitened bias parameters, shape (n_theta,).
    """

    xlm: XlmParams
    phi: jnp.ndarray


class MCLMCInfo(NamedTuple):
    """Per-sample MCLMC diagnostics, aggregated over each thinning block.

    Attributes
    ----------
    logdensity : jnp.ndarray
        Log density of the sampled position.
    energy_change : jnp.ndarray
        RMS-aggregated per-step energy-change diagnostic.
    nonans : jnp.ndarray
        Fraction of steps in the block that were NaN-free.
    """

    logdensity: jnp.ndarray
    energy_change: jnp.ndarray
    nonans: jnp.ndarray


class NUTSInfo(NamedTuple):
    """Per-sample NUTS diagnostics.

    Attributes
    ----------
    is_divergent : jnp.ndarray
        Whether the transition diverged.
    num_integration_steps : jnp.ndarray
        Number of leapfrog integration steps taken.
    acceptance_rate : jnp.ndarray
        Transition acceptance rate.
    energy : jnp.ndarray
        Hamiltonian energy of the sampled state.
    logdensity : jnp.ndarray
        Log density of the sampled position.
    """

    is_divergent: jnp.ndarray
    num_integration_steps: jnp.ndarray
    acceptance_rate: jnp.ndarray
    energy: jnp.ndarray
    logdensity: jnp.ndarray


class NutsConfig(NamedTuple):
    """MCMC configuration for the NUTS sampler backend.

    Attributes
    ----------
    n_samples : int
        Number of post-warmup samples to draw.
    key : jax.Array
        JAX PRNG key (`jax.random.PRNGKey(seed)`).
    seed : int
        Integer seed `key` was constructed from.
    num_warmup : int
        Number of window-adaptation warmup steps.
    step_size : float
        Initial step size for warmup.
    target_acceptance_rate : float
        Target acceptance rate for step-size adaptation.
    imm_shrinkage_to_previous : float
        Pseudo-count controlling shrinkage of each warmup window's
        adapted inverse mass matrix toward the previous window's.
    """

    n_samples: int
    key: jax.Array
    seed: int
    num_warmup: int
    step_size: float
    target_acceptance_rate: float
    imm_shrinkage_to_previous: float


class MclmcConfig(NamedTuple):
    """MCMC configuration for the MCLMC sampler backend.

    Attributes
    ----------
    n_samples : int
        Number of samples actually saved (post-thinning).
    key : jax.Array
        JAX PRNG key (`jax.random.PRNGKey(seed)`).
    seed : int
        Integer seed `key` was constructed from.
    frac_tune1 : float
        Fraction of warmup spent on phase 1 (step-size dual averaging).
    frac_tune2 : float
        Fraction of warmup spent on phase 2 (diagonal preconditioning).
    frac_tune3 : float
        Fraction of warmup spent on phase 3 (tuning `L` via effective
        sample size).
    l_factor : float
        Factor scaling the estimated autocorrelation length to obtain
        the momentum decoherence length `L`.
    thinning_warmup : int
        Thinning applied to phase 3 only (phases 1+2 always run unthinned).
    thinning_sampling : int
        Thinning applied during the final sampling phase.
    desired_energy_var : float
        Target per-step energy-change variance for step-size dual averaging.
    """

    n_samples: int
    key: jax.Array
    seed: int
    frac_tune1: float
    frac_tune2: float
    frac_tune3: float
    l_factor: float
    thinning_warmup: int
    thinning_sampling: int
    desired_energy_var: float


class AnalysisConfig(NamedTuple):
    """Configuration for the forward model's point-transform/survey setup.

    Attributes
    ----------
    nbins : int
        Number of tomographic bins.
    nside : int
        HEALPix resolution parameter.
    lbda : np.ndarray
        Point-transform parameters, shape (gn_order, nbins): rows (alpha,
        beta) for gn_order=2, or (a, b, c) for gn_order=3.
    gn_order : int
        Point-transform order — 2 or 3.
    """

    nbins: int
    nside: int
    lbda: np.ndarray
    gn_order: int


class IoConfig(NamedTuple):
    """Configuration for input/output: data paths and loaded arrays.

    Attributes
    ----------
    input_dir : str
        Directory `datafile`/`init_file`/`theta_file`/`cl_file`/`pixwin`
        are all resolved relative to.
    output_dir : str
        Directory to write sampling output to.
    datafile : str
        Resolved path to the input HDF5 datafile.
    dg_obs : np.ndarray
        Observed galaxy overdensity maps, shape (Nbins, npix).
    mask : np.ndarray
        Survey mask, shape (npix,); cast to bool.
    N_bar : np.ndarray
        Average galaxy count per pixel, per bin — averaged across the
        full sky, not just the observed/masked region.
    cl : np.ndarray
        Target (physical, non-Gaussian) angular power spectra, shape
        (Nbins, Nbins, gen_lmax + 1); off-diagonal entries `cl[i, j]`
        (i != j) are cross-power spectra between bins i and j.
    pixwin : np.ndarray or None
        Pixel window function, indexed by multipole (length >= lmax + 1),
        or `None` if not applied.
    initial_position : KarmmaPosition
        Starting position for sampling.
    """

    input_dir: str
    output_dir: str
    datafile: str
    dg_obs: np.ndarray
    mask: np.ndarray
    N_bar: np.ndarray
    cl: np.ndarray
    pixwin: np.ndarray | None
    initial_position: KarmmaPosition
