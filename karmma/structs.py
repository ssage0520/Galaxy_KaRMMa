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
        log(R / theta_pix), dimensionless pixel-scale units.
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
    theta : ThetaParams or None
        Bias/nuisance parameters, when sampled jointly with `xlm`;
        `None` when `theta` is held fixed instead.
    """

    xlm: XlmParams | None
    theta: ThetaParams | None = None


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
    infer_theta : bool
        Whether `theta` is sampled jointly with `xlm` (`True`) or held
        fixed (`False`).
    """

    n_samples: int
    key: jax.Array
    seed: int
    num_warmup: int
    step_size: float
    target_acceptance_rate: float
    imm_shrinkage_to_previous: float
    infer_theta: bool


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
    infer_theta : bool
        Whether `theta` is sampled jointly with `xlm` (`True`) or held
        fixed (`False`).
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
    infer_theta: bool


class AnalysisConfig(NamedTuple):
    """Configuration for the forward model's survey/statistics setup.

    Attributes
    ----------
    nbins : int
        Number of tomographic bins.
    nside : int
        HEALPix resolution parameter.
    alpha : np.ndarray
        Per-bin shifted-lognormal shape parameter.
    beta : np.ndarray
        Per-bin shifted-lognormal scale parameter.
    cl : np.ndarray
        Per-bin-pair angular power spectra.
    pixwin : np.ndarray or None
        Pixel window function, or `None` if not applied.
    """

    nbins: int
    nside: int
    alpha: np.ndarray
    beta: np.ndarray
    cl: np.ndarray
    pixwin: np.ndarray | None


class IoConfig(NamedTuple):
    """Configuration for input/output: data paths and loaded arrays.

    Attributes
    ----------
    datafile : str
        Path to the input HDF5 datafile.
    io_dir : str
        Directory to write sampling output to.
    dg_obs : np.ndarray
        Observed galaxy overdensity maps.
    mask : np.ndarray
        Survey mask.
    N_bar : np.ndarray
        Mean galaxy count per pixel, per bin.
    initial_position : KarmmaPosition
        Starting position for sampling.
    theta_fixed : ThetaParams or None
        Fixed bias parameters; `None` when `infer_theta=True`.
    """

    datafile: str
    io_dir: str
    dg_obs: np.ndarray
    mask: np.ndarray
    N_bar: np.ndarray
    initial_position: KarmmaPosition
    theta_fixed: ThetaParams | None
