from typing import Any, NamedTuple

import jax.numpy as jnp


class XlmParams(NamedTuple):
    real: jnp.ndarray  # (Nbins, n_real)
    imag: jnp.ndarray  # (Nbins, n_imag)


class ThetaParams(NamedTuple):
    A_t: jnp.ndarray  # (Nbins,)  log-density threshold
    log_T: jnp.ndarray  # log temperature (sigmoid sharpness)
    c: jnp.ndarray  # Gaussian smooth coupling amplitude (unconstrained)
    log_R: jnp.ndarray  # log(R/θ_pix), dimensionless pixel-scale units
    mu0: jnp.ndarray  # variance depletion offset (intercept, unconstrained)
    a: jnp.ndarray  # variance depletion slope vs delta_eff (unconstrained)


class KarmmaPosition(NamedTuple):
    xlm: XlmParams | None  # None when awaiting random init (resolved in run_karmma.py)
    theta: ThetaParams | None = None


class WhitenedKarmmaPosition(NamedTuple):
    xlm: XlmParams
    phi: jnp.ndarray  # (n_theta,) flat whitened bias parameters


class MCLMCInfo(NamedTuple):
    logdensity: jnp.ndarray
    energy_change: jnp.ndarray
    kinetic_change: jnp.ndarray
    nonans: jnp.ndarray


class McmcConfig(NamedTuple):
    n_samples: int
    key: Any
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
    nbins: int
    nside: int
    alpha: Any
    beta: Any
    cl: Any
    pixwin: Any


class IoConfig(NamedTuple):
    datafile: str
    io_dir: str
    dg_obs: Any
    mask: Any
    N_bar: Any
    initial_position: KarmmaPosition
    theta_fixed: ThetaParams | None  # None when infer_theta=True
