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


class NUTSInfo(NamedTuple):
    is_divergent: jnp.ndarray
    num_integration_steps: jnp.ndarray
    acceptance_rate: jnp.ndarray
    energy: jnp.ndarray
    logdensity: jnp.ndarray


class McmcConfig(NamedTuple):
    n_warmup: int
    n_samples: int
    key: Any
    seed: int
    step_size: float
    target_acceptance: float
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
