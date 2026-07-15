import os
import sys

import jax
import jax.flatten_util
import numpy as np

jax.config.update("jax_enable_x64", True)
import h5py as h5

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from karmma import ForwardModel, KarmmaConfig
from karmma.samplers.mclmc import MCLMCSampler
from karmma.samplers.nuts import NUTSSampler
from karmma.structs import (
    KarmmaPosition,
    MclmcConfig,
    NutsConfig,
    ThetaParams,
    XlmParams,
)

configfile = sys.argv[1]
config = KarmmaConfig(configfile)

analysis = config.analysis
io = config.io
mcmc = config.mcmc

model = ForwardModel(
    dg_obs=io.dg_obs,
    N_bar=io.N_bar,
    mask=io.mask,
    CL=analysis.cl,
    alpha=analysis.alpha,
    beta=analysis.beta,
    infer_theta=mcmc.infer_theta,
    theta_fixed=io.theta_fixed,
    pixwin=analysis.pixwin,
)

print(
    f"Model initialized (nside={model.Nside}, nbins={model.Nbins}, n_modes={model.n_modes}, infer_theta={mcmc.infer_theta})."
)

# resolve random xlm init now that model shape info is available
if io.initial_position.xlm is None:
    key, init_key = jax.random.split(mcmc.key)
    xlm_full = model.make_random_xlm(init_key)
    xlm = XlmParams(real=0.3 * xlm_full.real, imag=0.3 * xlm_full.imag)
    initial_position = KarmmaPosition(xlm=xlm, theta=io.initial_position.theta)
else:
    initial_position = io.initial_position

initial_imm = np.ones(jax.flatten_util.ravel_pytree(initial_position)[0].shape[0])

if isinstance(mcmc, NutsConfig):
    print("Sampler: NUTS")
    sampler = NUTSSampler(model)
    states, infos, tuned_params, winfo = sampler.sample(
        key=mcmc.key,
        num_warmup=mcmc.num_warmup,
        num_samples=mcmc.n_samples,
        initial_position=initial_position,
        initial_imm=initial_imm,
        imm_shrinkage_to_previous=mcmc.imm_shrinkage_to_previous,
        step_size=mcmc.step_size,
        target_acceptance_rate=mcmc.target_acceptance_rate,
    )
elif isinstance(mcmc, MclmcConfig):
    print("Sampler: MCLMC")
    sampler = MCLMCSampler(model)
    states, infos, tuned_params = sampler.sample(
        key=mcmc.key,
        num_samples=mcmc.n_samples,
        initial_position=initial_position,
        initial_imm=initial_imm,
        frac_tune1=mcmc.frac_tune1,
        frac_tune2=mcmc.frac_tune2,
        frac_tune3=mcmc.frac_tune3,
        l_factor=mcmc.l_factor,
        thinning_warmup=mcmc.thinning_warmup,
        thinning_sampling=mcmc.thinning_sampling,
        desired_energy_var=mcmc.desired_energy_var,
    )
else:
    raise ValueError(f"Unrecognized mcmc config type: {type(mcmc).__name__}")


os.makedirs(io.io_dir, exist_ok=True)

with h5.File(os.path.join(io.io_dir, "samples.h5"), "w") as f:
    xlm_grp = f.create_group("xlm")
    xlm_grp.create_dataset("real", data=np.array(states.xlm.real))
    xlm_grp.create_dataset("imag", data=np.array(states.xlm.imag))

    if states.theta is not None:
        theta_grp = f.create_group("theta")
        for field in ThetaParams._fields:
            theta_grp.create_dataset(field, data=np.array(getattr(states.theta, field)))

with h5.File(os.path.join(io.io_dir, "mcmc_metadata.h5"), "w") as f:
    # run info
    f["seed"] = np.array(mcmc.seed)

    if isinstance(mcmc, NutsConfig):
        # blackjax's window_adaptation returns tuned params as a plain dict
        f["step_size"] = np.array(tuned_params["step_size"])
        f["inverse_mass_matrix"] = np.array(tuned_params["inverse_mass_matrix"])

        # sampling diagnostics
        f["acceptance_rate"] = np.array(infos.acceptance_rate)
        f["is_divergent"] = np.array(infos.is_divergent)
        f["num_integration_steps"] = np.array(infos.num_integration_steps)
        f["energy"] = np.array(infos.energy)
        f["log_prob"] = np.array(infos.logdensity)

        # warmup diagnostics
        f["warmup_acceptance_rate"] = np.array(winfo.info.acceptance_rate)
        f["warmup_is_divergent"] = np.array(winfo.info.is_divergent)
        f["warmup_num_integration_steps"] = np.array(winfo.info.num_integration_steps)
    elif isinstance(mcmc, MclmcConfig):
        # MCLMCAdaptationState is a NamedTuple
        f["L"] = np.array(tuned_params.L)
        f["step_size"] = np.array(tuned_params.step_size)
        f["inverse_mass_matrix"] = np.array(tuned_params.inverse_mass_matrix)

        # sampling diagnostics (RMS-aggregated over each block of `thinning_sampling` raw steps)
        f["energy_change"] = np.array(infos.energy_change)
        f["nonans"] = np.array(infos.nonans)
        f["log_prob"] = np.array(infos.logdensity)
    else:
        raise ValueError(f"Unrecognized mcmc config type: {type(mcmc).__name__}")

    # full mcmc config, for reproducibility
    mcmc_config_grp = f.create_group("mcmc_config")
    for field in type(mcmc)._fields:
        if field == "key":
            continue  # a raw PRNGKey isn't independently useful; `seed` (above) already
            # lets jax.random.PRNGKey(seed) reconstruct it deterministically
        mcmc_config_grp.create_dataset(field, data=np.array(getattr(mcmc, field)))

    # theta eigenbasis whitening transform (needed to interpret the
    # phi-space theta block of inverse_mass_matrix above)
    reparam_grp = f.create_group("theta_reparam")
    reparam_grp.create_dataset("V", data=np.array(sampler.V))
    reparam_grp.create_dataset("w", data=np.array(sampler.w))
    theta0_grp = reparam_grp.create_group("theta0")
    for field in ThetaParams._fields:
        theta0_grp.create_dataset(field, data=np.array(getattr(sampler.theta0, field)))

    # fixed bias parameters (only when not sampling theta)
    if not mcmc.infer_theta:
        grp = f.create_group("theta_fixed")
        for field in ThetaParams._fields:
            grp.create_dataset(field, data=np.array(getattr(io.theta_fixed, field)))

print("Samples and metadata saved.")
