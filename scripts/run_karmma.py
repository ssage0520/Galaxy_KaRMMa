"""Run KaRMMa MCMC sampling (NUTS or MCLMC, chosen by config) and write output to disk.

Reads an analysis/IO/MCMC configuration from a YAML file, builds a
`ForwardModel`, dispatches to the configured sampler backend, and writes
posterior draws and sampler diagnostics to the config's `output_dir`.

Examples
--------
>>> python scripts/run_karmma.py config/mclmc.yaml
"""

import os
import sys

import jax
import jax.flatten_util
import numpy as np

jax.config.update("jax_enable_x64", True)
import h5py as h5

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from karmma import ForwardModel, KarmmaConfig
from karmma.initialization import init_xlm, init_xlm_theta_free, refine_theta
from karmma.samplers import MCLMCSampler, NUTSSampler
from karmma.structs import (
    KarmmaPosition,
    MclmcConfig,
    NutsConfig,
    ThetaParams,
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
    CL=io.cl,
    lbda=analysis.lbda,
    gn_order=analysis.gn_order,
    pixwin=io.pixwin,
)

print(
    f"Model initialized (nside={model.Nside}, nbins={model.Nbins}, n_modes={model.n_modes})."
)

# Whatever `init_position` did not supply is built here rather than at config
# time, since every builder needs a ForwardModel.
xlm = io.initial_position.xlm
theta = io.initial_position.theta

# fold_in keeps these decorrelated from the stream handed to the sampler while
# staying reproducible from mcmc.seed alone.
if xlm is None and theta is None:
    xlm = init_xlm_theta_free(model, jax.random.fold_in(mcmc.key, 1))
    theta = refine_theta(model, xlm, io.theta_guess)
elif xlm is None:
    xlm = init_xlm(model, theta, jax.random.fold_in(mcmc.key, 1))
elif theta is None:
    theta = refine_theta(model, xlm, io.theta_guess)

for extra_pass in range(io.init_passes):
    print(f"init pass {extra_pass + 1} of {io.init_passes}")
    xlm = init_xlm(model, theta, jax.random.fold_in(mcmc.key, 2 + extra_pass))
    theta = refine_theta(model, xlm, theta)

initial_position = KarmmaPosition(xlm=xlm, theta=theta)

initial_imm = np.ones(jax.flatten_util.ravel_pytree(initial_position)[0].shape[0])

if isinstance(mcmc, NutsConfig):
    print("Sampler: NUTS")
    sampler = NUTSSampler(model)
    if io.whitening is not None:
        sampler.set_whitening(*io.whitening)
    states, infos, tuned_params, winfo = sampler.sample(
        key=mcmc.key,
        num_warmup=mcmc.num_warmup,
        num_samples=mcmc.n_samples,
        initial_position=initial_position,
        initial_imm=initial_imm,
        imm_shrinkage_to_previous=mcmc.imm_shrinkage_to_previous,
        step_size=mcmc.step_size,
        target_acceptance_rate=mcmc.target_acceptance_rate,
        save_xlm=io.save_maps,
    )
elif isinstance(mcmc, MclmcConfig):
    print("Sampler: MCLMC")
    sampler = MCLMCSampler(model)
    if io.whitening is not None:
        sampler.set_whitening(*io.whitening)
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
        save_xlm=io.save_maps,
    )
else:
    raise ValueError(f"Unrecognized mcmc config type: {type(mcmc).__name__}")


os.makedirs(io.output_dir, exist_ok=True)

with h5.File(os.path.join(io.output_dir, "samples.h5"), "w") as f:
    if io.save_maps:
        xlm_grp = f.create_group("xlm")
        xlm_grp.create_dataset("real", data=np.array(states.xlm.real))
        xlm_grp.create_dataset("imag", data=np.array(states.xlm.imag))

    theta_grp = f.create_group("theta")
    for field in ThetaParams._fields:
        theta_grp.create_dataset(field, data=np.array(getattr(states.theta, field)))

with h5.File(os.path.join(io.output_dir, "mcmc_metadata.h5"), "w") as f:
    # run info
    f["seed"] = np.array(mcmc.seed)
    f["init_passes"] = np.array(io.init_passes)
    f["whitening_supplied"] = np.array(io.whitening is not None)

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

    # model shape info, saved regardless of save_maps — needed to interpret
    # the xlm block of inverse_mass_matrix above even when xlm samples aren't saved
    model_shape_grp = f.create_group("model_shape")
    model_shape_grp["nbins"] = np.array(model.Nbins)
    model_shape_grp["n_real"] = np.array(model.n_real)
    model_shape_grp["n_imag"] = np.array(model.n_imag)

print("Samples and metadata saved.")
