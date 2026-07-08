import os
import sys

import jax
import jax.flatten_util
import numpy as np

jax.config.update("jax_enable_x64", True)
import h5py as h5

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from karmma import KarmmaConfig, KarmmaSampler
from karmma.structs import (
    KarmmaPosition,
    ThetaParams,
    WhitenedKarmmaPosition,
    XlmParams,
)

configfile = sys.argv[1]
config = KarmmaConfig(configfile)

analysis = config.analysis
io = config.io
mcmc = config.mcmc

sampler = KarmmaSampler(
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
    f"Sampler initialized (nside={sampler.Nside}, nbins={sampler.Nbins}, n_modes={sampler.n_modes}, infer_theta={mcmc.infer_theta})."
)

# resolve random xlm init now that sampler shape info is available
if io.initial_position.xlm is None:
    key, init_key = jax.random.split(mcmc.key)
    xlm_full = sampler.make_random_xlm(init_key)
    xlm = XlmParams(real=0.3 * xlm_full.real, imag=0.3 * xlm_full.imag)
    initial_position = KarmmaPosition(xlm=xlm, theta=io.initial_position.theta)
else:
    initial_position = io.initial_position

print(
    "Computing dense theta covariance (Schur+CG) for eigenbasis reparametrization ..."
)
dense_theta_matrix = sampler.dense_theta_imm(initial_position)
sampler.build_theta_reparam(dense_theta_matrix, initial_position.theta)
sampling_position = WhitenedKarmmaPosition(
    xlm=initial_position.xlm, phi=sampler.theta_to_phi(initial_position.theta)
)
initial_imm = np.ones(jax.flatten_util.ravel_pytree(sampling_position)[0].shape[0])

states, infos, tuned_params, warmup_calls = sampler.sample(
    key=mcmc.key,
    num_samples=mcmc.n_samples,
    initial_position=sampling_position,
    initial_imm=initial_imm,
    frac_tune1=mcmc.frac_tune1,
    frac_tune2=mcmc.frac_tune2,
    frac_tune3=mcmc.frac_tune3,
    l_factor=mcmc.l_factor,
    thinning_warmup=mcmc.thinning_warmup,
    thinning_sampling=mcmc.thinning_sampling,
    desired_energy_var=mcmc.desired_energy_var,
)


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
    f["L"] = np.array(tuned_params.L)
    f["step_size"] = np.array(tuned_params.step_size)
    f["inverse_mass_matrix"] = np.array(tuned_params.inverse_mass_matrix)

    # sampling diagnostics (RMS-aggregated over each block of `thinning_sampling` raw steps)
    f["energy_change"] = np.array(infos.energy_change)
    f["kinetic_change"] = np.array(infos.kinetic_change)
    f["nonans"] = np.array(infos.nonans)
    f["log_prob"] = np.array(infos.logdensity)

    # warmup provenance
    f["warmup_calls"] = np.array(warmup_calls)
    f["thinning_warmup"] = np.array(mcmc.thinning_warmup)
    f["thinning_sampling"] = np.array(mcmc.thinning_sampling)
    f["initial_imm"] = np.array(initial_imm)

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
