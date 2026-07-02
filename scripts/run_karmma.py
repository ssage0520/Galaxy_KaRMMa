import os
import sys
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from karmma import KarmmaConfig, KarmmaSampler
from karmma.structs import KarmmaPosition, ThetaParams, XlmParams
import h5py as h5

configfile = sys.argv[1]
config     = KarmmaConfig(configfile)

analysis = config.analysis
io       = config.io
mcmc     = config.mcmc

sampler = KarmmaSampler(
    dg_obs      = io.dg_obs,
    N_bar       = io.N_bar,
    mask        = io.mask,
    CL          = analysis.cl,
    alpha       = analysis.alpha,
    beta        = analysis.beta,
    infer_theta = mcmc.infer_theta,
    theta_fixed = io.theta_fixed,
    pixwin      = analysis.pixwin
)

print(f'Sampler initialized (nside={sampler.Nside}, nbins={sampler.Nbins}, n_modes={sampler.n_modes}, infer_theta={mcmc.infer_theta}).')

# resolve random xlm init now that sampler shape info is available
if io.initial_position.xlm is None:
    key, init_key = jax.random.split(mcmc.key)
    xlm_full = sampler.make_random_xlm(init_key)
    xlm = XlmParams(real=0.3 * xlm_full.real, imag=0.3 * xlm_full.imag)
    initial_position = KarmmaPosition(xlm=xlm, theta=io.initial_position.theta)
else:
    initial_position = io.initial_position

states, infos, mcmc_parameters, winfo = sampler.sample(
    key                    = mcmc.key,
    num_warmup             = mcmc.n_warmup,
    num_samples            = mcmc.n_samples,
    step_size              = mcmc.step_size,
    target_acceptance_rate = mcmc.target_acceptance,
    initial_position       = initial_position
)


os.makedirs(io.io_dir, exist_ok=True)

with h5.File(os.path.join(io.io_dir, 'samples.h5'), 'w') as f:
    xlm_grp = f.create_group('xlm')
    xlm_grp.create_dataset('real', data=np.array(states.xlm.real))
    xlm_grp.create_dataset('imag', data=np.array(states.xlm.imag))

    if states.theta is not None:
        theta_grp = f.create_group('theta')
        for field in ThetaParams._fields:
            theta_grp.create_dataset(field, data=np.array(getattr(states.theta, field)))

with h5.File(os.path.join(io.io_dir, 'mcmc_metadata.h5'), 'w') as f:
    # run info
    f['seed']      = np.array(mcmc.seed)
    f['step_size'] = np.array(mcmc_parameters['step_size'])

    # sampling diagnostics
    f['acceptance_rate']       = np.array(infos.acceptance_rate)
    f['is_divergent']          = np.array(infos.is_divergent)
    f['num_integration_steps'] = np.array(infos.num_integration_steps)
    f['energy']                = np.array(infos.energy)

    # warmup diagnostics
    f['warmup_acceptance_rate']       = np.array(winfo.info.acceptance_rate)
    f['warmup_is_divergent']          = np.array(winfo.info.is_divergent)
    f['warmup_num_integration_steps'] = np.array(winfo.info.num_integration_steps)
    f['inverse_mass_matrix']          = np.array(mcmc_parameters['inverse_mass_matrix'])

    f['log_prob'] = np.array(infos.logdensity)

    # fixed bias parameters (only when not sampling theta)
    if not mcmc.infer_theta:
        grp = f.create_group('theta_fixed')
        for field in ThetaParams._fields:
            grp.create_dataset(field, data=np.array(getattr(io.theta_fixed, field)))

print('Samples and metadata saved.')
