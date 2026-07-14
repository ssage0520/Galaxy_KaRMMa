import time
from datetime import timedelta

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
from blackjax.adaptation.base import get_filter_adapt_info_fn

from karmma.samplers.base import WhitenedSampler
from karmma.structs import KarmmaPosition, NUTSInfo, WhitenedKarmmaPosition


class NUTSSampler(WhitenedSampler):
    def sample(
        self,
        key,
        num_warmup,
        num_samples,
        initial_position: KarmmaPosition,
        initial_imm: np.ndarray,
        imm_shrinkage_to_previous: float = 0.0,
        step_size: float = 0.05,
        target_acceptance_rate: float = 0.65,
    ):
        """Runs NUTS, seeding window_adaptation's inverse mass matrix.

        Requires a blackjax build with `initial_inverse_mass_matrix` /
        `imm_shrinkage_to_previous` support in `window_adaptation` (the
        jax_karmma_dev environment) — stock blackjax raises a TypeError.

        `initial_position` is the physical (theta-space) position; whitening
        is computed here as the first step (see
        `WhitenedSampler._build_reparam`). `initial_imm` must be a 1-D
        diagonal IMM in BlackJax pytree-flat layout for the whitened
        position — typically `np.ones(N_full)`, since phi is already
        whitened to ~unit variance. The returned `states` is converted back
        to a physical-coordinate KarmmaPosition (theta, not phi) before
        returning, so callers never see phi-space values.

        `tuned_params["inverse_mass_matrix"]`'s theta block is phi-space-
        scaled (expected, not a bug — needed to interpret it as a diagnostic
        later), but `infos.logdensity`'s log-density values remain in true
        physical units regardless, since the wrapped log_prob below
        evaluates `self.model.log_prob` at the untransformed point, adding
        no constant (the phi -> theta map is linear with a phi-independent
        Jacobian, which doesn't affect NUTS/HMC dynamics or acceptance).
        """
        self._build_reparam(initial_position)
        sampling_position = WhitenedKarmmaPosition(
            xlm=initial_position.xlm, phi=self.theta_to_phi(initial_position.theta)
        )

        def log_prob(params: WhitenedKarmmaPosition):
            theta = self.phi_to_theta(params.phi)
            return self.model.log_prob(KarmmaPosition(xlm=params.xlm, theta=theta))

        log_prob = jax.jit(log_prob)

        t0 = time.perf_counter()

        filter_fn = get_filter_adapt_info_fn(
            info_keys={"acceptance_rate", "is_divergent", "num_integration_steps"}
        )

        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            logdensity_fn=log_prob,
            initial_step_size=step_size,
            initial_inverse_mass_matrix=initial_imm,
            imm_shrinkage_to_previous=imm_shrinkage_to_previous,
            target_acceptance_rate=target_acceptance_rate,
            is_mass_matrix_diagonal=True,
            adaptation_info_fn=filter_fn,
        )
        key, warmup_key = jax.random.split(key)
        print()
        with blackjax.progress_bar(label="Warmup (window adaptation)"):
            (wstate, tuned_params), winfo = warmup.run(
                warmup_key, sampling_position, num_steps=num_warmup
            )
            # Forces real synchronization before the with-block exits — otherwise
            # JAX's async dispatch lets this return (and the progress bar close,
            # stamped at 100%) long before the warmup scan has actually finished.
            jax.block_until_ready((wstate, tuned_params))

        t1 = time.perf_counter()
        print()

        warmup_steps = np.array(winfo.info.num_integration_steps)
        time_per_leapfrog = (t1 - t0) / warmup_steps.sum()
        mean_steps_end = warmup_steps[-20:].mean()
        time_per_sample = time_per_leapfrog * mean_steps_end
        est_sampling_time = num_samples * time_per_sample

        print(f"Warmup time: {timedelta(seconds=int(t1 - t0))}")
        print(f"Adapted step size: {tuned_params['step_size']:.4f}")
        print(
            f"Mean integration steps (warmup): {warmup_steps.mean():.1f}  |  last 20: {mean_steps_end:.1f}"
        )
        print(
            f"Mean acceptance rate (warmup): {jnp.mean(winfo.info.acceptance_rate):.4f}"
        )
        print(f"Number of divergences (warmup): {jnp.sum(winfo.info.is_divergent)}")
        print(
            f"Estimated sampling time: ~{timedelta(seconds=int(est_sampling_time))}  ({num_samples} samples × ~{time_per_sample:.1f}s/sample)"
        )

        nuts = blackjax.nuts(log_prob, **tuned_params)

        key, sample_key = jax.random.split(key)
        print()
        with blackjax.progress_bar(label="Sampling"):
            _, (states, infos) = blackjax.util.run_inference_algorithm(
                rng_key=sample_key,
                inference_algorithm=nuts,
                num_steps=num_samples,
                initial_state=wstate,
                transform=lambda state, info: (
                    state.position,
                    NUTSInfo(
                        is_divergent=info.is_divergent,
                        num_integration_steps=info.num_integration_steps,
                        acceptance_rate=info.acceptance_rate,
                        energy=info.energy,
                        logdensity=state.logdensity,
                    ),
                ),
            )
            # See the analogous comment on the warmup block — same reason.
            jax.block_until_ready((states, infos))

        t2 = time.perf_counter()
        print()

        print(f"Sampling time:    {timedelta(seconds=int(t2 - t1))}")
        print(f"Total time (w+s): {timedelta(seconds=int(t2 - t0))}")
        print(
            f"Mean integration steps: {np.array(infos.num_integration_steps).mean():.1f}"
        )
        print(f"Mean acceptance rate: {jnp.mean(infos.acceptance_rate):.4f}")
        print(f"Number of divergences: {jnp.sum(infos.is_divergent)}")

        theta = jax.vmap(self.phi_to_theta)(states.phi)
        states = KarmmaPosition(xlm=states.xlm, theta=theta)

        return states, infos, tuned_params
