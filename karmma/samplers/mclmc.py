import time
from datetime import timedelta

import blackjax
import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState

from karmma.samplers.base import WhitenedSampler
from karmma.structs import KarmmaPosition, MCLMCInfo, WhitenedKarmmaPosition


class MCLMCSampler(WhitenedSampler):
    def sample(
        self,
        key,
        num_samples,
        initial_position: KarmmaPosition,
        initial_imm: np.ndarray,
        frac_tune1: float = 0.1,
        frac_tune2: float = 0.3,
        frac_tune3: float = 0.1,
        l_factor: float = 0.4,
        desired_energy_var: float = 5e-4,
        thinning_warmup: int = 5,
        thinning_sampling: int = 5,
    ):
        """Runs MCLMC, seeding its diagonal preconditioner from `initial_imm`.

        Requires the jax_karmma_dev blackjax build (mclmc's call-time kernel
        args, mclmc_find_L_and_step_size's logdensity_fn/l_factor kwargs).
        `initial_position` is the physical (theta-space) position — whitening
        is computed here as the first step (see WhitenedSampler._build_reparam).
        `initial_imm` must be a 1-D diagonal IMM in BlackJax pytree-flat
        layout for the whitened position, typically `np.ones(N_full)`, since
        phi is already whitened to ~unit variance. `num_samples` is the
        number of samples actually saved (post-thinning), not a raw
        integrator-step budget.

        Warmup runs as two chained `mclmc_find_L_and_step_size` calls, since
        phases 1+2 and phase 3 want opposite thinning:

        - Phases 1+2 (step size + diagonal IMM) always run unthinned — they
          only carry O(dim) running accumulators, so thinning gains nothing
          and only coarsens the step-size feedback.
        - Phase 3 (L via effective-sample-size) is thinned by
          `thinning_warmup`, since its FFT-based ESS calculation over the
          full position is what actually blows up memory at scale, while its
          estimate is fairly insensitive to draw spacing.

        Call 2 is seeded from call 1's tuned `state`/`params` (not
        `init_state`/`initial_params`) so phase 3 continues the chain instead
        of restarting cold.
        """
        self._build_reparam(initial_position)
        sampling_position = WhitenedKarmmaPosition(
            xlm=initial_position.xlm, phi=self.theta_to_phi(initial_position.theta)
        )

        def log_prob(params: WhitenedKarmmaPosition):
            theta = self.phi_to_theta(params.phi)
            return self.model.log_prob(KarmmaPosition(xlm=params.xlm, theta=theta))

        # jax.jit matters here since mclmc.init isn't itself jit-decorated;
        # elsewhere the enclosing lax.scan compiles everything regardless.
        log_prob = jax.jit(log_prob)
        dim = blackjax.util.pytree_size(sampling_position)

        t0 = time.perf_counter()

        key, key_init, key_warmup1, key_warmup2, key_sample = jax.random.split(key, 5)

        def sample_info(info):
            """Aggregates raw per-step MCLMC info over a thinning block.

            logdensity takes the last raw step's value — exactly matching
            the block's final (saved) position — rather than an aggregate;
            energy_change/kinetic_change are genuinely mean-zero step-error
            diagnostics where an RMS magnitude is meaningful, but RMS-ing
            logdensity (uniformly large-magnitude and negative) collapses to
            |logdensity|, silently flipping its sign. nonans is a 0/1
            indicator, so it needs a mean (fraction clean), not RMS — RMS of
            a 0/1 array is sqrt(fraction), not the fraction itself.
            """
            return info._replace(
                logdensity=info.logdensity[-1],
                energy_change=(info.energy_change**2).mean() ** 0.5,
                kinetic_change=(info.kinetic_change**2).mean() ** 0.5,
                nonans=info.nonans.mean(),
            )

        init_state = blackjax.mcmc.mclmc.init(
            position=sampling_position, logdensity_fn=log_prob, rng_key=key_init
        )
        initial_params = MCLMCAdaptationState(
            L=jnp.sqrt(dim),
            step_size=jnp.sqrt(dim) * 0.25,
            inverse_mass_matrix=initial_imm,
        )

        # desired_energy_var only matters via mclmc_find_L_and_step_size's own
        # kwarg (the step-size dual-averaging target) — build_kernel's use of
        # it is dead unless desired_energy_var_max_ratio is also set, which
        # we don't do, so it's passed only to the calls below.

        # Call 1: phases 1+2 only, always unthinned — raw kernel, since
        # thinning=1 needs no thin_kernel wrapper.
        print()
        with blackjax.progress_bar(label="Phases 1+2 (step size + IMM)"):
            state_12, params_12, warmup_calls_12 = blackjax.mclmc_find_L_and_step_size(
                mclmc_kernel=blackjax.mcmc.mclmc.build_kernel(
                    integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
                ),
                logdensity_fn=log_prob,
                num_steps=round(num_samples * thinning_sampling),
                state=init_state,
                rng_key=key_warmup1,
                diagonal_preconditioning=True,
                frac_tune1=frac_tune1,
                frac_tune2=frac_tune2,
                frac_tune3=0.0,
                desired_energy_var=desired_energy_var,
                params=initial_params,
                l_factor=l_factor,
            )
            # Forces real synchronization before the with-block exits — otherwise
            # JAX's async dispatch lets this return (and the progress bar close,
            # stamped at 100%) long before the tuning scan has actually finished.
            jax.block_until_ready((state_12, params_12))

        imm_12 = np.array(params_12.inverse_mass_matrix)
        step_size_12_finite = bool(np.isfinite(params_12.step_size))
        imm_12_finite = bool(np.all(np.isfinite(imm_12)))
        imm_12_positive = bool(np.all(imm_12 > 0))
        flat_init, _ = jax.flatten_util.ravel_pytree(sampling_position)
        flat_state_12, _ = jax.flatten_util.ravel_pytree(state_12.position)
        max_delta_12 = float(jnp.max(jnp.abs(flat_state_12 - flat_init)))

        print(
            f"[Phases 1+2] Tuned step size: {params_12.step_size:.5f}  "
            f"(finite={step_size_12_finite})"
        )
        print(
            f"[Phases 1+2] Inv. mass matrix: min={imm_12.min():.3e}  "
            f"mean={imm_12.mean():.3e}  max={imm_12.max():.3e}  "
            f"(finite={imm_12_finite}, all_positive={imm_12_positive})"
        )
        print(f"[Phases 1+2] Max |Δ position| from init: {max_delta_12:.3e}")

        # Call 2: phase 3 only, thinned by thinning_warmup, seeded from
        # call 1's state/params so it continues the tuned chain.
        print()
        with blackjax.progress_bar(label="Phase 3 (L via ESS)"):
            tuned_state, tuned_params, warmup_calls_3 = blackjax.mclmc_find_L_and_step_size(
                mclmc_kernel=blackjax.util.thin_kernel(
                    blackjax.mcmc.mclmc.build_kernel(
                        integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
                    ),
                    thinning=thinning_warmup,
                    info_transform=sample_info,
                ),
                logdensity_fn=log_prob,
                num_steps=round(num_samples * thinning_sampling / thinning_warmup),
                state=state_12,
                rng_key=key_warmup2,
                diagonal_preconditioning=True,
                frac_tune1=0.0,
                frac_tune2=0.0,
                frac_tune3=frac_tune3,
                desired_energy_var=desired_energy_var,
                params=params_12,
                l_factor=l_factor * thinning_warmup,
            )
            # See the analogous comment on Call 1 — same reason.
            jax.block_until_ready((tuned_state, tuned_params))

        t1 = time.perf_counter()
        print()

        # Confirms phase 3 didn't reset step_size/IMM (printed, not
        # asserted, so it can't crash a run).
        step_size_preserved = bool(
            np.array(tuned_params.step_size) == np.array(params_12.step_size)
        )
        imm_preserved = bool(
            np.array_equal(
                np.array(tuned_params.inverse_mass_matrix),
                np.array(params_12.inverse_mass_matrix),
            )
        )
        print(
            f"step_size/IMM unchanged by phase 3: "
            f"step_size={step_size_preserved}  inverse_mass_matrix={imm_preserved}"
        )

        warmup_calls = warmup_calls_12 + warmup_calls_3
        warmup_integration_steps = (
            warmup_calls_12 * 1 + warmup_calls_3 * thinning_warmup
        )
        imm = np.array(tuned_params.inverse_mass_matrix)

        L_finite = bool(np.isfinite(tuned_params.L))
        print(f"Warmup time: {timedelta(seconds=int(t1 - t0))}")
        print(f"Tuned L: {tuned_params.L:.4f}  (finite={L_finite})")
        print(f"Tuned step size: {tuned_params.step_size:.5f}")
        print(
            f"Steps per trajectory (L / step_size): "
            f"{tuned_params.L / tuned_params.step_size:.2f}"
        )
        print(
            f"Warmup calls (thinned): {warmup_calls}  "
            f"(phases 1+2: {warmup_calls_12}, phase 3: {warmup_calls_3})  |  "
            f"raw integration steps: {warmup_integration_steps}"
        )
        print(
            f"Inv. mass matrix: min={imm.min():.3e}  mean={imm.mean():.3e}  max={imm.max():.3e}"
        )

        # Bakes the tuned L/step_size/IMM into a fixed SamplingAlgorithm;
        # thin_algorithm (not thin_kernel) wraps that.
        mclmc_sampler = blackjax.mclmc(
            logdensity_fn=log_prob,
            L=tuned_params.L,
            step_size=tuned_params.step_size,
            inverse_mass_matrix=tuned_params.inverse_mass_matrix,
        )
        thinned_sampling_alg = blackjax.util.thin_algorithm(
            mclmc_sampler, thinning=thinning_sampling, info_transform=sample_info
        )

        print()
        with blackjax.progress_bar(label="Sampling"):
            _, (states, infos) = blackjax.util.run_inference_algorithm(
                rng_key=key_sample,
                inference_algorithm=thinned_sampling_alg,
                num_steps=num_samples,
                initial_state=tuned_state,
                transform=lambda state, info: (
                    state.position,
                    MCLMCInfo(
                        logdensity=info.logdensity,
                        energy_change=info.energy_change,
                        kinetic_change=info.kinetic_change,
                        nonans=info.nonans,
                    ),
                ),
            )
            # See the analogous comment on Call 1 — same reason.
            jax.block_until_ready((states, infos))

        t2 = time.perf_counter()
        print()

        print(f"Sampling time:    {timedelta(seconds=int(t2 - t1))}")
        print(f"Total time (w+s): {timedelta(seconds=int(t2 - t0))}")
        print(
            f"Samples saved:    {num_samples}  "
            f"(thinned by {thinning_sampling} for sampling, {thinning_warmup} for warmup phase 3 only)"
        )
        print(
            f"Mean |energy change| (RMS-thinned): {np.array(infos.energy_change).mean():.4e}"
        )
        print(f"Fraction of non-NaN steps: {np.array(infos.nonans).mean():.4f}")

        theta = jax.vmap(self.phi_to_theta)(states.phi)
        states = KarmmaPosition(xlm=states.xlm, theta=theta)

        return states, infos, tuned_params, warmup_calls
