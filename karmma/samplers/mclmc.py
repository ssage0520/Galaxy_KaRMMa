"""MCLMC sampler backend, built on WhitenedSampler's shared whitening/preconditioning."""

import time
from datetime import timedelta

import blackjax
import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState
from blackjax.mcmc.mclmc import MCLMCInfo as RawMCLMCInfo

from karmma.samplers.base import WhitenedSampler
from karmma.structs import KarmmaPosition, MCLMCInfo


class MCLMCSampler(WhitenedSampler):
    """Samples from a whitened KarmmaPosition using MCLMC, via blackjax's L/step-size tuning."""

    def sample(
        self,
        key: jax.Array,
        num_samples: int,
        initial_position: KarmmaPosition,
        initial_imm: np.ndarray,
        frac_tune1: float = 0.1,
        frac_tune2: float = 0.3,
        frac_tune3: float = 0.1,
        l_factor: float = 0.4,
        desired_energy_var: float = 5e-4,
        thinning_warmup: int = 5,
        thinning_sampling: int = 5,
    ) -> tuple[KarmmaPosition, MCLMCInfo, MCLMCAdaptationState]:
        """Run MCLMC, seeding its diagonal preconditioner from `initial_imm`.

        Parameters
        ----------
        key : jax.Array
            JAX PRNG key, split internally for initialization, warmup, and sampling.
        num_samples : int
            Number of samples actually saved (post-thinning), not a raw
            integrator-step budget.
        initial_position : KarmmaPosition
            Physical (theta-space) starting position; whitened internally as
            the first step (see `WhitenedSampler._prepare_sampling`).
        initial_imm : np.ndarray
            Initial diagonal inverse mass matrix, in BlackJax pytree-flat
            layout for the whitened position — typically `np.ones(N_full)`,
            since phi is already whitened to ~unit variance.
        frac_tune1 : float, optional
            Fraction of warmup spent on phase 1 (step-size dual averaging),
            by default 0.1.
        frac_tune2 : float, optional
            Fraction of warmup spent on phase 2 (diagonal preconditioning),
            by default 0.3.
        frac_tune3 : float, optional
            Fraction of warmup spent on phase 3 (tuning `L` via effective
            sample size), by default 0.1.
        l_factor : float, optional
            Factor scaling the estimated autocorrelation length to obtain
            the momentum decoherence length `L`, by default 0.4.
        desired_energy_var : float, optional
            Target per-step energy-change variance for step-size dual
            averaging, by default 5e-4.
        thinning_warmup : int, optional
            Thinning applied to phase 3 only (phases 1+2 always run
            unthinned), by default 5.
        thinning_sampling : int, optional
            Thinning applied during the final sampling phase, by default 5.

        Returns
        -------
        states : KarmmaPosition
            Posterior samples (physical theta-space), batched over `num_samples`.
        infos : MCLMCInfo
            Per-sample MCLMC diagnostics (log density, energy change,
            non-NaN fraction), aggregated over each thinning block (see
            `sample_info`).
        tuned_params : MCLMCAdaptationState
            Tuned `L`, `step_size`, and `inverse_mass_matrix` from warmup.
        """
        sampling_position, log_prob = self._prepare_sampling(initial_position)
        dim = blackjax.util.pytree_size(sampling_position)

        t0 = time.perf_counter()

        key, key_init, key_warmup1, key_warmup2, key_sample = jax.random.split(key, 5)

        def sample_info(info: RawMCLMCInfo) -> RawMCLMCInfo:
            """Aggregate raw per-step MCLMC info over one thinning block.

            Parameters
            ----------
            info : RawMCLMCInfo
                Raw per-step MCLMC info for every step in the block.

            Returns
            -------
            RawMCLMCInfo
                Block-aggregated info, with one value per field instead of
                one per raw step.

            Notes
            -----
            Each field uses a different reduction, since a single "RMS
            everything" or "mean everything" rule would silently corrupt
            the results:

            - `logdensity` takes the last raw step's value (matching the
              block's final, saved position) rather than an aggregate.
            - `energy_change`/`kinetic_change` are genuinely mean-zero
              step-error diagnostics, so an RMS magnitude is meaningful
              over the block.
            - `nonans` is a 0/1 indicator, so it needs a mean (fraction
              clean), not RMS — RMS of a 0/1 array is `sqrt(fraction)`,
              not the fraction itself.
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
        # L/step_size match blackjax's own params=None default (see
        # mclmc_adaptation.py) — done explicitly here only to seed
        # inverse_mass_matrix with initial_imm instead of ones(dim).
        initial_params = MCLMCAdaptationState(
            L=jnp.sqrt(dim),
            step_size=jnp.sqrt(dim) * 0.25,
            inverse_mass_matrix=initial_imm,
        )

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
            # stamped at 100%) long before the warmup has actually finished.
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
                # blackjax's internal L estimate is step_size * (num_steps / ess),
                # counted in thin_kernel-wrapped (thinned) calls — but each such
                # call actually advances the chain by thinning_warmup raw steps,
                # so the raw-step autocorrelation length is thinning_warmup times
                # longer than that count. Scaling l_factor up compensates.
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
        warmup_integration_steps = warmup_calls_12 + warmup_calls_3 * thinning_warmup
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

        states = self._unwhiten(states)

        return states, infos, tuned_params
