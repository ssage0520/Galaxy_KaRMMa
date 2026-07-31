"""NUTS sampler backend, built on WhitenedSampler's shared whitening/preconditioning."""

import time
from datetime import timedelta

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
from blackjax.adaptation.base import AdaptationInfo, get_filter_adapt_info_fn

from karmma.samplers.base import WhitenedSampler
from karmma.structs import KarmmaPosition, NUTSInfo, WhitenedKarmmaPosition


class NUTSSampler(WhitenedSampler):
    """Samples from a whitened KarmmaPosition using NUTS, via blackjax's window adaptation."""

    def sample(
        self,
        key: jax.Array,
        num_warmup: int,
        num_samples: int,
        initial_position: KarmmaPosition,
        initial_imm: np.ndarray,
        imm_shrinkage_to_previous: float = 0.0,
        step_size: float = 0.05,
        target_acceptance_rate: float = 0.65,
        save_xlm: bool = True,
    ) -> tuple[KarmmaPosition, NUTSInfo, dict, AdaptationInfo]:
        """Run NUTS, seeding window adaptation's inverse mass matrix from `initial_imm`.

        Parameters
        ----------
        key : jax.Array
            JAX PRNG key, split internally for warmup and sampling.
        num_warmup : int
            Number of window-adaptation warmup steps.
        num_samples : int
            Number of post-warmup samples to draw.
        initial_position : KarmmaPosition
            Physical (theta-space) starting position; whitened internally as
            the first step (see `WhitenedSampler._prepare_sampling`).
        initial_imm : np.ndarray
            Initial diagonal inverse mass matrix, in BlackJax pytree-flat
            layout for the whitened position.
        imm_shrinkage_to_previous : float, optional
            Pseudo-count controlling shrinkage of each warmup window's
            adapted inverse mass matrix toward the previous window's,
            by default 0.0 (no persistence, matches Stan's behavior).
        step_size : float, optional
            Initial step size for warmup, by default 0.05.
        target_acceptance_rate : float, optional
            Target acceptance rate for step-size adaptation, by default 0.65.
        save_xlm : bool, optional
            Whether to retain the sampled `xlm` trajectory, by default True.
            When False, `xlm` is never stacked across the sampling loop —
            only the current step's `xlm` exists in memory, as part of the
            scan carry — and `states.xlm` is `None` on return.

        Returns
        -------
        states : KarmmaPosition
            Posterior samples (physical theta-space), batched over
            `num_samples`. `states.xlm` is `None` when `save_xlm=False`.
        infos : NUTSInfo
            Per-sample NUTS diagnostics (divergences, integration steps,
            acceptance rate, energy, log density), batched over `num_samples`.
        tuned_params : dict
            Tuned `step_size`/`inverse_mass_matrix` from window adaptation.
        winfo : blackjax.adaptation.base.AdaptationInfo
            Full warmup-adaptation info; `.info` holds the filtered
            per-window warmup diagnostics used for the printed summary.
        """
        sampling_position, log_prob = self._prepare_sampling(initial_position)

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
            # stamped at 100%) long before the warmup has actually finished.
            jax.block_until_ready((wstate, tuned_params))

        t1 = time.perf_counter()
        print()

        # TODO: this whole estimated-sampling-time block is a pre-progress-bar
        # leftover — now that blackjax.progress_bar shows live timing during
        # sampling itself, this upfront estimate is largely redundant. Revisit
        # when NUTS's output/printing is reworked more generally.
        warmup_steps = np.array(winfo.info.num_integration_steps)
        time_per_leapfrog = (t1 - t0) / warmup_steps.sum()
        # Last 20 steps approximate steady-state (post-adaptation) integration-step
        # count, for a rough sampling-time estimate.
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
                    state.position
                    if save_xlm
                    else WhitenedKarmmaPosition(xlm=None, phi=state.position.phi),
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

        states = self._unwhiten(states)

        return states, infos, tuned_params, winfo
