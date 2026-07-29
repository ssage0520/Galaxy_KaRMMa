"""Loads and validates a KaRMMa run configuration from a YAML file."""

import os

import h5py as h5
import healpy as hp
import jax
import numpy as np
import yaml

from karmma.structs import (
    AnalysisConfig,
    IoConfig,
    KarmmaPosition,
    MclmcConfig,
    NutsConfig,
    ThetaParams,
    XlmParams,
)


def _h5_has(path: str, group: str) -> bool:
    """Check whether `group` exists in the HDF5 file at `path`."""
    with h5.File(path, "r") as f:
        return group in f


def _load_xlm(path: str, group: str) -> XlmParams:
    """Load an `XlmParams` from the `group` group of the HDF5 file at `path`."""
    with h5.File(path, "r") as f:
        return XlmParams(real=f[f"{group}/real"][:], imag=f[f"{group}/imag"][:])


def _load_theta(path: str, group: str) -> ThetaParams:
    """Load a `ThetaParams` from the `group` group of the HDF5 file at `path`."""
    with h5.File(path, "r") as f:
        return ThetaParams(
            **{field: f[f"{group}/{field}"][:] for field in ThetaParams._fields}
        )


class KarmmaConfig:
    """Load and validate a KaRMMa run configuration from a YAML file.

    Parameters
    ----------
    config_file : str
        Path to a YAML config file with `mcmc`, `analysis`, and `io`
        sections.

    Attributes
    ----------
    mcmc : NutsConfig or MclmcConfig
        Sampler configuration — which one depends on the config's
        `mcmc.sampler` key ("nuts" or "mclmc", default "mclmc").
    analysis : AnalysisConfig
        Point-transform/survey setup: number of tomographic bins, HEALPix
        resolution, and the `G_N` point-transform parameters/order.
    io : IoConfig
        Input/output configuration: data paths, the observed maps and
        mask loaded from `datafile`, the target power spectrum and pixel
        window, and the resolved initial sampling position (from an init
        file, truth in the mock, or left `None` for random
        initialization).
    """

    def __init__(self, config_file: str) -> None:
        with open(config_file) as f:
            config = yaml.safe_load(f)
        self.mcmc = self._set_mcmc(config["mcmc"])
        self.analysis = self._set_analysis(config["analysis"])
        self.io = self._set_io(config["io"], nside=self.analysis.nside)

    def _set_analysis(self, cfg: dict) -> AnalysisConfig:
        """Build an `AnalysisConfig` from the `analysis` config section."""
        nbins = int(cfg["nbins"])
        nside = int(cfg["nside"])
        lbda, gn_order = self._set_gn(cfg["gn"])
        return AnalysisConfig(nbins=nbins, nside=nside, lbda=lbda, gn_order=gn_order)

    def _set_gn(self, cfg: dict) -> tuple[np.ndarray, int]:
        """Build `(lbda, gn_order)` from the `analysis.gn` config section.

        Parameters
        ----------
        cfg : dict
            The `analysis.gn` section — either `{alpha, beta}` (G2) or
            `{a, b, c}` (G3), each a comma-separated per-bin string.

        Returns
        -------
        lbda : np.ndarray
            Shape (gn_order, Nbins), rows in the order `ForwardModel.gn`
            expects: (alpha, beta) for gn_order=2, (a, b, c) for gn_order=3.
        gn_order : int
            2 or 3, inferred from which keys are present.

        Raises
        ------
        ValueError
            If `cfg` doesn't contain exactly one of the two supported key sets.
        """
        is_g2 = {"alpha", "beta"} <= cfg.keys()
        is_g3 = {"a", "b", "c"} <= cfg.keys()
        if is_g2 and not is_g3:
            keys, gn_order = ("alpha", "beta"), 2
        elif is_g3 and not is_g2:
            keys, gn_order = ("a", "b", "c"), 3
        else:
            raise ValueError(
                "analysis.gn must contain exactly one of {alpha, beta} (G2) or "
                "{a, b, c} (G3)."
            )
        lbda = np.array([np.asarray(cfg[k].split(","), dtype=float) for k in keys])
        return lbda, gn_order

    def _set_io(self, cfg: dict, nside: int) -> IoConfig:
        """Build an `IoConfig` from the `io` config section.

        Loads the observed maps, target power spectrum, and pixel window
        (`datafile`/`cl_file`/`pixwin`, all resolved relative to
        `input_dir`), then resolves `xlm`/`theta` initial values by
        priority order (see Notes).

        Parameters
        ----------
        cfg : dict
            The `io` config section.
        nside : int
            HEALPix resolution, from `AnalysisConfig.nside` — needed only
            if `pixwin: healpix` requests an analytically-computed pixel
            window.

        Returns
        -------
        IoConfig
            Resolved input/output configuration.

        Raises
        ------
        ValueError
            If `init_file` is provided but missing a `theta` group while
            `infer_theta=True`, or if no theta source resolves at all.

        Notes
        -----
        `xlm` priority order: `init_file`'s `xlm` group, then `datafile`'s
        `true_xlm` group, then `None` (deferred to a random draw via
        `sampler.make_random_xlm()` in `run_karmma.py`).

        `theta` priority order: `init_file`'s `theta` group, then
        `datafile`'s `true_theta` group, then `theta_file`'s `theta`
        group, then a `ValueError` if none resolve. Unlike `xlm`, `theta`
        has no random-init fallback and must always resolve to a concrete
        value here — even when `infer_theta=False`, the resolved value is
        still needed as `theta_fixed` for `log_prob`.

        The `init_file`-specific validation (checked before the `theta`
        priority chain runs) exists to catch a specific inconsistent
        case: an `init_file` that has `xlm` but is missing `theta` while
        `infer_theta=True`, so the caller doesn't get silently bounced to
        `true_theta`/`theta_file` instead of continuing from their
        intended init file.

        TODO: `theta_fixed`/`infer_theta=False` is no longer actually
        supported by the sampler backends — `WhitenedSampler`'s whitening
        machinery assumes `theta` is part of the sampled position, so a
        fixed theta breaks it. This whole section needs reworking, likely
        removing `infer_theta` as an option entirely.
        """
        input_dir = cfg["input_dir"]
        output_dir = cfg["output_dir"]

        def _resolve(key):
            value = cfg.get(key)
            return os.path.join(input_dir, value) if value else None

        datafile = _resolve("datafile")
        init_file = _resolve("init_file")
        theta_file = _resolve("theta_file")

        with h5.File(datafile, "r") as f:
            dg_obs = f["dg_obs"][:]
            mask = f["mask"][:].astype(bool)
            N_bar = f["N_bar"][:]

        cl = np.load(_resolve("cl_file"))

        # 3 options for pixwin: null, healpix, or a filename (resolved via input_dir)
        pixwin_cfg = cfg.get("pixwin")
        if pixwin_cfg == "healpix":
            pixwin = hp.sphtfunc.pixwin(nside, lmax=3 * nside - 1)
            print("Pixel window: healpix")
        elif pixwin_cfg is not None:
            pixwin = np.load(_resolve("pixwin"))
            print(f"Pixel window: empirical ({pixwin_cfg})")
        else:
            pixwin = None
            print("Pixel window: none (warning: this may bias your results)")

        # --- xlm (priority order) ---
        # `init_file and ...` short-circuits safely when init_file is None
        if init_file and _h5_has(init_file, "xlm"):
            xlm = _load_xlm(init_file, "xlm")
            print(f"xlm init: {init_file}")
        elif _h5_has(datafile, "true_xlm"):
            xlm = _load_xlm(datafile, "true_xlm")
            print("xlm init: truth from datafile")
        else:
            xlm = None  # signals run_karmma.py to call sampler.make_random_xlm()
            print("xlm init: random (deferred to sampler)")

        # --- theta (priority order) ---
        # validate init_file completeness before falling through
        if init_file and self.mcmc.infer_theta and not _h5_has(init_file, "theta"):
            raise ValueError(
                "init_file provided but missing 'theta' group; required when infer_theta=True."
            )
        if init_file and _h5_has(init_file, "theta"):
            theta = _load_theta(init_file, "theta")
            print(f"theta init: {init_file}")
        elif _h5_has(datafile, "true_theta"):
            theta = _load_theta(datafile, "true_theta")
            print("theta init: truth from datafile")
        elif theta_file:
            # theta_file is an HDF5 file with a 'theta/' group
            theta = _load_theta(theta_file, "theta")
            print(f"theta init: {theta_file}")
        else:
            raise ValueError(
                "No theta source found. Provide init_file with a 'theta/' group, "
                "a theta_file (HDF5 with 'theta/' group), or ensure datafile contains 'true_theta/'."
            )

        # --- assemble ---
        if self.mcmc.infer_theta:
            initial_position = KarmmaPosition(xlm=xlm, theta=theta)
            theta_fixed = None
        else:
            initial_position = KarmmaPosition(xlm=xlm)
            theta_fixed = theta

        return IoConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            datafile=datafile,
            dg_obs=dg_obs,
            mask=mask,
            N_bar=N_bar,
            cl=cl,
            pixwin=pixwin,
            initial_position=initial_position,
            theta_fixed=theta_fixed,
        )

    def _resolve_seed_and_key(self, cfg: dict) -> tuple[int, jax.Array]:
        """Resolve the mcmc config's seed (or generate one) into `(seed, PRNGKey(seed))`."""
        seed = cfg.get("seed")
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**31))
            print(f"No seed provided — using randomly generated seed: {seed}")
        else:
            seed = int(seed)
        return seed, jax.random.PRNGKey(seed)

    def _get_or_default(
        self,
        cfg: dict,
        key: str,
        default: float | int,
        cast: type[float] | type[int] = float,
    ) -> float | int:
        """Get `cfg[key]`, falling back to `default` when absent or explicitly null."""
        # `cfg.get(key, default)` only falls back when `key` is absent, not when
        # it's present with an explicit YAML `null` (e.g. config/nuts.yaml's
        # `target_acceptance_rate: null`) — this treats both cases the same.
        value = cfg.get(key)
        return default if value is None else cast(value)

    def _set_mcmc(self, cfg: dict) -> NutsConfig | MclmcConfig:
        """Dispatch to `_set_nuts` or `_set_mclmc` per the config's `sampler` key.

        Raises
        ------
        ValueError
            If `sampler` isn't "nuts" or "mclmc".
        """
        # Named `sampler_backend`, not `sampler` — `sampler` is reserved elsewhere
        # (e.g. run_karmma.py's dispatch) for the actual constructed sampler
        # *object*; this is just the dispatch string read from the config.
        sampler_backend = cfg.get("sampler", "mclmc")
        if sampler_backend == "nuts":
            return self._set_nuts(cfg)
        if sampler_backend == "mclmc":
            return self._set_mclmc(cfg)
        raise ValueError(
            f"Unknown mcmc.sampler {sampler_backend!r}; expected 'nuts' or 'mclmc'."
        )

    def _set_nuts(self, cfg: dict) -> NutsConfig:
        """Build a `NutsConfig` from the `mcmc` config section."""
        n_samples = int(cfg["n_samples"])
        seed, key = self._resolve_seed_and_key(cfg)

        num_warmup = int(cfg["num_warmup"])
        step_size = self._get_or_default(cfg, "step_size", 0.05)
        target_acceptance_rate = self._get_or_default(cfg, "target_acceptance_rate", 0.65)
        imm_shrinkage_to_previous = self._get_or_default(cfg, "imm_shrinkage_to_previous", 0.0)

        infer_theta = bool(cfg.get("infer_theta", False))

        return NutsConfig(
            n_samples=n_samples,
            key=key,
            seed=seed,
            num_warmup=num_warmup,
            step_size=step_size,
            target_acceptance_rate=target_acceptance_rate,
            imm_shrinkage_to_previous=imm_shrinkage_to_previous,
            infer_theta=infer_theta,
        )

    def _set_mclmc(self, cfg: dict) -> MclmcConfig:
        """Build an `MclmcConfig` from the `mcmc` config section."""
        n_samples = int(cfg["n_samples"])
        seed, key = self._resolve_seed_and_key(cfg)

        frac_tune1 = self._get_or_default(cfg, "frac_tune1", 0.1)
        # 0.3, not blackjax's stock 0.1 — validated in dev_notebooks/mclmc.ipynb
        # to give diagonal preconditioning enough samples to converge.
        frac_tune2 = self._get_or_default(cfg, "frac_tune2", 0.3)
        frac_tune3 = self._get_or_default(cfg, "frac_tune3", 0.1)
        l_factor = self._get_or_default(cfg, "l_factor", 0.4)
        thinning_warmup = self._get_or_default(cfg, "thinning_warmup", 5, cast=int)
        thinning_sampling = self._get_or_default(cfg, "thinning_sampling", 5, cast=int)
        desired_energy_var = self._get_or_default(cfg, "desired_energy_var", 5e-4)

        infer_theta = bool(cfg.get("infer_theta", False))

        return MclmcConfig(
            n_samples=n_samples,
            key=key,
            seed=seed,
            frac_tune1=frac_tune1,
            frac_tune2=frac_tune2,
            frac_tune3=frac_tune3,
            l_factor=l_factor,
            thinning_warmup=thinning_warmup,
            thinning_sampling=thinning_sampling,
            desired_energy_var=desired_energy_var,
            infer_theta=infer_theta,
        )
