"""Loads and validates a KaRMMa run configuration from a YAML file."""

import os

import h5py as h5
import healpy as hp
import jax
import jax.numpy as jnp
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


def _load_whitening(path: str) -> tuple[np.ndarray, np.ndarray, ThetaParams]:
    """Load `(V, w, theta0)` from a `theta_reparam` group, as written to `mcmc_metadata.h5`."""
    if not _h5_has(path, "theta_reparam"):
        raise ValueError(f"io.whitening {path} has no 'theta_reparam' group.")
    with h5.File(path, "r") as f:
        grp = f["theta_reparam"]
        return (
            grp["V"][:],
            grp["w"][:],
            ThetaParams(
                **{field: grp[f"theta0/{field}"][:] for field in ThetaParams._fields}
            ),
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
        Input/output configuration: data paths, the observed maps and mask
        loaded from `datafile`, the target power spectrum and pixel window,
        and the initial sampling position.
    """

    def __init__(self, config_file: str) -> None:
        with open(config_file) as f:
            config = yaml.safe_load(f)
        self.mcmc = self._set_mcmc(config["mcmc"])
        self.analysis = self._set_analysis(config["analysis"])
        self.io = self._set_io(
            config["io"], nside=self.analysis.nside, nbins=self.analysis.nbins
        )

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
            Shape (gn_order, Nbins), rows in the order `ForwardModel.gn_inv`
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

    def _set_io(self, cfg: dict, nside: int, nbins: int) -> IoConfig:
        """Build an `IoConfig` from the `io` config section.

        Loads the observed maps, target power spectrum, and pixel window
        (`datafile`/`cl_file`/`pixwin`, all resolved relative to `input_dir`),
        and resolves the initial position and whitening.

        Parameters
        ----------
        cfg : dict
            The `io` config section.
        nside : int
            HEALPix resolution, from `AnalysisConfig.nside` — needed only
            if `pixwin: healpix` requests an analytically-computed pixel
            window.
        nbins : int
            Number of tomographic bins, from `AnalysisConfig.nbins` — used to
            broadcast scalar `theta_guess` entries.

        Returns
        -------
        IoConfig
            Resolved input/output configuration.

        Raises
        ------
        ValueError
            On unrecognized `io` keys, a negative `init_passes`, an
            `init_position` file with neither group, a derived `theta` with no
            `theta_guess`, or a supplied `whitening` alongside an incomplete
            position or nonzero `init_passes`.

        Notes
        -----
        `init_position` is `auto` or a path to an HDF5 file with an `xlm`
        group, a `theta` group, or both; `run_karmma.py` derives whatever is
        missing. `whitening` is `auto` or a path to a file with a
        `theta_reparam` group.

        `save_maps` (default `True`) controls whether `xlm` is retained
        during sampling and saved to `samples.h5`; `theta` is always saved.
        """
        # Reject unknown keys: silently ignoring a retired knob like `xlm_init`
        # would change what a run does without saying so.
        known = {
            "input_dir",
            "output_dir",
            "datafile",
            "cl_file",
            "pixwin",
            "save_maps",
            "init_position",
            "init_passes",
            "theta_guess",
            "whitening",
        }
        unknown = set(cfg) - known
        if unknown:
            raise ValueError(
                f"Unrecognized io keys: {sorted(unknown)}. Valid keys are {sorted(known)}."
            )

        input_dir = cfg["input_dir"]
        output_dir = cfg["output_dir"]
        save_maps = bool(cfg.get("save_maps", True))
        print(f"save_maps: {save_maps}")

        def _resolve(key: str) -> str | None:
            """Join `cfg[key]` onto `input_dir`, or return `None` if unset/empty."""
            value = cfg.get(key)
            return os.path.join(input_dir, value) if value else None

        datafile = _resolve("datafile")

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

        # --- initial position ---
        init_passes = int(cfg.get("init_passes", 0))
        if init_passes < 0:
            raise ValueError(f"io.init_passes must be >= 0, got {init_passes}.")

        init_position = cfg.get("init_position", "auto")
        if init_position == "auto":
            xlm, theta = None, None
            print("init position: auto (theta-free field, deferred until the model exists)")
        else:
            path = _resolve("init_position")
            has_xlm, has_theta = _h5_has(path, "xlm"), _h5_has(path, "theta")
            if not (has_xlm or has_theta):
                raise ValueError(
                    f"io.init_position {path} has neither an 'xlm' nor a 'theta' group. "
                    "Use init_position: auto to build both from the data."
                )
            xlm = _load_xlm(path, "xlm") if has_xlm else None
            theta = _load_theta(path, "theta") if has_theta else None
            supplied = " + ".join(
                n for n, present in (("xlm", has_xlm), ("theta", has_theta)) if present
            )
            print(f"init position: {path} ({supplied})")

        theta_guess = self._set_theta_guess(cfg, nbins)
        if theta is None and theta_guess is None:
            raise ValueError(
                "theta must be derived (io.init_position supplies none), which needs "
                "io.theta_guess as the starting point for refine_theta."
            )

        # --- whitening ---
        whitening_cfg = cfg.get("whitening", "auto")
        if whitening_cfg == "auto":
            whitening = None
        else:
            # A supplied basis is only valid at the position it was built for.
            if xlm is None or theta is None:
                raise ValueError(
                    "io.whitening requires io.init_position to supply both 'xlm' and "
                    "'theta'; a derived position would not match the supplied basis."
                )
            if init_passes:
                raise ValueError(
                    f"io.whitening is incompatible with io.init_passes={init_passes}: "
                    "the extra passes move the position away from the supplied basis."
                )
            whitening = _load_whitening(_resolve("whitening"))
            print(f"whitening: {_resolve('whitening')}")

        return IoConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            datafile=datafile,
            dg_obs=dg_obs,
            mask=mask,
            N_bar=N_bar,
            cl=cl,
            pixwin=pixwin,
            initial_position=KarmmaPosition(xlm=xlm, theta=theta),
            save_maps=save_maps,
            init_passes=init_passes,
            theta_guess=theta_guess,
            whitening=whitening,
        )

    @staticmethod
    def _set_theta_guess(cfg: dict, nbins: int) -> ThetaParams | None:
        """Parse `io.theta_guess`: each field a scalar or `nbins` comma-separated values."""
        raw = cfg.get("theta_guess")
        if raw is None:
            return None
        missing = set(ThetaParams._fields) - set(raw)
        if missing:
            raise ValueError(f"io.theta_guess is missing fields: {sorted(missing)}.")
        values = {}
        for field in ThetaParams._fields:
            arr = np.asarray(str(raw[field]).split(","), dtype=float)
            if arr.size == 1:
                arr = np.full(nbins, arr[0])
            if arr.size != nbins:
                raise ValueError(
                    f"io.theta_guess.{field} has {arr.size} values, expected 1 or {nbins}."
                )
            values[field] = jnp.asarray(arr)
        return ThetaParams(**values)

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

        return NutsConfig(
            n_samples=n_samples,
            key=key,
            seed=seed,
            num_warmup=num_warmup,
            step_size=step_size,
            target_acceptance_rate=target_acceptance_rate,
            imm_shrinkage_to_previous=imm_shrinkage_to_previous,
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
        )
