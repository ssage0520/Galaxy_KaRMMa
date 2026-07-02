import h5py as h5
import healpy as hp
import jax
import numpy as np
import yaml

from karmma.structs import (
    AnalysisConfig,
    IoConfig,
    KarmmaPosition,
    McmcConfig,
    ThetaParams,
    XlmParams,
)


def _h5_has(path, group):
    with h5.File(path, "r") as f:
        return group in f


def _load_xlm(path, group):
    with h5.File(path, "r") as f:
        return XlmParams(real=f[f"{group}/real"][:], imag=f[f"{group}/imag"][:])


def _load_theta(path, group):
    with h5.File(path, "r") as f:
        return ThetaParams(
            **{field: f[f"{group}/{field}"][:] for field in ThetaParams._fields}
        )


class KarmmaConfig:
    def __init__(self, config_file):
        with open(config_file) as f:
            config = yaml.safe_load(f)
        self.mcmc = self._set_mcmc(config["mcmc"])
        self.analysis = self._set_analysis(config["analysis"])
        self.io = self._set_io(config["io"])

    def _set_analysis(self, cfg):
        nbins = int(cfg["nbins"])
        nside = int(cfg["nside"])
        alpha = np.asarray(cfg["alpha"].split(","), dtype=float)
        beta = np.asarray(cfg["beta"].split(","), dtype=float)
        cl = np.load(cfg["cl_file"])

        # 3 options for pixwin: null, healpix, or a path to a .npy file
        pixwin_cfg = cfg.get("pixwin")
        if pixwin_cfg == "healpix":
            pixwin = hp.sphtfunc.pixwin(nside, lmax=3 * nside - 1)
            print("Pixel window: healpix")
        elif pixwin_cfg is not None:
            pixwin = np.load(pixwin_cfg)
            print(f"Pixel window: empirical ({pixwin_cfg})")
        else:
            pixwin = None
            print("Pixel window: none (warning: this may bias your results)")

        return AnalysisConfig(
            nbins=nbins, nside=nside, alpha=alpha, beta=beta, cl=cl, pixwin=pixwin
        )

    def _set_io(self, cfg):
        datafile = cfg["datafile"]
        io_dir = cfg["io_dir"]
        init_file = cfg.get("init_file")  # None is the common case
        theta_file = cfg.get("theta_file")

        with h5.File(datafile, "r") as f:
            dg_obs = f["dg_obs"][:]
            mask = f["mask"][:].astype(bool)
            N_bar = f["N_bar"][:]

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
            datafile=datafile,
            io_dir=io_dir,
            dg_obs=dg_obs,
            mask=mask,
            N_bar=N_bar,
            initial_position=initial_position,
            theta_fixed=theta_fixed,
        )

    def _set_mcmc(self, cfg):
        n_warmup = int(cfg["n_warmup"])
        n_samples = int(cfg["n_samples"])

        seed = cfg.get("seed")
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**31))
            print(f"No seed provided — using randomly generated seed: {seed}")
        else:
            seed = int(seed)
        key = jax.random.PRNGKey(seed)

        step_size = cfg.get("init_step_size")
        if step_size is None:
            step_size = 0.05
            print("No step size provided — using default: 0.05")
        else:
            step_size = float(step_size)

        target_acceptance = cfg.get("target_acceptance_rate")
        if target_acceptance is None:
            target_acceptance = 0.65
            print("No target acceptance rate provided — using default: 0.65")
        else:
            target_acceptance = float(target_acceptance)

        infer_theta = bool(cfg.get("infer_theta", False))

        return McmcConfig(
            n_warmup=n_warmup,
            n_samples=n_samples,
            key=key,
            seed=seed,
            step_size=step_size,
            target_acceptance=target_acceptance,
            infer_theta=infer_theta,
        )
