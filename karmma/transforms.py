"""SHT transforms for KaRMMa.

ducc0-based, JIT-compatible via jaxbind. All bins are passed as a single
ntrans call. nside/lmax/spin must be Python literals (not traced JAX
values). Spin-2 is supported via the spin argument. Supports jax.grad,
jax.hessian, and jax.jacrev(jax.jacrev(...)).

`jaxbind.get_linear_call` registers a true JAX primitive with its own
jvp/transpose rules, so JAX derives every derivative directly from the
ducc0 synthesis/adjoint_synthesis pair — no hand-written adjoint.

The SHT is real-linear but not complex-linear (complex alm in, real map
out), and JAX's transpose rule for a linear primitive assumes
complex-linearity. So the primitive operates on the interleaved real
view of the complex alm array,

    v = [Re a_0, Im a_0, Re a_1, Im a_1, ...],  shape (..., 2 * n_alm)

which is bit-identical to the complex array's memory layout — the
callbacks reinterpret it with a zero-copy `ndarray.view`, and ducc0
reads/writes the FFI buffers directly. In this representation the
transpose of `synthesis` is `adjoint_synthesis` with the m>0
coefficients doubled (each stored m>0 coefficient contributes to both +m
and -m in the real sum), verified to machine precision by a dot-product
test. Everything else — the complex/real repacking, the pixel weights,
the 4π/n_pix normalization, and the compensating 1/2 on m>0 in
`map2alm` — lives in JAX-space and is differentiated natively.
"""

import jax

jax.config.update("jax_enable_x64", True)
import time
import urllib.request
from functools import cache, lru_cache
from pathlib import Path

import astropy.io.fits as fits
import ducc0.healpix
import ducc0.sht
import healpy as hp
import jax.numpy as jnp
import numpy as np
from jaxbind import get_linear_call


@lru_cache(maxsize=1)
def get_pixel_weights(nside: int) -> np.ndarray:
    """Return the HEALPix full pixel weight map for nside, downloaded on first use.

    Equivalent to healpy's use_pixel_weights=True. Results are cached to
    ~/.cache/karmma/full_weights/.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter.

    Returns
    -------
    np.ndarray
        Pixel weight map, shape (n_pix,), with values near 1.0 (stored
        as w + 1 per the FITS convention).
    """
    nside = int(nside)
    nside_str = f"{nside:04d}"
    filename = f"healpix_full_weights_nside_{nside_str}.fits"
    cache_dir = Path.home() / ".cache" / "karmma" / "full_weights"
    path = cache_dir / filename

    if not path.exists():
        url = (
            "https://raw.githubusercontent.com/healpy/healpy-data"
            f"/master/full_weights/{filename}"
        )
        print(f"Downloading pixel weights for nside={nside} from healpy-data...")
        cache_dir.mkdir(parents=True, exist_ok=True)
        for attempt in range(3):
            try:
                urllib.request.urlretrieve(url, path)
                break
            except Exception as e:
                if attempt < 2:
                    print(f"  Attempt {attempt + 1} failed ({e}), retrying...")
                    time.sleep(2)
                else:
                    path.unlink(missing_ok=True)
                    raise
        print("Download complete.")

    with fits.open(path) as hdul:
        w8list = hdul[1].data.field(0).astype(np.float64)

    npix = hp.nside2npix(nside)
    w8map = np.zeros(npix, dtype=np.float64)

    pnorth = vpix = 0
    for ring in range(2 * nside):
        qpix = min(ring + 1, nside)
        shifted = int(ring < nside - 1 or (ring + nside) % 2 == 1)
        qp4 = 4 * qpix

        for p in range(qp4):
            j4 = p % qpix
            rpix = min(j4, qpix - shifted - j4)
            w8map[pnorth + p] = w8list[vpix + rpix]

        if ring < 2 * nside - 1:
            psouth = npix - pnorth - qp4
            w8map[psouth : psouth + qp4] = w8map[pnorth : pnorth + qp4]

        pnorth += qp4
        vpix += (qpix + 1) // 2 + 1 - ((qpix % 2) | shifted)

    return w8map + 1.0


@cache
def _sht_ops(nside: int, lmax: int, spin: int, nthreads: int) -> tuple:
    """Build the forward/adjoint SHT primitive pair for one configuration.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter.
    lmax : int
        Maximum multipole; also used as mmax.
    spin : int
        Spin of the transform (0 or 2).
    nthreads : int
        ducc0 thread count; 0 means all hardware threads.

    Returns
    -------
    tuple
        `(synthesis_op, adjoint_synthesis_op)`. Each is a JAX primitive
        taking one array and returning a 1-tuple of arrays;
        `synthesis_op` maps real-packed alm (..., 2 * n_alm) to a map
        (..., n_pix), and `adjoint_synthesis_op` maps back. Each is
        registered as the other's transpose.

    Notes
    -----
    Never evicted (`@cache`, unbounded) on purpose: jaxbind identifies
    the callbacks by `id()`, baked into the compiled executable, so a
    collected closure whose address got reused would silently corrupt
    results.

    The configuration is captured by closure rather than passed through
    jaxbind's `kwargs` channel, which avoids a `pickle.loads` on every
    single call.
    """
    geo = ducc0.healpix.Healpix_Base(nside, "RING").sht_info()
    geo = {k: geo[k] for k in ("theta", "phi0", "nphi", "ringstart")}
    n_pix = hp.nside2npix(nside)
    n_real = 2 * hp.Alm.getsize(lmax)  # real degrees of freedom per component
    m0_end = 2 * (lmax + 1)  # end of the m=0 block, in real DOFs

    def _synthesis(
        out: tuple[np.ndarray, ...],
        args: tuple[np.ndarray, ...],
        kwargs_dump: np.ndarray,
    ) -> None:
        (alm_real,) = args
        ducc0.sht.synthesis(
            alm=alm_real.view(np.complex128),
            map=out[0],
            **geo,
            lmax=lmax,
            mmax=lmax,
            spin=spin,
            nthreads=nthreads,
        )

    def _adjoint_synthesis(
        out: tuple[np.ndarray, ...],
        args: tuple[np.ndarray, ...],
        kwargs_dump: np.ndarray,
    ) -> None:
        (maps,) = args
        ducc0.sht.adjoint_synthesis(
            map=maps,
            alm=out[0].view(np.complex128),
            **geo,
            lmax=lmax,
            mmax=lmax,
            spin=spin,
            nthreads=nthreads,
        )
        out[0][..., m0_end:] *= 2.0

    def _synthesis_abstract(
        *args: np.ndarray, **kwargs: object
    ) -> tuple[tuple[tuple[int, ...], np.dtype], ...]:
        (alm_real,) = args
        return ((alm_real.shape[:-1] + (n_pix,), alm_real.dtype),)

    def _adjoint_synthesis_abstract(
        *args: np.ndarray, **kwargs: object
    ) -> tuple[tuple[tuple[int, ...], np.dtype], ...]:
        (maps,) = args
        return ((maps.shape[:-1] + (n_real,), maps.dtype),)

    fwd = get_linear_call(
        _synthesis,
        _adjoint_synthesis,
        _synthesis_abstract,
        _adjoint_synthesis_abstract,
    )
    adj = get_linear_call(
        _adjoint_synthesis,
        _synthesis,
        _adjoint_synthesis_abstract,
        _synthesis_abstract,
    )
    return fwd, adj


@lru_cache(maxsize=4)
def _map2alm_scale(lmax: int, n_pix: int) -> np.ndarray:
    """Per-real-DOF rescaling that turns the adjoint's output into `map2alm`'s.

    Parameters
    ----------
    lmax : int
        Maximum multipole.
    n_pix : int
        Number of map pixels, for the 4π/n_pix normalization.

    Returns
    -------
    np.ndarray
        Shape (2 * n_alm,), interleaved to match the real-packed alm
        layout. Undoes the primitive's 2x on m>0 and applies the
        4π/n_pix normalization, so the result matches
        `healpy.map2alm(..., use_pixel_weights=True)`.
    """
    ms = hp.Alm.getlm(lmax)[1]
    return np.repeat(np.where(ms == 0, 1.0, 0.5) * (4.0 * np.pi / n_pix), 2)


def _to_real(alms: jnp.ndarray) -> jnp.ndarray:
    """Interleave complex alm (..., n_alm) into real DOFs (..., 2 * n_alm)."""
    stacked = jnp.stack([alms.real, alms.imag], axis=-1)
    return stacked.reshape(alms.shape[:-1] + (2 * alms.shape[-1],))


def _to_complex(v: jnp.ndarray) -> jnp.ndarray:
    """De-interleave real DOFs (..., 2 * n_alm) into complex alm (..., n_alm)."""
    pairs = v.reshape(v.shape[:-1] + (v.shape[-1] // 2, 2))
    return jax.lax.complex(pairs[..., 0], pairs[..., 1])


# ── public API ────────────────────────────────────────────────────────────────


def alm2map(
    alms: jnp.ndarray, nside: int, lmax: int, spin: int = 0, nthreads: int = 0
) -> jnp.ndarray:
    """Synthesis SHT for all bins via ducc0.

    Parameters
    ----------
    alms : jnp.ndarray
        (Nbins, n_alm) or (Nbins, 1, n_alm) for spin=0; (Nbins, 2, n_alm)
        for spin=2.
    nside : int
        HEALPix resolution (Python int, not a traced value).
    lmax : int
        Maximum multipole (Python int, not a traced value).
    spin : int, optional
        0 or 2, by default 0.
    nthreads : int, optional
        ducc0 thread count; 0 (default) means all available hardware
        threads.

    Returns
    -------
    jnp.ndarray
        (Nbins, n_pix) if input was (Nbins, n_alm), else
        (Nbins, ncomp, n_pix).
    """
    squeeze = alms.ndim == 2
    if squeeze:
        alms = alms[:, np.newaxis, :]
    fwd, _ = _sht_ops(int(nside), int(lmax), int(spin), int(nthreads))
    (out,) = fwd(_to_real(alms))
    return out[:, 0, :] if squeeze else out


def map2alm(
    maps: jnp.ndarray, lmax: int, spin: int = 0, nthreads: int = 0
) -> jnp.ndarray:
    """Analysis SHT for all bins via ducc0 (adjoint synthesis with pixel weights).

    Applies full pixel weights and (4π/n_pix) normalization, matching
    healpy.map2alm with use_pixel_weights=True. nside is inferred from
    the map size.

    Parameters
    ----------
    maps : jnp.ndarray
        (Nbins, n_pix) or (Nbins, 1, n_pix) for spin=0; (Nbins, 2, n_pix)
        for spin=2.
    lmax : int
        Maximum multipole (Python int, not a traced value).
    spin : int, optional
        0 or 2, by default 0.
    nthreads : int, optional
        ducc0 thread count; 0 (default) means all available hardware
        threads.

    Returns
    -------
    jnp.ndarray
        (Nbins, n_alm) if input was (Nbins, n_pix), else
        (Nbins, ncomp, n_alm).
    """
    squeeze = maps.ndim == 2
    if squeeze:
        maps = maps[:, np.newaxis, :]
    n_pix = maps.shape[-1]
    nside = hp.npix2nside(n_pix)
    w = get_pixel_weights(nside)
    _, adj = _sht_ops(int(nside), int(lmax), int(spin), int(nthreads))
    (v,) = adj(w * maps)
    out = _to_complex(v * _map2alm_scale(int(lmax), n_pix))
    return out[:, 0, :] if squeeze else out
