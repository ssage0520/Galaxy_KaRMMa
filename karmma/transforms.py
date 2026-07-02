"""SHT transforms for KaRMMa.

ducc0-based, JIT-compatible. Supports jax.grad (reverse mode) and
jax.hessian = jacfwd(jacrev) (forward-over-reverse). All bins are passed
as a single ntrans call. nside/lmax/spin must be Python literals (not traced
JAX values). Spin-2 is supported via the spin argument.

Design
------
Two-layer structure so that jacfwd can differentiate through the gradient trace:

  Inner layer  (_synthesis, _adjoint_synthesis) — custom_jvp
    Bare linear callbacks. JVP = same linear op applied to the tangent.
    vmap_method='sequential' required for jax.hessian (which vmaps internally).

  Outer layer  (_alm2map, _map2alm) — custom_vjp
    Validated adjoints (reverse mode). _fwd functions call the inner
    custom_jvp primitives — NOT the outer custom_vjp functions — so that
    jacfwd can apply JVP through them.

jax.hessian works; jacrev(jacrev) does not (transposing a pure_callback
is unsupported by JAX).
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import healpy as hp
import numpy as np
import astropy.io.fits as fits
from functools import partial, lru_cache
import time
import urllib.request
from pathlib import Path

import ducc0.sht
import ducc0.healpix


@lru_cache(maxsize=1)
def get_pixel_weights(nside: int) -> np.ndarray:
    """Return the HEALPix full pixel weight map for nside, downloaded on first use.

    Equivalent to healpy's use_pixel_weights=True. Returns an (n_pix,) array
    with values near 1.0 (stored as w + 1 per the FITS convention). Results
    are cached to ~/.cache/karmma/full_weights/.
    """
    nside = int(nside)
    nside_str = f"{nside:04d}"
    filename  = f"healpix_full_weights_nside_{nside_str}.fits"
    cache_dir = Path.home() / ".cache" / "karmma" / "full_weights"
    path      = cache_dir / filename

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
                    print(f"  Attempt {attempt+1} failed ({e}), retrying...")
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
            w8map[psouth:psouth + qp4] = w8map[pnorth:pnorth + qp4]

        pnorth += qp4
        vpix += (qpix + 1) // 2 + 1 - ((qpix % 2) | shifted)

    return w8map + 1.0


@lru_cache(maxsize=2)
def get_ms(lmax):
    """Return the m-index array for all alm coefficients up to lmax.

    Returned as a plain numpy array so it is safe to close over inside
    jax.pure_callback and jax.jit (a jnp array would capture a tracer on
    first call and break reuse across different trace contexts).
    """
    _, ms = hp.Alm.getlm(lmax)
    return ms


@lru_cache(maxsize=4)
def _healpix_geo(nside: int) -> dict:
    """HEALPix ring geometry for ducc0, cached per nside. Unpack with **geo."""
    geo = ducc0.healpix.Healpix_Base(nside, 'RING').sht_info()
    return {k: geo[k] for k in ('theta', 'phi0', 'nphi', 'ringstart')}


# ── inner layer: bare linear callbacks, forward-mode capable ──────────────────
#
# JVP of a linear map L: d/dε L(x + ε v)|_{ε=0} = L(v).
# vmap_method='sequential' is required: jax.hessian vmaps internally over
# Hessian basis directions, and pure_callback requires an explicit vmap_method.
# Reverse-mode flow never reaches these primitives — it is fully handled by the
# custom_vjp wrappers below.

@partial(jax.custom_jvp, nondiff_argnums=(1, 2, 3))
def _synthesis(alms, nside, lmax, spin):
    """Bare synthesis SHT: (Nbins, ncomp, n_alm) complex → (Nbins, ncomp, n_pix) real."""
    Nbins, ncomp = alms.shape[0], alms.shape[1]
    n_pix = hp.nside2npix(nside)
    geo   = _healpix_geo(nside)
    def _cb(a):
        return ducc0.sht.synthesis(
            alm=np.asarray(a, dtype=np.complex128),
            **geo, lmax=lmax, mmax=lmax, spin=spin, nthreads=0,
        )
    return jax.pure_callback(
        _cb, jax.ShapeDtypeStruct((Nbins, ncomp, n_pix), jnp.float64),
        alms, vmap_method='sequential',
    )

@_synthesis.defjvp
def _synthesis_jvp(nside, lmax, spin, primals, tangents):
    (alms,), (dalms,) = primals, tangents
    return (_synthesis(alms, nside, lmax, spin),
            _synthesis(dalms, nside, lmax, spin))


@partial(jax.custom_jvp, nondiff_argnums=(1, 2, 3))
def _adjoint_synthesis(maps, nside, lmax, spin):
    """Bare adjoint-synthesis SHT: (Nbins, ncomp, n_pix) real → (Nbins, ncomp, n_alm) complex.

    No pixel weights or normalization — those are applied in JAX-space by the
    callers so they are differentiated natively.
    """
    Nbins, ncomp = maps.shape[0], maps.shape[1]
    n_alm = hp.Alm.getsize(lmax)
    geo   = _healpix_geo(nside)
    def _cb(g):
        return ducc0.sht.adjoint_synthesis(
            map=np.asarray(g, dtype=np.float64),
            **geo, lmax=lmax, mmax=lmax, spin=spin, nthreads=0,
        )
    return jax.pure_callback(
        _cb, jax.ShapeDtypeStruct((Nbins, ncomp, n_alm), jnp.complex128),
        maps, vmap_method='sequential',
    )

@_adjoint_synthesis.defjvp
def _adjoint_synthesis_jvp(nside, lmax, spin, primals, tangents):
    (maps,), (dmaps,) = primals, tangents
    return (_adjoint_synthesis(maps, nside, lmax, spin),
            _adjoint_synthesis(dmaps, nside, lmax, spin))


# ── outer layer: validated adjoints via custom_vjp ────────────────────────────
#
# _fwd functions call the inner custom_jvp primitives (not the outer custom_vjp
# functions), so that jacfwd can apply JVP through them.

@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3))
def _alm2map(alms, nside, lmax, spin):
    """Synthesis SHT: (Nbins, ncomp, n_alm) → (Nbins, ncomp, n_pix)."""
    return _synthesis(alms, nside, lmax, spin)

def _alm2map_fwd(alms, nside, lmax, spin):
    return _synthesis(alms, nside, lmax, spin), ()

def _alm2map_bwd(nside, lmax, spin, _, g_maps):
    # VJP of synthesis: adjoint_synthesis(g_maps), then double m>0 modes to
    # account for conjugate-symmetry (each m>0 coefficient appears once in
    # the stored alm but contributes to both +m and -m in the full sum).
    ms    = get_ms(lmax)
    g_alm = _adjoint_synthesis(g_maps, nside, lmax, spin)
    return (jnp.conj(jnp.where(ms == 0, g_alm, 2.0 * g_alm)),)

_alm2map.defvjp(_alm2map_fwd, _alm2map_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def _map2alm(maps, lmax, spin):
    """Adjoint synthesis SHT: (Nbins, ncomp, n_pix) → (Nbins, ncomp, n_alm).

    Applies full pixel weights and (4π/n_pix) normalization, matching
    healpy.map2alm with use_pixel_weights=True. Weights are applied in
    JAX-space so that _adjoint_synthesis stays a bare reusable primitive.
    """
    n_pix = maps.shape[-1]
    nside = hp.npix2nside(n_pix)
    w     = get_pixel_weights(nside)
    return _adjoint_synthesis(w * maps, nside, lmax, spin) * (4.0 * jnp.pi / n_pix)

def _map2alm_fwd(maps, lmax, spin):
    n_pix  = maps.shape[-1]
    nside  = hp.npix2nside(n_pix)
    w      = get_pixel_weights(nside)
    result = _adjoint_synthesis(w * maps, nside, lmax, spin) * (4.0 * jnp.pi / n_pix)
    return result, (n_pix,)

def _map2alm_bwd(lmax, spin, res, g_alm):
    # VJP of (4π/n_pix) · adjoint_synthesis(w · maps):
    # result is w · synthesis(g_alm') · (4π/n_pix), where g_alm' halves m>0
    # modes (inverse of the 2x doubling in _alm2map_bwd).
    n_pix, = res
    nside  = hp.npix2nside(n_pix)
    w      = get_pixel_weights(nside)
    ms     = get_ms(lmax)
    g_scaled = jnp.where(ms == 0, jnp.conj(g_alm), 0.5 * jnp.conj(g_alm))
    return (w * _synthesis(g_scaled, nside, lmax, spin) * (4.0 * jnp.pi / n_pix),)

_map2alm.defvjp(_map2alm_fwd, _map2alm_bwd)


# ── public API ────────────────────────────────────────────────────────────────

def alm2map(alms, nside, lmax, spin=0):
    """Synthesis SHT for all bins via ducc0.

    Args:
        alms:  (Nbins, n_alm) or (Nbins, 1, n_alm) for spin=0;
               (Nbins, 2, n_alm) for spin=2.
        nside: HEALPix resolution (Python int, not a traced value).
        lmax:  Maximum multipole (Python int, not a traced value).
        spin:  0 or 2.

    Returns:
        (Nbins, n_pix) if input was (Nbins, n_alm), else (Nbins, ncomp, n_pix).
    """
    squeeze = alms.ndim == 2
    if squeeze:
        alms = alms[:, np.newaxis, :]
    out = _alm2map(alms, nside, lmax, spin)
    return out[:, 0, :] if squeeze else out


def map2alm(maps, lmax, spin=0):
    """Analysis SHT for all bins via ducc0 (adjoint synthesis with pixel weights).

    Applies full pixel weights and (4π/n_pix) normalization, matching
    healpy.map2alm with use_pixel_weights=True. nside is inferred from the
    map size.

    Args:
        maps: (Nbins, n_pix) or (Nbins, 1, n_pix) for spin=0;
              (Nbins, 2, n_pix) for spin=2.
        lmax: Maximum multipole (Python int, not a traced value).
        spin: 0 or 2.

    Returns:
        (Nbins, n_alm) if input was (Nbins, n_pix), else (Nbins, ncomp, n_alm).
    """
    squeeze = maps.ndim == 2
    if squeeze:
        maps = maps[:, np.newaxis, :]
    out = _map2alm(maps, lmax, spin)
    return out[:, 0, :] if squeeze else out
