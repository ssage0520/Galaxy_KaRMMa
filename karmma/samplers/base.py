"""Shared whitening/preconditioning base class for sampler backends (NUTS, MCLMC)."""

from collections.abc import Callable

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import cg

from karmma.forward_model import ForwardModel
from karmma.structs import KarmmaPosition, ThetaParams, WhitenedKarmmaPosition


class WhitenedSampler:
    """Base class for sampler backends, holding shared whitening/IMM-preconditioning machinery.

    Parameters
    ----------
    model : ForwardModel
        The forward model to sample from.

    Attributes
    ----------
    model : ForwardModel
        The forward model being sampled from.
    V : jnp.ndarray or None
        Eigenvectors of the whitening transform; `None` until `_build_reparam`
        is called.
    w : jnp.ndarray or None
        Eigenvalues of the whitening transform; `None` until `_build_reparam`
        is called.
    theta0 : ThetaParams or None
        Reference theta the whitening transform is centered on; `None`
        until `_build_reparam` is called.
    """

    def __init__(self, model: ForwardModel) -> None:
        self.model = model
        self.V = None
        self.w = None
        self.theta0 = None

    def dense_theta_imm(
        self,
        position: KarmmaPosition,
        tol: float = 1e-3,
        maxiter: int = 300,
        kappa_max: float = 1e9,
        verbose: bool = True,
    ) -> np.ndarray:
        """Compute a dense theta-only covariance-like matrix via Schur complement + CG.

        Parameters
        ----------
        position : KarmmaPosition
            Position to linearize around; `position.theta` must be set
            (requires the sampler's `infer_theta=True`).
        tol : float, optional
            CG solver tolerance, by default 1e-3.
        maxiter : int, optional
            CG solver maximum iterations, by default 300.
        kappa_max : float, optional
            Maximum condition number enforced on the corrected matrix via
            eigenvalue-magnitude clipping, by default 1e9.
        verbose : bool, optional
            Whether to print progress and diagnostic statistics, by default True.

        Returns
        -------
        np.ndarray
            Dense (n_theta, n_theta) matrix, field-major/bin-minor layout
            matching `jax.flatten_util.ravel_pytree(ThetaParams(...))`.

        Notes
        -----
        Marginalizes over the xlm block with `n_theta` CG solves against
        `H_xx`, then fixes the resulting indefinite `n_theta`×`n_theta`
        Schur complement to positive-definite via `|λ|` eigenvalue
        correction.
        """
        n_theta = len(ThetaParams._fields) * self.model.Nbins

        # ravel_pytree matches BlackJax's pytree flattening: xlm-first, theta-last.
        flat_pos, unravel_fn = jax.flatten_util.ravel_pytree(position)
        N_full = flat_pos.shape[0]
        n_x = N_full - n_theta

        def _flat_log_prob(flat: jax.Array) -> jax.Array:
            """Evaluate log_prob on a flattened position vector."""
            return self.model.log_prob(unravel_fn(flat))

        @jax.jit
        def _hvp(v: jax.Array) -> jax.Array:
            """Compute the negative log-density Hessian-vector product."""
            _, g = jax.jvp(jax.grad(_flat_log_prob), (flat_pos,), (v,))
            return -g

        @jax.jit
        def _hvp_xx(vx: jax.Array) -> jax.Array:
            """Restrict `_hvp` to the xlm (non-theta) block."""
            v_full = jnp.zeros(N_full).at[:n_x].set(vx)
            return _hvp(v_full)[:n_x]

        if verbose:
            print(
                f"dense_theta_imm: step 1 — {n_theta} b-indicator HVPs ...", flush=True
            )
        # HVP against unit vector e_{n_x+i} extracts the (n_x+i)-th row of the
        # full Hessian; stacking one row per theta index gives every row of
        # the full Hessian that touches the theta block.
        rows_b = jnp.stack(
            [_hvp(jnp.zeros(N_full).at[n_x + i].set(1.0)) for i in range(n_theta)]
        )
        H_bb_est = rows_b[:, n_x:]
        H_bx_est = rows_b[:, :n_x]

        if verbose:
            n_finite = int(jnp.sum(jnp.all(jnp.isfinite(rows_b), axis=1)))
            abs_rows_b = jnp.abs(rows_b)
            print(
                f"  HVP finiteness: {n_finite}/{n_theta} finite  |  "
                f"|HVP| range: min={abs_rows_b.min():.2e} max={abs_rows_b.max():.2e}"
            )
            print(
                f"dense_theta_imm: step 2 — {n_theta} CG solves "
                f"(tol={tol}, maxiter={maxiter}) ...",
                flush=True,
            )
        X = jnp.stack(
            [
                cg(_hvp_xx, H_bx_est[j], tol=tol, maxiter=maxiter)[0]
                for j in range(n_theta)
            ]
        )

        # Schur complement of the theta block: H_bb - H_bx @ Hxx^-1 @ H_bx^T,
        # with X solving Hxx @ X = H_bx^T via CG above.
        precision_bb = H_bb_est - H_bx_est @ X.T

        if verbose:
            evals = np.array(jnp.linalg.eigvalsh(precision_bb))
            resid = np.array(
                jax.vmap(
                    lambda x, r: jnp.linalg.norm(_hvp_xx(x) - r) / jnp.linalg.norm(r)
                )(X, H_bx_est)
            )
            print(f"  CG rel residuals: max={resid.max():.2e}  mean={resid.mean():.2e}")
            print(
                f"  Schur eigenvalues: min={evals.min():.4e}  max={evals.max():.4e}  "
                f"negative={np.sum(evals < 0)}"
            )

        S = 0.5 * (precision_bb + precision_bb.T)
        w, U = jnp.linalg.eigh(S)
        # `position` isn't the true MAP, so the Hessian isn't guaranteed PSD
        # here — genuine negative curvature can appear, not just CG-solve
        # noise. This is only used to build an initial guess (the whitening
        # scale), so only the magnitude of the curvature matters; |w| keeps
        # that while discarding the sign.
        w_fixed = jnp.clip(jnp.abs(w), min=float(jnp.max(jnp.abs(w))) / kappa_max)

        # precision_bb (via _hvp's -Hessian(log_prob) convention) is a
        # precision-like matrix; dividing by (rather than multiplying by)
        # the eigenvalues inverts it into the covariance-like matrix returned.
        return np.array((U / w_fixed) @ U.T)

    def _build_reparam(
        self,
        initial_position: KarmmaPosition,
        tol: float = 1e-3,
        maxiter: int = 300,
        kappa_max: float = 1e9,
        verbose: bool = True,
    ) -> None:
        """Compute and store the whitening eigenbasis transform for theta.

        Sets `self.V`, `self.w`, and `self.theta0`, used by `theta_to_phi`/
        `phi_to_theta`.

        Parameters
        ----------
        initial_position : KarmmaPosition
            Position to build the reparametrization around; forwarded to
            `dense_theta_imm`.
        tol : float, optional
            Forwarded to `dense_theta_imm`, by default 1e-3.
        maxiter : int, optional
            Forwarded to `dense_theta_imm`, by default 300.
        kappa_max : float, optional
            Forwarded to `dense_theta_imm`, by default 1e9.
        verbose : bool, optional
            Forwarded to `dense_theta_imm`, by default True.
        """
        dense_theta_matrix = self.dense_theta_imm(
            initial_position, tol, maxiter, kappa_max, verbose
        )
        self.w, self.V = jnp.linalg.eigh(jnp.asarray(dense_theta_matrix))
        self.theta0 = initial_position.theta

    def theta_to_phi(self, theta: ThetaParams) -> jnp.ndarray:
        """Transform physical theta to whitened phi, via the eigenbasis transform.

        Requires `_build_reparam` to have been called first, since it
        depends on `self.V`/`self.w`/`self.theta0`.

        Parameters
        ----------
        theta : ThetaParams
            Physical theta to transform.

        Returns
        -------
        jnp.ndarray
            Flat whitened phi vector.
        """
        theta_flat, _ = jax.flatten_util.ravel_pytree(theta)
        theta0_flat, _ = jax.flatten_util.ravel_pytree(self.theta0)
        return (self.V.T @ (theta_flat - theta0_flat)) / jnp.sqrt(self.w)

    def phi_to_theta(self, phi: jnp.ndarray) -> ThetaParams:
        """Transform whitened phi back to physical theta, via the eigenbasis transform.

        Requires `_build_reparam` to have been called first, since it
        depends on `self.V`/`self.w`/`self.theta0`.

        Parameters
        ----------
        phi : jnp.ndarray
            Flat whitened phi vector.

        Returns
        -------
        ThetaParams
            Physical theta.
        """
        theta0_flat, unravel = jax.flatten_util.ravel_pytree(self.theta0)
        theta_flat = theta0_flat + self.V @ (phi * jnp.sqrt(self.w))
        return unravel(theta_flat)

    def _prepare_sampling(
        self, initial_position: KarmmaPosition
    ) -> tuple[WhitenedKarmmaPosition, Callable[[WhitenedKarmmaPosition], jax.Array]]:
        """Whiten `initial_position` and build a phi-space log_prob wrapper.

        Builds the whitening eigenbasis as a side effect (see
        `_build_reparam`). This is the setup shared by every sampler
        backend's `sample()`.

        Parameters
        ----------
        initial_position : KarmmaPosition
            Physical (theta-space) starting position.

        Returns
        -------
        sampling_position : WhitenedKarmmaPosition
            `initial_position`, whitened into phi-space.
        log_prob : Callable[[WhitenedKarmmaPosition], jax.Array]
            Jitted log-density function operating on whitened positions.
        """
        self._build_reparam(initial_position)
        sampling_position = WhitenedKarmmaPosition(
            xlm=initial_position.xlm, phi=self.theta_to_phi(initial_position.theta)
        )

        def log_prob(params: WhitenedKarmmaPosition) -> jax.Array:
            """Evaluate the model log-density on a whitened (phi-space) position."""
            theta = self.phi_to_theta(params.phi)
            return self.model.log_prob(KarmmaPosition(xlm=params.xlm, theta=theta))

        # Explicit jit matters for callers whose init/warmup entry points
        # aren't themselves jit-decorated (e.g. MCLMC's mclmc.init); harmless
        # where the enclosing lax.scan would compile it anyway (e.g. NUTS).
        return sampling_position, jax.jit(log_prob)

    def _unwhiten(self, states: WhitenedKarmmaPosition) -> KarmmaPosition:
        """Convert whitened-phi-space sampled states back to physical theta-space.

        Parameters
        ----------
        states : WhitenedKarmmaPosition
            Sampled states in whitened phi-space, batched over samples.

        Returns
        -------
        KarmmaPosition
            Sampled states in physical theta-space, batched over samples.
        """
        theta = jax.vmap(self.phi_to_theta)(states.phi)
        return KarmmaPosition(xlm=states.xlm, theta=theta)
