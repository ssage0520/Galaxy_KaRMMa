import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import cg

from karmma.structs import KarmmaPosition, ThetaParams


class WhitenedSampler:
    def __init__(self, model):
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
        """Dense theta-only covariance-like matrix via Schur complement + CG.

        Marginalises over the xlm block with n_theta CG solves against H_xx,
        then fixes the resulting indefinite n_theta×n_theta Schur complement
        to PD via |λ| eigenvalue correction. Used directly for eigenbasis
        reparametrization (see _build_reparam).

        Requires `infer_theta=True` (theta must be part of the sampled
        position for the Schur complement over the theta block to apply).

        Returns
        -------
        np.ndarray of shape (n_theta, n_theta), field-major/bin-minor layout
        matching jax.flatten_util.ravel_pytree(ThetaParams(...)).
        """
        n_theta = len(ThetaParams._fields) * self.model.Nbins

        # ravel_pytree matches BlackJax's pytree flattening: xlm-first, theta-last.
        flat_pos, unravel_fn = jax.flatten_util.ravel_pytree(position)
        N_full = flat_pos.shape[0]
        n_x = N_full - n_theta

        def _flat_log_prob(flat):
            return self.model.log_prob(unravel_fn(flat))

        @jax.jit
        def _hvp(v):
            _, g = jax.jvp(jax.grad(_flat_log_prob), (flat_pos,), (v,))
            return -g

        @jax.jit
        def _hvp_xx(vx):
            v_full = jnp.zeros(N_full).at[:n_x].set(vx)
            return _hvp(v_full)[:n_x]

        if verbose:
            print(
                f"dense_theta_imm: step 1 — {n_theta} b-indicator HVPs ...", flush=True
            )
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
        w_fixed = jnp.clip(jnp.abs(w), min=float(jnp.max(jnp.abs(w))) / kappa_max)

        return np.array((U / w_fixed) @ U.T)

    def _build_reparam(
        self,
        initial_position: KarmmaPosition,
        tol: float = 1e-3,
        maxiter: int = 300,
        kappa_max: float = 1e9,
        verbose: bool = True,
    ) -> None:
        """Computes dense_theta_imm(...) and eigendecomposes it, storing the
        whitening transform (self.V, self.w, self.theta0) for theta_to_phi/
        phi_to_theta."""
        dense_theta_matrix = self.dense_theta_imm(
            initial_position, tol, maxiter, kappa_max, verbose
        )
        self.w, self.V = jnp.linalg.eigh(jnp.asarray(dense_theta_matrix))
        self.theta0 = initial_position.theta

    def theta_to_phi(self, theta: ThetaParams) -> jnp.ndarray:
        """Physical theta -> whitened phi, via the eigenbasis transform set
        by _build_reparam."""
        theta_flat, _ = jax.flatten_util.ravel_pytree(theta)
        theta0_flat, _ = jax.flatten_util.ravel_pytree(self.theta0)
        return (self.V.T @ (theta_flat - theta0_flat)) / jnp.sqrt(self.w)

    def phi_to_theta(self, phi: jnp.ndarray) -> ThetaParams:
        """Whitened phi -> physical theta, via the eigenbasis transform set
        by _build_reparam."""
        theta0_flat, unravel = jax.flatten_util.ravel_pytree(self.theta0)
        theta_flat = theta0_flat + self.V @ (phi * jnp.sqrt(self.w))
        return unravel(theta_flat)
