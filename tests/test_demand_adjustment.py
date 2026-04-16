"""Unit tests for pyRVtest/solve/demand_adjustment.py.

v0.4 step 4b lands ``_residualize_on_xd``, the shared 2SLS profile-out
helper used by every ``SupportsDemandAdjustment`` backend. Later
sub-commits (4d) land the unified ``compute_demand_adjustment`` function
and its tests will live here too.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyRVtest.solve.demand_adjustment import _residualize_on_xd


class TestResidualizeOnXd:
    """``_residualize_on_xd`` is the single source of truth for the 2SLS profile-out.

    The formula is algebraically equivalent to the inline one-liner
    ``dxi - X_D @ inv(X_D' Z_D W_D Z_D' X_D) @ X_D' Z_D W_D Z_D' @ dxi``
    used prior to step 4b in ``PyBLPBackend.xi_gradient`` and the inline
    analytical path in ``Problem._compute_analytical_demand_adjustment``.
    These tests assert algebraic identity to machine precision on random
    inputs across a variety of shapes.
    """

    @pytest.mark.parametrize("seed,N,K_x,K_z,n_theta", [
        (0, 100, 3, 5, 2),
        (1, 250, 5, 8, 4),
        (2, 500, 1, 4, 1),
        (3, 80, 7, 7, 6),       # exactly-identified (K_z == K_x)
        (4, 120, 2, 12, 3),     # heavy over-identification
    ])
    def test_matches_inline_formula(self, seed, N, K_x, K_z, n_theta):
        rng = np.random.default_rng(seed=seed)
        dxi = rng.normal(size=(N, n_theta))
        X_D = rng.normal(size=(N, K_x))
        Z_D = rng.normal(size=(N, K_z))
        W_D_raw = rng.normal(size=(K_z, K_z))
        W_D = W_D_raw @ W_D_raw.T + K_z * np.eye(K_z)  # positive-definite

        # Helper
        got = _residualize_on_xd(dxi, X_D, Z_D, W_D)

        # Reference: inline one-liner form from PyBLPBackend.xi_gradient pre-4b
        product = X_D @ np.linalg.inv(X_D.T @ Z_D @ W_D @ Z_D.T @ X_D) @ (
            X_D.T @ Z_D @ W_D @ Z_D.T @ dxi
        )
        expected = dxi - product

        np.testing.assert_allclose(
            got, expected, rtol=1e-11, atol=1e-11,
            err_msg=f"helper output diverges from inline formula at "
                    f"seed={seed} N={N} K_x={K_x} K_z={K_z} n_theta={n_theta}"
        )

    def test_empty_x_d_returns_dxi_unchanged(self):
        """If X_D has zero columns there is nothing to concentrate out."""
        rng = np.random.default_rng(seed=42)
        N, K_z, n_theta = 60, 3, 2
        dxi = rng.normal(size=(N, n_theta))
        X_D = np.empty((N, 0))
        Z_D = rng.normal(size=(N, K_z))
        W_D = np.eye(K_z)

        got = _residualize_on_xd(dxi, X_D, Z_D, W_D)

        # Not ``assert_allclose``: we want identity (not a copy) when the fast
        # path triggers. This documents the "no copy on empty X_D" contract.
        assert got is dxi

    def test_output_is_orthogonal_to_x_d_under_2sls_weight(self):
        """Standard 2SLS identity: X_D' Z_D W_D Z_D' (dxi - residualized) projects
        back onto X_D' Z_D W_D Z_D' X_D times the projection coefficients, so
        the residualized vector is orthogonal to X_D's column space under the
        (Z_D W_D Z_D') metric.
        """
        rng = np.random.default_rng(seed=7)
        N, K_x, K_z, n_theta = 150, 4, 6, 3
        dxi = rng.normal(size=(N, n_theta))
        X_D = rng.normal(size=(N, K_x))
        Z_D = rng.normal(size=(N, K_z))
        W_D_raw = rng.normal(size=(K_z, K_z))
        W_D = W_D_raw @ W_D_raw.T + K_z * np.eye(K_z)

        residualized = _residualize_on_xd(dxi, X_D, Z_D, W_D)

        # X_D' Z_D W_D Z_D' residualized  should be (very close to) zero,
        # since ``residualized`` is 2SLS orthogonal to X_D under this metric.
        orth = X_D.T @ Z_D @ W_D @ Z_D.T @ residualized
        np.testing.assert_allclose(
            orth, np.zeros_like(orth), atol=1e-8,
            err_msg="residualized output is not 2SLS-orthogonal to X_D"
        )
