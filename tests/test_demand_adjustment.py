"""Unit tests for pyRVtest/solve/demand_adjustment.py.

v0.4 step 4b lands ``_residualize_on_xd``, the shared 2SLS profile-out
helper used by every ``SupportsDemandAdjustment`` backend. Later
sub-commits (4d) land the unified ``compute_demand_adjustment`` function
and its tests will live here too.

v0.4 step 4c: cross-validation that ``LogitBackend`` and
``NestedLogitBackend``'s ``SupportsDemandAdjustment`` implementations
produce the same ``H`` and ``h_i`` as the inline
``Problem._compute_analytical_demand_adjustment`` on identical inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyRVtest
from pyRVtest.backends import LogitBackend, NestedLogitBackend
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


# ===========================================================================
# v0.4 step 4c: LogitBackend / NestedLogitBackend SupportsDemandAdjustment
# equivalence with Problem._compute_analytical_demand_adjustment.
# ===========================================================================

def _make_logit_fixture(seed: int = 42, T: int = 40, J: int = 3,
                       alpha: float = -1.5, beta_intercept: float = 0.0,
                       beta_x: float = 1.0, sigma: list | None = None,
                       with_nesting: bool = False) -> pd.DataFrame:
    """Synthetic logit / nested-logit dataset with enough columns for both the
    inline demand-adjustment path and SupportsDemandAdjustment backends.

    Columns: market_ids, firm_ids, shares, prices, x1, intercept, rival_x1,
    rival_x1_sq, z1, rival_z1, (optionally nesting_ids). No real demand
    inversion is performed; shares are drawn from random utilities so they
    are positive and sum to < 1 per market.
    """
    rng = np.random.default_rng(seed=seed)
    N = T * J
    market_ids = np.repeat(np.arange(T), J)
    firm_ids = np.tile(np.arange(J), T)
    x1 = rng.normal(size=N)
    prices = rng.uniform(1.0, 3.0, size=N)
    intercept = np.ones(N)
    u = rng.normal(size=N) * 0.5
    shares = np.zeros(N)
    for t in range(T):
        idx = np.where(market_ids == t)[0]
        e = np.exp(u[idx])
        shares[idx] = e / (1.0 + e.sum())
    # Rival-mean instruments (standard BLP-style)
    rival_x1 = np.zeros(N)
    for t in range(T):
        idx = np.where(market_ids == t)[0]
        for j in idx:
            others = [i for i in idx if i != j]
            rival_x1[j] = x1[others].mean()
    rival_x1_sq = rival_x1 ** 2
    # Cost-side inputs (pyRVtest Problem needs these)
    z1 = rng.normal(size=N) + 2.0
    rival_z1 = np.zeros(N)
    for t in range(T):
        idx = np.where(market_ids == t)[0]
        for j in idx:
            others = [i for i in idx if i != j]
            rival_z1[j] = z1[others].mean()
    data = {
        'market_ids': market_ids, 'firm_ids': firm_ids,
        'shares': shares, 'prices': prices,
        'x1': x1, 'intercept': intercept,
        'rival_x1': rival_x1, 'rival_x1_sq': rival_x1_sq,
        'z1': z1, 'rival_z1': rival_z1,
    }
    if with_nesting:
        # Two nests per market (products 0 in nest A, rest in B, so both nests
        # exist for J>=2 and within-nest shares are well-defined).
        nest_pattern = np.array(['A'] + ['B'] * (J - 1))
        data['nesting_ids'] = np.tile(nest_pattern, T)
    return pd.DataFrame(data)


def _assemble_problem(df: pd.DataFrame, demand_params: dict) -> pyRVtest.Problem:
    """Build the minimum Problem that flows through
    `_compute_analytical_demand_adjustment`.
    """
    return pyRVtest.Problem(
        cost_formulation=pyRVtest.Formulation('1 + z1'),
        instrument_formulation=pyRVtest.Formulation('0 + rival_z1'),
        model_formulations=(
            pyRVtest.ModelFormulation(
                model_downstream='bertrand', ownership_downstream='firm_ids',
            ),
        ),
        product_data=df,
        demand_params=demand_params,
    )


class TestLogitBackendDemandAdjustmentEquivalence:
    """LogitBackend (plain logit) equivalence vs inline analytical path."""

    @pytest.fixture(scope='class')
    def fixture(self):
        df = _make_logit_fixture(with_nesting=False)
        alpha = -1.5
        beta = np.array([0.0, 1.0])  # intercept + x1
        demand_params = {
            'alpha': alpha, 'sigma': [],
            'beta': beta,
            'x_columns': ['intercept', 'x1'],
            'demand_instrument_columns': ['rival_x1', 'rival_x1_sq', 'intercept', 'x1'],
        }
        return df, demand_params

    def test_h_i_matches_inline(self, fixture):
        """xi from backend.demand_moments() reproduces the inline path's h_i = Z_D * xi."""
        df, dp = fixture
        problem = _assemble_problem(df, dp)
        markups, _, _ = problem._perturb_and_build_markups()
        M = problem.M
        _, _, _, h_i_inline, _ = problem._compute_analytical_demand_adjustment(
            M, problem.N, markups,
            advalorem_tax_adj=[1.0] * M, cost_scaling=[0.0] * M,
        )

        backend = LogitBackend(
            alpha=dp['alpha'], product_data=df,
            beta=dp['beta'], x_columns=dp['x_columns'],
            demand_instrument_columns=dp['demand_instrument_columns'],
        )
        xi_b, Z_D_b, _ = backend.demand_moments()
        h_i_backend = Z_D_b * xi_b[:, np.newaxis]
        np.testing.assert_allclose(
            h_i_backend, h_i_inline, atol=1e-12,
            err_msg="LogitBackend.demand_moments xi diverges from inline path"
        )

    def test_H_matches_inline(self, fixture):
        """(1/N) Z_D' @ backend.xi_gradient() matches inline H (the profiled moment)."""
        df, dp = fixture
        problem = _assemble_problem(df, dp)
        markups, _, _ = problem._perturb_and_build_markups()
        M, N = problem.M, problem.N
        _, _, H_inline, _, _ = problem._compute_analytical_demand_adjustment(
            M, N, markups,
            advalorem_tax_adj=[1.0] * M, cost_scaling=[0.0] * M,
        )

        backend = LogitBackend(
            alpha=dp['alpha'], product_data=df,
            beta=dp['beta'], x_columns=dp['x_columns'],
            demand_instrument_columns=dp['demand_instrument_columns'],
        )
        _, Z_D_b, _ = backend.demand_moments()
        xi_grad = backend.xi_gradient()
        H_backend = (1.0 / N) * Z_D_b.T @ xi_grad
        np.testing.assert_allclose(
            H_backend, H_inline, atol=1e-12,
            err_msg="LogitBackend.xi_gradient diverges from inline H"
        )

    def test_jacobian_gradient_alpha_column_equals_D_over_alpha(self, fixture):
        """d(D)/d(alpha) = D / alpha since D is linear in alpha for plain logit."""
        df, dp = fixture
        backend = LogitBackend(
            alpha=dp['alpha'], product_data=df,
            beta=dp['beta'], x_columns=dp['x_columns'],
            demand_instrument_columns=dp['demand_instrument_columns'],
        )
        for t in np.unique(df['market_ids'])[:3]:  # sample a few markets
            grad = backend.jacobian_gradient(market_id=t)
            D_t = backend.compute_jacobian(market_id=t)
            np.testing.assert_allclose(
                grad[:, :, 0], D_t / dp['alpha'], atol=1e-14,
                err_msg=f"d(D)/d(alpha) != D/alpha for market {t}"
            )
            assert grad.shape[2] == 1, (
                f"plain logit jacobian_gradient should have 1 theta column, got {grad.shape[2]}"
            )


class TestNestedLogitBackendDemandAdjustmentEquivalence:
    """NestedLogitBackend (1-level nested logit) equivalence vs inline analytical path."""

    @pytest.fixture(scope='class')
    def fixture(self):
        df = _make_logit_fixture(with_nesting=True)
        alpha = -1.5
        sigma = [0.3]
        beta = np.array([0.0, 1.0])
        demand_params = {
            'alpha': alpha, 'sigma': sigma,
            'beta': beta,
            'x_columns': ['intercept', 'x1'],
            'demand_instrument_columns': ['rival_x1', 'rival_x1_sq', 'intercept', 'x1'],
            'nesting_ids_columns': ['nesting_ids'],
        }
        return df, demand_params

    def test_h_i_matches_inline(self, fixture):
        df, dp = fixture
        problem = _assemble_problem(df, dp)
        markups, _, _ = problem._perturb_and_build_markups()
        M = problem.M
        _, _, _, h_i_inline, _ = problem._compute_analytical_demand_adjustment(
            M, problem.N, markups,
            advalorem_tax_adj=[1.0] * M, cost_scaling=[0.0] * M,
        )
        backend = NestedLogitBackend(
            alpha=dp['alpha'], sigma=dp['sigma'], product_data=df,
            nesting_ids_columns=dp['nesting_ids_columns'],
            beta=dp['beta'], x_columns=dp['x_columns'],
            demand_instrument_columns=dp['demand_instrument_columns'],
        )
        xi_b, Z_D_b, _ = backend.demand_moments()
        h_i_backend = Z_D_b * xi_b[:, np.newaxis]
        np.testing.assert_allclose(
            h_i_backend, h_i_inline, atol=1e-12,
            err_msg="NestedLogitBackend.demand_moments xi diverges from inline path"
        )

    def test_H_matches_inline(self, fixture):
        df, dp = fixture
        problem = _assemble_problem(df, dp)
        markups, _, _ = problem._perturb_and_build_markups()
        M, N = problem.M, problem.N
        _, _, H_inline, _, _ = problem._compute_analytical_demand_adjustment(
            M, N, markups,
            advalorem_tax_adj=[1.0] * M, cost_scaling=[0.0] * M,
        )
        backend = NestedLogitBackend(
            alpha=dp['alpha'], sigma=dp['sigma'], product_data=df,
            nesting_ids_columns=dp['nesting_ids_columns'],
            beta=dp['beta'], x_columns=dp['x_columns'],
            demand_instrument_columns=dp['demand_instrument_columns'],
        )
        _, Z_D_b, _ = backend.demand_moments()
        xi_grad = backend.xi_gradient()
        H_backend = (1.0 / N) * Z_D_b.T @ xi_grad
        np.testing.assert_allclose(
            H_backend, H_inline, atol=1e-12,
            err_msg="NestedLogitBackend.xi_gradient diverges from inline H"
        )

    def test_jacobian_gradient_shape_and_sigma_column(self, fixture):
        """jacobian_gradient(t) returns (J_t, J_t, 2) with sigma column matching
        `_nested_logit_jacobian_derivative` directly.
        """
        from pyRVtest.backends.logit import _nested_logit_jacobian_derivative
        df, dp = fixture
        backend = NestedLogitBackend(
            alpha=dp['alpha'], sigma=dp['sigma'], product_data=df,
            nesting_ids_columns=dp['nesting_ids_columns'],
            beta=dp['beta'], x_columns=dp['x_columns'],
            demand_instrument_columns=dp['demand_instrument_columns'],
        )
        for t in np.unique(df['market_ids'])[:3]:
            idx = np.where(df['market_ids'].values == t)[0]
            s_t = df['shares'].values[idx]
            nest_ids_t = df['nesting_ids'].values[idx]

            grad = backend.jacobian_gradient(market_id=t)
            assert grad.shape == (len(idx), len(idx), 2), (
                f"expected shape ({len(idx)}, {len(idx)}, 2) got {grad.shape}"
            )
            # Sigma column — compute reference directly
            expected = _nested_logit_jacobian_derivative(
                dp['alpha'], dp['sigma'], s_t, [nest_ids_t], 0
            )
            np.testing.assert_allclose(
                grad[:, :, 1], expected, atol=1e-14,
                err_msg=f"d(D)/d(sigma_0) mismatch for market {t}"
            )


class TestLogitBackendSigmaFiltering:
    """NestedLogitBackend filters sigmas of exactly 0 at construction."""

    def test_zero_sigmas_are_dropped(self):
        df = _make_logit_fixture(with_nesting=True)
        backend = NestedLogitBackend(
            alpha=-1.5, sigma=[0.0, 0.3, 0.0], product_data=df,
            nesting_ids_columns=['nesting_ids'],
        )
        assert backend._sigma == [0.3], (
            "NestedLogitBackend should filter sigmas of exactly 0 at construction"
        )
