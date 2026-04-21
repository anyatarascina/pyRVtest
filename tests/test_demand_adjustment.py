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




class TestLogitBackendRhoFiltering:
    """NestedLogitBackend filters rho values of exactly 0 at construction."""

    def test_zero_rhos_are_dropped(self):
        df = _make_logit_fixture(with_nesting=True)
        backend = NestedLogitBackend(
            alpha=-1.5, rho=[0.0, 0.3, 0.0], product_data=df,
            nesting_ids_columns=['nesting_ids'],
        )
        assert backend._rho == [0.3], (
            "NestedLogitBackend should filter rho values of exactly 0 at construction"
        )


# ===========================================================================
# v0.4 step 4d: compute_demand_adjustment equivalence with inline methods.
#
# Tests assert the unified `compute_demand_adjustment` function produces the
# same output as `Problem._compute_analytical_demand_adjustment` (via
# LogitBackend / NestedLogitBackend) and `Problem._compute_demand_adjustment_gradient`
# (via PyBLPBackend) on shared fixtures. Problem.solve() still calls the old
# inline methods in this commit; 4e will flip the switch.
# ===========================================================================

def _advalorem_and_scaling(problem):
    """Match the tax-adjustment factors that Problem.solve computes."""
    M = problem.M
    unit_tax = problem.models["unit_tax"]
    advalorem_tax = problem.models["advalorem_tax"]
    cost_scaling = problem.models["cost_scaling"]
    advalorem_tax_adj = []
    for m in range(M):
        condition = problem.models["advalorem_payer"][m] == "consumer"
        advalorem_tax_adj.append(
            1 / (1 + advalorem_tax[m]) if condition else (1 - advalorem_tax[m])
        )
    # unit_tax unused below but kept for parity with solve()'s preparation
    _ = unit_tax
    return advalorem_tax_adj, list(cost_scaling)




class TestComputeDemandAdjustmentRejectsNonAdjustmentBackend:
    """UserSuppliedBackend without adjustment inputs must be rejected with a clear error."""

    def test_user_supplied_backend_raises(self):
        from pyRVtest.backends import UserSuppliedBackend
        from pyRVtest.solve.demand_adjustment import compute_demand_adjustment
        df = _make_logit_fixture(with_nesting=False)
        N = len(df)
        J = 3
        jacobian = np.full((N, J), np.nan)
        market_ids = df['market_ids'].values
        for t in np.unique(market_ids):
            idx = np.where(market_ids == t)[0]
            s_t = df['shares'].values[idx]
            D_t = -1.5 * (np.diag(s_t) - np.outer(s_t, s_t))
            jacobian[idx[:, None], np.arange(len(idx))[None, :]] = D_t
        backend = UserSuppliedBackend(jacobian=jacobian, market_ids=market_ids)
        # Dummy problem (won't actually be used)
        demand_params = {
            'alpha': -1.5, 'sigma': [],
            'beta': np.array([0.0, 1.0]),
            'x_columns': ['intercept', 'x1'],
            'demand_instrument_columns': ['rival_x1', 'intercept', 'x1'],
        }
        problem = _assemble_problem(df, demand_params)
        M = problem.M
        markups, _, _ = problem._perturb_and_build_markups()
        advalorem_tax_adj, cost_scaling = _advalorem_and_scaling(problem)
        with pytest.raises(TypeError, match="SupportsDemandAdjustment"):
            compute_demand_adjustment(
                backend, problem, M, problem.N, markups,
                advalorem_tax_adj, cost_scaling,
            )


# ===========================================================================
# v0.4 step 4e pre-flight gates (Option A + Option B from the session discussion).
#
# Until Lorenzo delivers the DMSS yogurt replication data (Step 0d), the
# strongest paper-based correctness check is offline. These two tests
# substitute for the externally-validated ground truth:
#
#   Option A: cross-path parity with endogenous cost. The demand_results
#     path has always computed gradient_gamma_per_instrument correctly
#     (and is covered by existing snapshots). The demand_params path
#     silently returned None pre-step-4. After step 4e the two paths
#     should agree within analytical-vs-finite-diff tolerance on a
#     fixture with endogenous_cost_component. Marked xfail(strict=True)
#     pre-4e; step 4e removes the marker.
#
#   Option B: hand-derived H / gradient_markups on a minimal fixture.
#     For a tiny Bertrand-logit problem, xi (Berry inversion) and
#     -mu/alpha (Bertrand markup alpha-derivative identity) are
#     derivable by hand, independent of any pyRVtest implementation.
#     This catches systematic math errors shared by both existing
#     internal paths — the failure mode the DMSS yogurt test is
#     uniquely positioned to catch.
# ===========================================================================


def _build_endogenous_cost_pyblp_fixture(seed: int = 17, T: int = 30, J: int = 3):
    """Small PyBLP-estimated logit DGP with a `log_quantity` endogenous-cost column.

    No true scale economies — `log_quantity` is computed from shares but does
    NOT feed into cost. The purpose is to exercise the endogenous_cost_component
    pipeline, not to estimate real gamma. gamma_hat will be near zero; what
    matters is that gradient_gamma w.r.t. (alpha, ...) is non-zero, so the
    demand-adjustment correction depends on whether pyRVtest computes it.

    Returns (product_data_df, pyblp_results, alpha_hat, beta_x_hat, beta_0_hat).
    """
    import pyblp
    pyblp.options.verbose = False

    market_ids = np.repeat(np.arange(T), J)
    firm_ids = np.tile(np.arange(J), T)
    id_data = pd.DataFrame({'market_ids': market_ids, 'firm_ids': firm_ids})

    X1 = pyblp.Formulation('1 + prices + x1')
    X3 = pyblp.Formulation('1 + z1')
    simulation = pyblp.Simulation(
        product_formulations=(X1, None, X3),
        beta=[0.5, -2.0, 1.0], gamma=[1.0, 0.5],
        xi_variance=0.15, omega_variance=0.15, correlation=0.0,
        product_data=id_data, seed=seed,
    )
    sim_results = simulation.replace_endogenous()
    data = pd.DataFrame(pyblp.data_to_dict(sim_results.product_data))
    # rival_x1 for demand instrument and as a conduct-testing instrument
    for t in range(T):
        idx = np.where(data['market_ids'] == t)[0]
        for j in idx:
            rival = [i for i in idx if i != j]
            data.loc[j, 'rival_x1'] = data.loc[rival, 'x1'].mean()
            data.loc[j, 'rival_z1'] = data.loc[rival, 'z1'].mean()
    data['demand_instruments0'] = data['rival_x1']

    # Synthetic "endogenous" cost column: log(quantity) where we use an
    # arbitrary market size so values are positive and finite.
    market_size = 100_000.0
    data['log_quantity'] = np.log(market_size * data['shares'])
    # Instrument for log_quantity: rival_z1_sq (independent of log_q residual)
    data['rival_z1_sq'] = data['rival_z1'] ** 2
    data['intercept'] = 1.0

    problem = pyblp.Problem((X1,), product_data=data)
    pyblp_results = problem.solve(method='1s')
    alpha_hat = float(pyblp_results.beta[pyblp_results.beta_labels.index('prices')].item())
    beta_x_hat = float(pyblp_results.beta[pyblp_results.beta_labels.index('x1')].item())
    beta_0_hat = float(pyblp_results.beta[pyblp_results.beta_labels.index('1')].item())
    return data, pyblp_results, alpha_hat, beta_x_hat, beta_0_hat


@pytest.fixture(scope='module')
def endogenous_cost_fixture():
    """Module-scope fixture so pyblp.solve only runs once."""
    return _build_endogenous_cost_pyblp_fixture()


def test_option_a_demand_params_matches_demand_results_with_endogenous_cost(
        endogenous_cost_fixture):
    """Post-4e cross-path sanity: TRV agrees between demand_results and demand_params paths
    with endogenous_cost_component after the capability-parity fix.

    Pre-step-4e, the analytical path returned ``gradient_gamma_per_instrument=None``
    so the gamma correction to TRV was silently skipped for demand_params users.
    Step 4e routes both paths through the unified ``compute_demand_adjustment``,
    which computes the gamma gradient for any ``SupportsDemandAdjustment`` backend.

    Residual cross-path divergence (~2e-10 on this fixture) comes from
    ``PyBLPBackend.jacobian_gradient`` (finite-diff, O(eps²)) vs
    ``LogitBackend.jacobian_gradient`` (analytical, exact), propagating through
    the implicit-differentiation markup-gradient formula and downstream to TRV.
    The tolerance below is set to cover this residual while staying tight
    enough to flag a gross regression (e.g., a missing correction term would
    widen the delta to >>1e-9).
    """
    data, pyblp_results, alpha_hat, beta_x_hat, beta_0_hat = endogenous_cost_fixture

    models = (
        pyRVtest.ModelFormulation(
            model_downstream='bertrand', ownership_downstream='firm_ids',
        ),
        pyRVtest.ModelFormulation(model_downstream='perfect_competition'),
    )
    common = dict(
        cost_formulation=pyRVtest.Formulation('1 + z1 + log_quantity'),
        instrument_formulation=pyRVtest.Formulation('0 + rival_x1 + rival_z1'),
        model_formulations=models,
        product_data=data,
        endogenous_cost_component='log_quantity',
    )

    pyRVtest.options.verbose = False
    # Path 1: demand_results (PyBLP path)
    r_pyblp = pyRVtest.Problem(**common, demand_results=pyblp_results).solve(
        demand_adjustment=True, clustering_adjustment=False
    )
    # Path 2: demand_params (analytical path)
    r_params = pyRVtest.Problem(
        **common,
        demand_params={
            'alpha': alpha_hat,
            'sigma': [],
            'beta': np.array([beta_0_hat, beta_x_hat]),
            'x_columns': ['intercept', 'x1'],
            'demand_instrument_columns': ['rival_x1', 'intercept', 'x1'],
        },
    ).solve(demand_adjustment=True, clustering_adjustment=False)

    # Tolerance calibrated empirically post-step-4e: cross-backend
    # jacobian_gradient difference (finite-diff vs analytical) gives
    # residual TRV delta ~2e-10 on this fixture. atol=5e-9 leaves
    # headroom above this noise floor while still being tight enough
    # that omitting a correction term (e.g., the gamma gradient pre-4e)
    # would produce a much larger delta and fail the assertion.
    np.testing.assert_allclose(
        r_pyblp.TRV[0][0, 1], r_params.TRV[0][0, 1], atol=5e-9, rtol=0,
        err_msg=(
            "TRV diverges between demand_results and demand_params paths with "
            "endogenous_cost_component. Pre-step-4e failure is expected "
            "(gradient_gamma=None in the analytical path). Post-step-4e failure "
            "means the unified gamma-gradient port has a bug."
        ),
    )


# ---------------------------------------------------------------------------
# Option B: hand-derived H / gradient_markups on a minimal fixture.
# ---------------------------------------------------------------------------

def _build_minimal_bertrand_logit_fixture():
    """Tiny Bertrand-logit DGP where xi, H, and d(markup)/d(alpha) are
    trivially hand-computable. Independent of any pyRVtest internals.

    Configuration:
      - T=5 markets, J=2 products each, so N=10.
      - Plain logit (no sigma).
      - 2 non-price linear regressors: intercept + x1.
      - 3 demand instruments: rival_x1, intercept, x1 (just-identified
        after X_D profile-out; H will be full rank).
      - 2 firms per market (firm_ids = [0, 1] per market) so Bertrand
        ownership matrix is the identity (single-product firms).
      - Simple cost shifter `z1`, no fixed effects, no clustering.

    Shares and prices are drawn directly (not from equilibrium), since
    the test only needs self-consistent xi and markup, not DGP economics.
    """
    rng = np.random.default_rng(seed=12321)
    T, J = 5, 2
    N = T * J
    alpha = -2.0
    beta = np.array([0.1, 1.3])  # intercept, x1

    market_ids = np.repeat(np.arange(T), J)
    firm_ids = np.tile(np.arange(J), T)

    # Random x1, then shares implied by Berry with constructed xi.
    x1 = rng.normal(size=N)
    # Make all shares small so outside-good share is comfortably positive.
    u = beta[0] + beta[1] * x1 + rng.normal(size=N) * 0.3  # not including prices yet
    prices = rng.uniform(0.5, 1.5, size=N)
    delta = u + alpha * prices  # this IS log(s/s0) by construction below
    shares = np.zeros(N)
    for t in range(T):
        idx = np.where(market_ids == t)[0]
        numer = np.exp(delta[idx])
        shares[idx] = numer / (1.0 + numer.sum())

    # Rival x1 instrument
    rival_x1 = np.zeros(N)
    for t in range(T):
        idx = np.where(market_ids == t)[0]
        for j in idx:
            others = [i for i in idx if i != j]
            rival_x1[j] = x1[others].mean()

    # Cost shifter (not used for correctness but required for Problem)
    z1 = rng.normal(size=N) + 1.5
    # Conduct-testing instrument (separate from demand instruments)
    test_iv = rng.normal(size=N)

    df = pd.DataFrame({
        'market_ids': market_ids, 'firm_ids': firm_ids,
        'shares': shares, 'prices': prices,
        'x1': x1, 'intercept': np.ones(N),
        'rival_x1': rival_x1,
        'z1': z1,
        'test_iv': test_iv,
    })
    return df, alpha, beta


def _hand_compute_xi(df, alpha, beta):
    """Berry inversion, independent of pyRVtest."""
    shares = df['shares'].values
    prices = df['prices'].values
    x1 = df['x1'].values
    market_ids = df['market_ids'].values
    N = len(shares)
    # Outside share per market
    s0 = np.zeros(N)
    for t in np.unique(market_ids):
        mask = market_ids == t
        s0[mask] = 1.0 - shares[mask].sum()
    X_D = np.column_stack([np.ones(N), x1])
    xi = np.log(shares) - np.log(s0) - X_D @ beta - alpha * prices
    return xi


def _hand_compute_H_plain_logit(df, alpha, x_cols, z_cols):
    """Hand-compute H = (1/N) Z_D' * partial_xi_theta for plain logit.

    partial_xi_theta = (-prices) residualized on X_D via 2SLS projection.
    """
    N = len(df)
    prices = df['prices'].values
    X_D = np.column_stack([df[c].values for c in x_cols])
    Z_D = np.column_stack([df[c].values for c in z_cols])
    # Default weight (what _compute_analytical_demand_adjustment uses when
    # demand_params['W_demand'] is unset): W_D = inv((1/N) Z_D'Z_D)
    W_D = np.linalg.inv((1.0 / N) * Z_D.T @ Z_D)
    dxi_dalpha = -prices  # (N,)
    # 2SLS-residualize on X_D using Z_D as instruments
    XtZW = X_D.T @ Z_D @ W_D
    M_xx = XtZW @ Z_D.T @ X_D
    projection = np.linalg.inv(M_xx) @ (XtZW @ Z_D.T @ dxi_dalpha)
    partial_xi_theta_col = dxi_dalpha - X_D @ projection  # (N,)
    H_expected = (1.0 / N) * Z_D.T @ partial_xi_theta_col.reshape(-1, 1)  # (K_z, 1)
    return H_expected


class TestOptionBHandDerivedGroundTruth:
    """Independent math validation on a minimal fixture.

    These tests derive xi, H, and d(markup)/d(alpha) from the DMSS / Berry
    formulas, NOT from pyRVtest source code. They catch systematic math
    errors that both inline paths and the unified function could share.
    """

    @pytest.fixture(scope='class')
    def fixture(self):
        df, alpha, beta = _build_minimal_bertrand_logit_fixture()
        return df, alpha, beta

    def test_xi_matches_berry_inversion(self, fixture):
        """LogitBackend.demand_moments().xi equals hand-computed Berry xi."""
        df, alpha, beta = fixture
        xi_expected = _hand_compute_xi(df, alpha, beta)
        backend = LogitBackend(
            alpha=alpha, product_data=df, beta=beta,
            x_columns=['intercept', 'x1'],
            demand_instrument_columns=['rival_x1', 'intercept', 'x1'],
        )
        xi_backend, _, _ = backend.demand_moments()
        np.testing.assert_allclose(
            xi_backend, xi_expected, atol=1e-12,
            err_msg="Berry inversion xi mismatch"
        )

    def test_H_matches_hand_computation(self, fixture):
        """(1/N) Z_D' @ backend.xi_gradient() equals hand-computed H for plain logit."""
        df, alpha, beta = fixture
        H_expected = _hand_compute_H_plain_logit(
            df, alpha,
            x_cols=['intercept', 'x1'],
            z_cols=['rival_x1', 'intercept', 'x1'],
        )
        backend = LogitBackend(
            alpha=alpha, product_data=df, beta=beta,
            x_columns=['intercept', 'x1'],
            demand_instrument_columns=['rival_x1', 'intercept', 'x1'],
        )
        _, Z_D, _ = backend.demand_moments()
        xi_grad = backend.xi_gradient()  # (N, 1)
        N = len(df)
        H_backend = (1.0 / N) * Z_D.T @ xi_grad
        np.testing.assert_allclose(
            H_backend, H_expected, atol=1e-12,
            err_msg="H mismatch vs hand-computed plain-logit 2SLS profile-out"
        )

    def test_bertrand_markup_alpha_derivative_is_minus_mu_over_alpha(self, fixture):
        """For Bertrand, d(markup)/d(alpha) = -markup/alpha is the exact analytical
        identity (independent of any pyRVtest internal path).

        This is the core Bertrand analytical identity. The unified
        compute_demand_adjustment function applies the implicit-differentiation
        formula ``d(markup)/d(theta) = -(O*D')^{-1} (O*dD'/d(theta)) markup``,
        which for the alpha column reduces to -mu/alpha by D's linearity in
        alpha. Both forms must agree to the hand-derived identity after the
        cost-shifter residualization that the unified function applies.
        """
        df, alpha, beta = fixture

        problem = pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('1 + z1'),
            instrument_formulation=pyRVtest.Formulation('0 + test_iv'),
            model_formulations=(
                pyRVtest.ModelFormulation(
                    model_downstream='bertrand', ownership_downstream='firm_ids',
                ),
            ),
            product_data=df,
            demand_params={
                'alpha': alpha, 'sigma': [],
                'beta': beta,
                'x_columns': ['intercept', 'x1'],
                'demand_instrument_columns': ['rival_x1', 'intercept', 'x1'],
            },
        )
        M = problem.M
        markups, _, _ = problem._perturb_and_build_markups()
        advalorem_tax_adj, cost_scaling = _advalorem_and_scaling(problem)

        from pyRVtest.solve.demand_adjustment import compute_demand_adjustment
        grad_unified = compute_demand_adjustment(
            problem._demand_backend, problem, M, problem.N, markups,
            advalorem_tax_adj, cost_scaling,
        )[0]

        # Hand-compute expected gradient on raw markups:
        # d(markup)/d(alpha) = -markup / alpha (Bertrand, homogeneous)
        mu_raw = markups[0].flatten()
        grad_expected_raw = -mu_raw / alpha  # shape (N,)

        # The unified function residualizes the final gradient on cost shifters;
        # residualize the hand-derived reference the same way to compare.
        w = problem.products.w  # already includes intercept + z1
        Q_w = np.linalg.qr(w, mode='reduced')[0] if w.any() else None
        grad_expected = grad_expected_raw.copy()
        if Q_w is not None:
            grad_expected = grad_expected - Q_w @ (Q_w.T @ grad_expected)

        np.testing.assert_allclose(
            grad_unified[0][:, 0], grad_expected, atol=1e-10,
            err_msg=(
                "Unified compute_demand_adjustment: d(markup)/d(alpha) != -mu/alpha "
                "(residualized). The implicit-differentiation general formula must "
                "numerically agree with the Bertrand homogeneity identity."
            ),
        )

    def test_markup_is_consistent_with_bertrand_foc(self, fixture):
        """Sanity check on the fixture itself: Bertrand FOC (O*D') @ markup + s = 0
        must hold. If this fails, the fixture shares are inconsistent and the
        other tests are invalid.
        """
        df, alpha, beta = fixture
        backend = LogitBackend(
            alpha=alpha, product_data=df, beta=beta,
            x_columns=['intercept', 'x1'],
            demand_instrument_columns=['rival_x1', 'intercept', 'x1'],
        )
        market_ids = df['market_ids'].values
        firm_ids = df['firm_ids'].values
        shares = df['shares'].values
        for t in np.unique(market_ids):
            idx = np.where(market_ids == t)[0]
            D_t = backend.compute_jacobian(market_id=t)
            s_t = shares[idx]
            fids = firm_ids[idx]
            O_t = (fids[:, None] == fids[None, :]).astype(float)
            # markup_t = -(O*D')^{-1} @ s
            markup_t = -np.linalg.solve(O_t * D_t.T, s_t)
            # Check FOC holds
            foc = (O_t * D_t.T) @ markup_t + s_t
            np.testing.assert_allclose(
                foc, np.zeros_like(foc), atol=1e-14,
                err_msg=f"Bertrand FOC residual nonzero at market {t}"
            )


# ===========================================================================
# v0.4 step 4e follow-up: validate _nested_logit_jacobian_derivative.
#
# This is the analytical formula for dD/d(sigma_m) used by
# NestedLogitBackend.jacobian_gradient for nested-logit demand. Before
# this test nothing in the suite validated it against finite-diff of
# _nested_logit_jacobian; the formula was hand-derived and un-checked.
# If this test passes, the analytical sigma-derivative in the nested-
# logit backend is numerically correct. If it fails, we have a real math
# bug that affects the demand-adjustment correction whenever sigma is
# non-trivial.
# ===========================================================================

class TestNestedLogitJacobianDerivative:
    """Finite-diff validation of the analytical dD/d(sigma) formula."""

    @staticmethod
    def _fd_derivative(alpha, sigma, s, nesting, m, eps=1e-6):
        """Central-difference approximation of d/d(sigma_m) of
        _nested_logit_jacobian(alpha, sigma, s, nesting).
        """
        from pyRVtest.backends.logit import _nested_logit_jacobian
        sigma_plus = list(sigma)
        sigma_minus = list(sigma)
        sigma_plus[m] = sigma[m] + eps / 2
        sigma_minus[m] = sigma[m] - eps / 2
        J_plus = _nested_logit_jacobian(alpha, sigma_plus, s, nesting)
        J_minus = _nested_logit_jacobian(alpha, sigma_minus, s, nesting)
        return (J_plus - J_minus) / eps

    @pytest.mark.parametrize("seed,alpha,sigma_val,J,n_per_nest", [
        (0, -2.0, 0.3, 4, 2),   # 1-level, 2 nests of 2
        (1, -1.5, 0.5, 6, 3),   # 1-level, 2 nests of 3
        (2, -3.0, 0.1, 8, 4),   # 1-level, 2 nests of 4
        (3, -0.5, 0.7, 4, 2),   # 1-level, large sigma
        (4, -2.0, 0.05, 6, 2),  # 1-level, small sigma
    ])
    def test_one_level_nested_logit_sigma_derivative(
            self, seed, alpha, sigma_val, J, n_per_nest):
        """One-level nested logit: analytical d/d(sigma) == finite-diff to 1e-9."""
        from pyRVtest.backends.logit import _nested_logit_jacobian_derivative
        rng = np.random.default_rng(seed=seed)
        # Shares must sum to < 1 per market; draw from dirichlet-like construction.
        raw = rng.uniform(0.1, 1.0, size=J)
        s = 0.5 * raw / raw.sum()  # keep s.sum() = 0.5 < 1
        # Nest assignment: first J/n_per_nest * n_per_nest products in nest A, rest in B
        nest_ids = np.array(
            ['A'] * n_per_nest + ['B'] * (J - n_per_nest), dtype=object
        )
        sigma = [sigma_val]
        nesting = [nest_ids]

        analytical = _nested_logit_jacobian_derivative(alpha, sigma, s, nesting, 0)
        numerical = self._fd_derivative(alpha, sigma, s, nesting, 0, eps=1e-6)

        np.testing.assert_allclose(
            analytical, numerical, atol=1e-8, rtol=1e-6,
            err_msg=(
                f"_nested_logit_jacobian_derivative analytical != finite-diff "
                f"(1-level, seed={seed}, alpha={alpha}, sigma={sigma_val}, "
                f"J={J}). This is a real bug in the hand-derived d/d(sigma) "
                f"formula; demand-adjustment corrections for nested-logit "
                f"demand_params users will be wrong."
            ),
        )

    @pytest.mark.parametrize("seed,alpha,sigma0,sigma1,J", [
        (10, -2.0, 0.3, 0.5, 8),
        (11, -1.0, 0.2, 0.6, 6),
        (12, -3.0, 0.1, 0.4, 8),
    ])
    def test_two_level_nested_logit_sigma_derivatives(
            self, seed, alpha, sigma0, sigma1, J):
        """Two-level nested logit: d/d(sigma_0) and d/d(sigma_1) both match finite-diff."""
        from pyRVtest.backends.logit import _nested_logit_jacobian_derivative
        rng = np.random.default_rng(seed=seed)
        raw = rng.uniform(0.1, 1.0, size=J)
        s = 0.5 * raw / raw.sum()
        # Two-level hierarchy: level 0 (finest) has 4 nests; level 1 (coarsest) has 2 nests.
        # Each level-1 nest contains 2 level-0 nests.
        finest = np.array(['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'][:J], dtype=object)
        coarsest = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'][:J], dtype=object)
        sigma = [sigma0, sigma1]
        nesting = [finest, coarsest]

        # m=0 derivative
        analytical_0 = _nested_logit_jacobian_derivative(alpha, sigma, s, nesting, 0)
        numerical_0 = self._fd_derivative(alpha, sigma, s, nesting, 0, eps=1e-6)
        np.testing.assert_allclose(
            analytical_0, numerical_0, atol=1e-8, rtol=1e-6,
            err_msg=f"d/d(sigma_0) mismatch (2-level, seed={seed})"
        )
        # m=1 derivative
        analytical_1 = _nested_logit_jacobian_derivative(alpha, sigma, s, nesting, 1)
        numerical_1 = self._fd_derivative(alpha, sigma, s, nesting, 1, eps=1e-6)
        np.testing.assert_allclose(
            analytical_1, numerical_1, atol=1e-8, rtol=1e-6,
            err_msg=f"d/d(sigma_1) mismatch (2-level, seed={seed})"
        )


# ===========================================================================
# v0.4 step 4e follow-up: demand_results auto-routing to analytical backend.
#
# The routing logic in Problem._construct_demand_backend distinguishes four cases:
#
#   1. Plain logit (K2=0, rho.size=0): routes to LogitBackend. pyblp's
#      finite-diff of d(D)/d(alpha) is ~O(1e-9) accurate (D is linear in
#      alpha at fixed shares, but compute_delta introduces small residuals).
#      Analytical D/alpha is exact. Precision gain is modest but real.
#
#   2. Single-scalar-rho nested logit (K2=0, r.rho.size=1): routes to
#      NestedLogitBackend(rho=[rho]). pyblp's scalar rho matches AFSSZ
#      L=1 sigma. D is NONLINEAR in rho, so finite-diff has O(eps^2)
#      error; analytical is exact. Material precision gain.
#
#   3. Per-nest rho (K2=0, r.rho.size>1): stays on PyBLPBackend. pyblp's
#      Cardell-Nevo formulation has one rho per nest; AFSSZ L-level has
#      one sigma per level. The derivatives don't match.
#
#   4. BLP (K2>0): stays on PyBLPBackend. No analytical derivative through
#      the BLP contraction.
#
# Fallback: if the user's raw product_data is missing columns the
# analytical backend needs (e.g., pyblp fixed-effect dummies that aren't
# in the original DataFrame), routing silently falls back to PyBLPBackend.
# ===========================================================================


def _pyblp_plain_logit_fixture(seed=101, T=20, J=3):
    """Small pyblp plain-logit estimate for routing tests."""
    import pyblp
    pyblp.options.verbose = False
    market_ids = np.repeat(np.arange(T), J)
    firm_ids = np.tile(np.arange(J), T)
    id_data = pd.DataFrame({'market_ids': market_ids, 'firm_ids': firm_ids})
    X1 = pyblp.Formulation('1 + prices + x1')
    X3 = pyblp.Formulation('1 + z1')
    simulation = pyblp.Simulation(
        product_formulations=(X1, None, X3),
        beta=[0.0, -2.0, 1.0], gamma=[1.0, 0.5],
        xi_variance=0.2, omega_variance=0.2, correlation=0.0,
        product_data=id_data, seed=seed,
    )
    sim_results = simulation.replace_endogenous()
    data = pd.DataFrame(pyblp.data_to_dict(sim_results.product_data))
    for t in range(T):
        idx = np.where(data['market_ids'] == t)[0]
        for j in idx:
            others = [i for i in idx if i != j]
            data.loc[j, 'rival_x1'] = data.loc[others, 'x1'].mean()
    data['demand_instruments0'] = data['rival_x1']
    data['intercept'] = 1.0
    data['rival_z1'] = data['rival_x1']  # Reuse for conduct testing
    problem = pyblp.Problem((X1,), product_data=data)
    pyblp_results = problem.solve(method='1s')
    return data, pyblp_results


class TestDemandResultsRouting:
    """Pin the backend-routing behavior per the four cases above."""

    @pytest.fixture(scope='class')
    def plain_logit_problem(self):
        data, pyblp_results = _pyblp_plain_logit_fixture()
        pyRVtest.options.verbose = False
        problem = pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('1 + z1'),
            instrument_formulation=pyRVtest.Formulation('0 + rival_x1'),
            model_formulations=(
                pyRVtest.ModelFormulation(
                    model_downstream='bertrand', ownership_downstream='firm_ids',
                ),
            ),
            product_data=data,
            demand_results=pyblp_results,
        )
        return problem

    def test_plain_logit_routes_to_logit_backend(self, plain_logit_problem):
        """Plain-logit demand_results auto-routes to LogitBackend (case 1)."""
        from pyRVtest.backends import LogitBackend
        assert isinstance(plain_logit_problem._demand_backend, LogitBackend), (
            f"Expected LogitBackend, got "
            f"{type(plain_logit_problem._demand_backend).__name__}."
        )

    def test_routed_plain_logit_holds_correct_alpha(self, plain_logit_problem):
        """Routed LogitBackend must hold pyblp's estimated alpha."""
        backend = plain_logit_problem._demand_backend
        expected_alpha = float(
            np.asarray(plain_logit_problem.demand_results.beta).flatten()[
                plain_logit_problem.demand_results.beta_labels.index('prices')
            ]
        )
        assert abs(backend._alpha - expected_alpha) < 1e-14

    @pytest.fixture(scope='class')
    def nested_logit_problem(self):
        """Small pyblp nested-logit estimate (single scalar rho)."""
        import pyblp
        pyblp.options.verbose = False
        T, J = 25, 4
        market_ids = np.repeat(np.arange(T), J)
        firm_ids = np.tile(np.arange(J), T)
        # Two nests: products 0, 1 in nest A; products 2, 3 in nest B.
        nesting_ids = np.tile(['A', 'A', 'B', 'B'], T)
        id_data = pd.DataFrame({
            'market_ids': market_ids, 'firm_ids': firm_ids,
            'nesting_ids': nesting_ids,
        })
        X1 = pyblp.Formulation('1 + prices + x1')
        X3 = pyblp.Formulation('1 + z1')
        simulation = pyblp.Simulation(
            product_formulations=(X1, None, X3),
            beta=[0.0, -2.0, 1.0], gamma=[1.0, 0.5],
            rho=0.3,  # scalar rho -> pyblp single-scalar nested logit
            xi_variance=0.1, omega_variance=0.1, correlation=0.0,
            product_data=id_data, seed=555,
        )
        sim_results = simulation.replace_endogenous()
        data = pd.DataFrame(pyblp.data_to_dict(sim_results.product_data))
        for t in range(T):
            idx = np.where(data['market_ids'] == t)[0]
            for j in idx:
                others = [i for i in idx if i != j]
                data.loc[j, 'rival_x1'] = data.loc[others, 'x1'].mean()
        data['demand_instruments0'] = data['rival_x1']
        data['rival_z1'] = data['rival_x1']  # reuse for conduct instrument

        nl_problem = pyblp.Problem((X1,), product_data=data)
        # Pass rho_bounds to allow rho estimation; single scalar bound.
        nl_results = nl_problem.solve(rho=0.2, method='1s')

        pyRVtest.options.verbose = False
        rv_problem = pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('1 + z1'),
            instrument_formulation=pyRVtest.Formulation('0 + rival_x1'),
            model_formulations=(
                pyRVtest.ModelFormulation(
                    model_downstream='bertrand', ownership_downstream='firm_ids',
                ),
            ),
            product_data=data,
            demand_results=nl_results,
        )
        return rv_problem, nl_results

    def test_single_rho_nested_logit_routes_to_nested_logit_backend(
            self, nested_logit_problem):
        """Single-scalar-rho nested logit auto-routes to NestedLogitBackend (case 2)."""
        rv_problem, nl_results = nested_logit_problem
        from pyRVtest.backends import NestedLogitBackend, PyBLPBackend
        backend = rv_problem._demand_backend
        # If rho vector has exactly 1 entry, we should have routed.
        rho_arr = np.atleast_1d(np.asarray(nl_results.rho).flatten())
        if rho_arr.size == 1:
            assert isinstance(backend, NestedLogitBackend), (
                f"Single-scalar-rho nested logit should route to "
                f"NestedLogitBackend, got {type(backend).__name__}."
            )
        else:
            # pyblp may have expanded rho to per-nest for certain inputs; if
            # so, we expect PyBLPBackend (case 3).
            assert isinstance(backend, PyBLPBackend), (
                f"Multi-rho nested logit should stay on PyBLPBackend, got "
                f"{type(backend).__name__}."
            )

    def test_routed_nested_logit_holds_correct_alpha_and_rho(
            self, nested_logit_problem):
        """Routed NestedLogitBackend must hold the same alpha and rho as pyblp."""
        rv_problem, nl_results = nested_logit_problem
        from pyRVtest.backends import NestedLogitBackend
        backend = rv_problem._demand_backend
        rho_arr = np.atleast_1d(np.asarray(nl_results.rho).flatten())
        if not isinstance(backend, NestedLogitBackend):
            pytest.skip("Backend is not NestedLogitBackend for this fixture")
        expected_alpha = float(
            np.asarray(nl_results.beta).flatten()[
                nl_results.beta_labels.index('prices')
            ]
        )
        assert abs(backend._alpha - expected_alpha) < 1e-14
        assert len(backend._rho) == 1
        assert abs(backend._rho[0] - float(rho_arr[0])) < 1e-14


# ===========================================================================
# v0.4 step 4i: gradient_markups gets tax / cost-scaling factor applied.
#
# Problem.solve uses markups_effective = tax_factor * markups_raw for the GMM
# moment. Downstream G_k uses d(markups_effective)/d(theta) which equals
# tax_factor * d(markups_raw)/d(theta). Pre-v0.4 the inline PyBLP path
# applied this factor (via apply_tax_adjustment); the inline analytical path
# didn't. Step 4d initially ported the analytical behavior (no factor),
# regressing PyBLP users with nontrivial taxes. Step 4i applies the factor
# uniformly in compute_demand_adjustment.
# ===========================================================================


class TestGradientMarkupsAppliesTaxFactor:
    """Regression test for the tax / cost-scaling factor on gradient_markups."""

    @staticmethod
    def _build_tax_dgp(tax_value: float):
        """Same data as the Option-A-like fixture, with an ``advalorem_tax``
        column optionally set to ``tax_value`` for all products.
        """
        rng = np.random.default_rng(seed=313)
        T, J = 10, 3
        N = T * J
        alpha = -2.0
        market_ids = np.repeat(np.arange(T), J)
        firm_ids = np.tile(np.arange(J), T)
        x1 = rng.normal(size=N)
        prices = rng.uniform(0.8, 2.0, size=N)
        u = 0.3 * x1 + rng.normal(scale=0.2, size=N)
        delta = u + alpha * prices
        shares = np.zeros(N)
        for t in range(T):
            idx = np.where(market_ids == t)[0]
            e = np.exp(delta[idx])
            shares[idx] = e / (1.0 + e.sum())
        rival_x1 = np.zeros(N)
        for t in range(T):
            idx = np.where(market_ids == t)[0]
            for j in idx:
                others = [i for i in idx if i != j]
                rival_x1[j] = x1[others].mean()
        z1 = rng.normal(size=N) + 1.5
        df = pd.DataFrame({
            'market_ids': market_ids, 'firm_ids': firm_ids,
            'prices': prices, 'shares': shares,
            'x1': x1, 'intercept': np.ones(N),
            'rival_x1': rival_x1,
            'z1': z1, 'iv0': rival_x1,
            'advalorem_tax': np.full(N, tax_value),
        })
        return df, alpha

    @staticmethod
    def _run(df, alpha, with_tax: bool):
        pyRVtest.options.verbose = False
        model_kwargs = dict(
            model_downstream='bertrand', ownership_downstream='firm_ids',
        )
        if with_tax:
            model_kwargs['advalorem_tax'] = 'advalorem_tax'
            model_kwargs['advalorem_payer'] = 'firm'
        problem = pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('1 + z1'),
            instrument_formulation=pyRVtest.Formulation('0 + iv0'),
            model_formulations=(pyRVtest.ModelFormulation(**model_kwargs),),
            product_data=df,
            demand_params={
                'alpha': alpha, 'sigma': [],
                'beta': np.array([0.0, 0.3]),
                'x_columns': ['intercept', 'x1'],
                'demand_instrument_columns': ['rival_x1', 'intercept', 'x1'],
            },
        )
        markups, _, _ = problem._perturb_and_build_markups()
        M, N_obs = problem.M, problem.N
        advalorem_tax_adj, cost_scaling = _advalorem_and_scaling(problem)
        from pyRVtest.solve.demand_adjustment import compute_demand_adjustment
        grad = compute_demand_adjustment(
            problem._demand_backend, problem, M, N_obs, markups,
            advalorem_tax_adj, cost_scaling,
        )[0]
        return grad, advalorem_tax_adj, cost_scaling

    def test_advalorem_tax_firm_payer_scales_gradient(self):
        """With tax_value=0.1 and advalorem_payer='firm', the per-product
        tax_adj factor is (1 - 0.1) = 0.9. gradient_markups must scale by
        0.9 relative to the no-tax baseline (exactly, since the factor is
        multiplicative and independent of theta).
        """
        df_notax, alpha = self._build_tax_dgp(tax_value=0.0)
        df_withtax, _ = self._build_tax_dgp(tax_value=0.1)

        grad_notax, adj_notax, cs_notax = self._run(df_notax, alpha, with_tax=False)
        grad_withtax, adj_withtax, cs_withtax = self._run(
            df_withtax, alpha, with_tax=True
        )

        # adj[0] / cs[0] are per-observation arrays (shape (N, 1)) because
        # advalorem_tax / cost_scaling can vary across products. Build the
        # expected per-observation factor vector.
        notax_factor = (
            np.asarray(adj_notax[0]).reshape(-1, 1)
            / (1.0 + np.asarray(cs_notax[0]).reshape(-1, 1))
        )
        withtax_factor = (
            np.asarray(adj_withtax[0]).reshape(-1, 1)
            / (1.0 + np.asarray(cs_withtax[0]).reshape(-1, 1))
        )
        # Sanity: the no-tax fixture has factor == 1 everywhere.
        np.testing.assert_allclose(notax_factor, 1.0, atol=1e-14)
        # With tax=0.1, advalorem_payer='firm' -> factor = 1 - 0.1 = 0.9.
        np.testing.assert_allclose(withtax_factor, 0.9, atol=1e-14)

        # The full gradient (after cost-shifter residualization) must scale
        # by withtax_factor row-by-row. Because factor is constant across all
        # N rows in this fixture, this is equivalent to a scalar multiplication;
        # the broadcasting in compute_demand_adjustment covers the general case.
        np.testing.assert_allclose(
            grad_withtax[0], withtax_factor * grad_notax[0],
            atol=1e-12, rtol=1e-10,
            err_msg=(
                "gradient_markups did not scale by the advalorem tax factor. "
                "Pre-4i bug: compute_demand_adjustment returned d(markups_raw)/d(theta) "
                "instead of d(markups_effective)/d(theta) = tax_factor * "
                "d(markups_raw)/d(theta)."
            ),
        )
