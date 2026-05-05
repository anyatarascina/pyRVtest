"""Tests for the in-package plain-logit 2SLS estimator.

Validates LogitEstimator against:
1. Hand-computed Berry-inversion / 2SLS values (unit tests).
2. PyBLP's logit estimates on the same data (cross-check).
3. End-to-end Problem.solve() consistency vs. the demand_results path.
"""

import numpy as np
import pandas as pd
import pytest
import warnings

import pyRVtest
from pyRVtest.estimators._base import compute_delta, two_stage_least_squares
from pyRVtest.exceptions import ValidationError


def _simulate_logit_data(T=30, J=3, alpha_true=-2.0, seed=99):
    """Simulate a plain-logit DGP using pyblp (matches conventions of the
    existing test_demand_params suite).

    The cost shifter ``z1`` doubles as the demand instrument for price
    (via the synthesized ``demand_instruments0`` column, which pyblp's
    Problem reads automatically). ``rival_x1`` is reserved for the
    pyRVtest testing-instrument role.
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
        beta=[0, alpha_true, 1],
        gamma=[1, 0.5],
        xi_variance=0.2,
        omega_variance=0.2,
        correlation=0.0,
        product_data=id_data,
        seed=seed,
    )
    sim_results = simulation.replace_endogenous()
    data = pd.DataFrame(pyblp.data_to_dict(sim_results.product_data))

    # Build a rival-x1 testing instrument and use z1 as the (excluded) demand
    # instrument so pyblp's Problem(...) is identified for alpha.
    for t in range(T):
        idx = np.where(data['market_ids'] == t)[0]
        for j in idx:
            rivals = [i for i in idx if i != j]
            data.loc[j, 'rival_x1'] = data.loc[rivals, 'x1'].mean()
    data['demand_instruments0'] = data['z1']

    return data


# ---------------------------------------------------------------------------
# Unit tests: 2SLS / Berry inversion correctness
# ---------------------------------------------------------------------------


class TestComputeDelta:
    def test_simple_two_market(self):
        shares = np.array([0.4, 0.4, 0.3, 0.3])
        market_ids = np.array([0, 0, 1, 1])
        delta = compute_delta(shares, market_ids)
        # s0_0 = 0.2, s0_1 = 0.4
        np.testing.assert_allclose(delta[0], np.log(0.4) - np.log(0.2))
        np.testing.assert_allclose(delta[2], np.log(0.3) - np.log(0.4))

    def test_zero_share_raises(self):
        shares = np.array([0.5, 0.0, 0.3])
        market_ids = np.array([0, 0, 1])
        with pytest.raises(ValidationError, match="strictly positive"):
            compute_delta(shares, market_ids)

    def test_full_inside_market_raises(self):
        shares = np.array([0.6, 0.4, 0.3])
        market_ids = np.array([0, 0, 1])
        with pytest.raises(ValidationError, match="sum to >= 1"):
            compute_delta(shares, market_ids)


class TestTwoStageLeastSquares:
    def test_exactly_identified_recovers_ols(self):
        """When K_w == K_d (just-identified) and W is the design matrix,
        2SLS reduces to OLS.
        """
        rng = np.random.default_rng(0)
        N = 200
        X = rng.standard_normal((N, 2))
        beta_true = np.array([1.5, -0.7])
        y = X @ beta_true + rng.standard_normal(N) * 0.1
        # Use W = X (just-identified, exogenous regressors only)
        theta, _ = two_stage_least_squares(y, X, X)
        np.testing.assert_allclose(theta, beta_true, atol=0.05)

    def test_underidentified_raises(self):
        rng = np.random.default_rng(0)
        D = rng.standard_normal((50, 3))
        W = rng.standard_normal((50, 2))
        with pytest.raises(ValidationError, match="order condition"):
            two_stage_least_squares(np.zeros(50), D, W)

    def test_singular_W_raises(self):
        rng = np.random.default_rng(0)
        N = 100
        X = rng.standard_normal((N, 2))
        # Make W rank-deficient
        W = np.column_stack([X[:, 0], X[:, 0], X[:, 1]])
        with pytest.raises(ValidationError, match="full column rank"):
            two_stage_least_squares(np.zeros(N), X, W)


# ---------------------------------------------------------------------------
# Recovery test on synthetic DGP
# ---------------------------------------------------------------------------


class TestRecovery:
    def test_alpha_recovered_within_mc_noise(self):
        """LogitEstimator should recover alpha within ~0.4 on a moderate-N DGP."""
        data = _simulate_logit_data(T=50, J=3, alpha_true=-2.0, seed=42)
        est = pyRVtest.LogitEstimator(
            product_data=data,
            formulation_X=pyRVtest.Formulation('1 + x1'),
            formulation_Z=pyRVtest.Formulation('0 + z1'),
        )
        params = est.solve()
        assert params['alpha'] < 0
        assert abs(params['alpha'] - (-2.0)) < 0.4, (
            f"alpha={params['alpha']:.4f}, true=-2.0"
        )

    def test_returned_dict_has_expected_keys(self):
        data = _simulate_logit_data(T=30, J=3)
        est = pyRVtest.LogitEstimator(
            product_data=data,
            formulation_X=pyRVtest.Formulation('1 + x1'),
            formulation_Z=pyRVtest.Formulation('0 + z1'),
        )
        params = est.solve()
        assert set(params.keys()) >= {
            'alpha', 'beta', 'rho', 'x_columns',
            'demand_instrument_columns', 'W_demand',
        }
        assert isinstance(params['alpha'], float)
        assert isinstance(params['beta'], np.ndarray)
        assert params['rho'] == []
        assert params['x_columns'] == ['1', 'x1']
        # excluded IVs first, then x columns
        assert params['demand_instrument_columns'][:1] == ['z1']
        assert set(params['demand_instrument_columns']) == {'1', 'x1', 'z1'}
        assert params['W_demand'].shape == (3, 3)

    def test_no_intercept_in_X(self):
        """Estimator should work with formulation_X='0 + ...' (no intercept)."""
        data = _simulate_logit_data(T=30, J=3)
        est = pyRVtest.LogitEstimator(
            product_data=data,
            formulation_X=pyRVtest.Formulation('0 + x1'),
            formulation_Z=pyRVtest.Formulation('0 + z1'),
        )
        params = est.solve()
        assert '1' not in params['x_columns']
        assert params['alpha'] < 0


# ---------------------------------------------------------------------------
# Cross-check vs. PyBLP on the same data
# ---------------------------------------------------------------------------


class TestVsPyBLP:
    @pytest.fixture(scope='class')
    def comparison(self):
        import pyblp
        pyblp.options.verbose = False

        data = _simulate_logit_data(T=40, J=3, alpha_true=-2.0, seed=7)

        # PyBLP plain-logit estimate. The simulation helper sets
        # demand_instruments0 = z1 so pyblp's Problem is identified.
        pyblp_problem = pyblp.Problem(
            (pyblp.Formulation('1 + prices + x1'),), product_data=data,
        )
        pyblp_results = pyblp_problem.solve(method='1s')
        alpha_pyblp = float(
            pyblp_results.beta[pyblp_results.beta_labels.index('prices')].item()
        )

        # In-package estimator on the same data + same excluded IV. PyBLP's
        # ZD for plain logit is [demand_instruments0, 1, x1]; our
        # demand_instrument_columns end up as [z1, 1, x1]. Same up to a
        # column-rename (z1 == demand_instruments0), so alpha must match.
        est = pyRVtest.LogitEstimator(
            product_data=data,
            formulation_X=pyRVtest.Formulation('1 + x1'),
            formulation_Z=pyRVtest.Formulation('0 + z1'),
        )
        params = est.solve()
        return alpha_pyblp, params

    def test_alpha_matches_pyblp(self, comparison):
        alpha_pyblp, params = comparison
        # Same data, same instruments, same 2SLS weight ((Z'Z/N)^-1) -> same alpha.
        np.testing.assert_allclose(params['alpha'], alpha_pyblp, atol=1e-8)


# ---------------------------------------------------------------------------
# End-to-end: Problem.solve() with the estimator's output
# ---------------------------------------------------------------------------


class TestEndToEnd:
    @pytest.fixture(scope='class')
    def end_to_end_results(self):
        import pyblp
        pyblp.options.verbose = False
        pyRVtest.options.verbose = False

        data = _simulate_logit_data(T=40, J=3, alpha_true=-2.0, seed=11)

        models = (
            pyRVtest.ModelFormulation(
                model_downstream='bertrand', ownership_downstream='firm_ids',
            ),
            pyRVtest.ModelFormulation(model_downstream='perfect_competition'),
        )
        iv_form = 'rival_x1'

        # Path A: pyblp results (the existing canonical path).
        pyblp_results = pyblp.Problem(
            (pyblp.Formulation('1 + prices + x1'),), product_data=data,
        ).solve(method='1s')
        r_pyblp = pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('1 + z1'),
            instrument_formulation=pyRVtest.Formulation('0 + ' + iv_form),
            model_formulations=models,
            product_data=data,
            demand_results=pyblp_results,
        ).solve(demand_adjustment=False)

        # Path B: in-package LogitEstimator -> demand_params dict.
        demand_iv_cols = [c for c in data.columns if c.startswith('demand_instruments')]
        est = pyRVtest.LogitEstimator(
            product_data=data,
            formulation_X=pyRVtest.Formulation('1 + x1'),
            formulation_Z=pyRVtest.Formulation('0 + ' + ' + '.join(demand_iv_cols)),
        )
        demand_params = est.solve()
        r_est = pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('1 + z1'),
            instrument_formulation=pyRVtest.Formulation('0 + ' + iv_form),
            model_formulations=models,
            product_data=data,
            demand_params=demand_params,
        ).solve(demand_adjustment=False)

        pyRVtest.options.verbose = True
        return r_pyblp, r_est

    def test_markups_match(self, end_to_end_results):
        r_pyblp, r_est = end_to_end_results
        for m in range(2):
            np.testing.assert_allclose(
                r_pyblp.markups[m].flatten(), r_est.markups[m].flatten(),
                atol=1e-8,
            )

    def test_trv_matches(self, end_to_end_results):
        r_pyblp, r_est = end_to_end_results
        np.testing.assert_allclose(
            r_pyblp.TRV[0][0, 1], r_est.TRV[0][0, 1], atol=1e-6,
        )

    def test_f_matches(self, end_to_end_results):
        r_pyblp, r_est = end_to_end_results
        np.testing.assert_allclose(
            r_pyblp.F[0][0, 1], r_est.F[0][0, 1], atol=1e-6,
        )


class TestInlineShortcut:
    """Phase 3: Problem(demand_params={'estimate': 'logit', ...}) inline path.

    The inline path runs LogitEstimator internally and replaces
    demand_params with its output before constructing the demand backend.
    Result-by-result equivalence with the standalone estimator path is
    the contract.
    """

    @pytest.fixture(scope='class')
    def comparison_results(self):
        pyRVtest.options.verbose = False
        data = _simulate_logit_data(T=40, J=3, alpha_true=-2.0, seed=53)
        models = (
            pyRVtest.ModelFormulation(
                model_downstream='bertrand', ownership_downstream='firm_ids',
            ),
            pyRVtest.ModelFormulation(model_downstream='perfect_competition'),
        )

        # Standalone path
        est = pyRVtest.LogitEstimator(
            product_data=data,
            formulation_X=pyRVtest.Formulation('1 + x1'),
            formulation_Z=pyRVtest.Formulation('0 + z1'),
        )
        demand_params = est.solve()
        r_standalone = pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('1 + z1'),
            instrument_formulation=pyRVtest.Formulation('0 + rival_x1'),
            model_formulations=models,
            product_data=data,
            demand_params=demand_params,
        ).solve(demand_adjustment=False)

        # Inline path
        r_inline = pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('1 + z1'),
            instrument_formulation=pyRVtest.Formulation('0 + rival_x1'),
            model_formulations=models,
            product_data=data,
            demand_params={
                'estimate': 'logit',
                'formulation_X': pyRVtest.Formulation('1 + x1'),
                'formulation_Z': pyRVtest.Formulation('0 + z1'),
            },
        ).solve(demand_adjustment=False)
        pyRVtest.options.verbose = True
        return r_standalone, r_inline

    def test_markups_identical(self, comparison_results):
        r_a, r_b = comparison_results
        for m in range(2):
            np.testing.assert_allclose(
                r_a.markups[m].flatten(), r_b.markups[m].flatten(), atol=1e-12,
            )

    def test_trv_identical(self, comparison_results):
        r_a, r_b = comparison_results
        np.testing.assert_allclose(
            r_a.TRV[0][0, 1], r_b.TRV[0][0, 1], atol=1e-12,
        )

    def test_f_identical(self, comparison_results):
        r_a, r_b = comparison_results
        np.testing.assert_allclose(
            r_a.F[0][0, 1], r_b.F[0][0, 1], atol=1e-12,
        )


# ---------------------------------------------------------------------------
# Validation paths
# ---------------------------------------------------------------------------


class TestValidation:
    def test_missing_prices_column(self):
        data = pd.DataFrame({
            'market_ids': [0, 0, 1, 1],
            'shares': [0.3, 0.3, 0.2, 0.2],
            'x1': [1.0, 2.0, 3.0, 4.0],
            'z1': [0.5, 1.5, 2.5, 3.5],
        })
        est = pyRVtest.LogitEstimator(
            product_data=data,
            formulation_X=pyRVtest.Formulation('0 + x1'),
            formulation_Z=pyRVtest.Formulation('0 + z1'),
        )
        with pytest.raises(ValidationError, match="prices"):
            est.solve()

    def test_no_excluded_iv_raises(self):
        with pytest.raises(ValidationError, match="formulation_Z"):
            pyRVtest.LogitEstimator(
                product_data=pd.DataFrame(
                    {'market_ids': [0], 'shares': [0.5], 'prices': [1.0]}
                ),
                formulation_X=None,
                formulation_Z=None,
            )

    def test_prices_in_formulation_X_rejected(self):
        data = _simulate_logit_data(T=20, J=2)
        est = pyRVtest.LogitEstimator(
            product_data=data,
            formulation_X=pyRVtest.Formulation('1 + prices + x1'),
            formulation_Z=pyRVtest.Formulation('0 + z1 + rival_x1'),
        )
        with pytest.raises(ValidationError, match="exclude the 'prices' column"):
            est.solve()

    def test_inline_unknown_kind_raises(self):
        data = _simulate_logit_data(T=20, J=2)
        with pytest.raises(ValueError, match="logit.*nested_logit"):
            pyRVtest.Problem(
                cost_formulation=pyRVtest.Formulation('1 + z1'),
                instrument_formulation=pyRVtest.Formulation('0 + rival_x1'),
                model_formulations=(
                    pyRVtest.ModelFormulation(model_downstream='perfect_competition'),
                ),
                product_data=data,
                demand_params={
                    'estimate': 'random_coefficients',
                    'formulation_X': pyRVtest.Formulation('1 + x1'),
                    'formulation_Z': pyRVtest.Formulation('0 + z1'),
                },
            )

    def test_inline_with_prefilled_alpha_raises(self):
        data = _simulate_logit_data(T=20, J=2)
        with pytest.raises(ValueError, match="fresh estimator"):
            pyRVtest.Problem(
                cost_formulation=pyRVtest.Formulation('1 + z1'),
                instrument_formulation=pyRVtest.Formulation('0 + rival_x1'),
                model_formulations=(
                    pyRVtest.ModelFormulation(model_downstream='perfect_competition'),
                ),
                product_data=data,
                demand_params={
                    'estimate': 'logit',
                    'alpha': -2.0,
                    'formulation_X': pyRVtest.Formulation('1 + x1'),
                    'formulation_Z': pyRVtest.Formulation('0 + z1'),
                },
            )

    def test_inline_unknown_config_key_raises(self):
        data = _simulate_logit_data(T=20, J=2)
        with pytest.raises(ValueError, match="estimator-config keys"):
            pyRVtest.Problem(
                cost_formulation=pyRVtest.Formulation('1 + z1'),
                instrument_formulation=pyRVtest.Formulation('0 + rival_x1'),
                model_formulations=(
                    pyRVtest.ModelFormulation(model_downstream='perfect_competition'),
                ),
                product_data=data,
                demand_params={
                    'estimate': 'logit',
                    'formulation_X': pyRVtest.Formulation('1 + x1'),
                    'formulation_Z': pyRVtest.Formulation('0 + z1'),
                    'wat': 1,
                },
            )

    def test_inline_logit_with_nesting_kwarg_raises(self):
        data = _simulate_logit_data(T=20, J=2)
        with pytest.raises(ValueError, match="nesting_ids_column"):
            pyRVtest.Problem(
                cost_formulation=pyRVtest.Formulation('1 + z1'),
                instrument_formulation=pyRVtest.Formulation('0 + rival_x1'),
                model_formulations=(
                    pyRVtest.ModelFormulation(model_downstream='perfect_competition'),
                ),
                product_data=data,
                demand_params={
                    'estimate': 'logit',
                    'formulation_X': pyRVtest.Formulation('1 + x1'),
                    'formulation_Z': pyRVtest.Formulation('0 + z1'),
                    'nesting_ids_column': 'x1',
                },
            )

    def test_positive_alpha_warns(self):
        """If price's instrument is mis-signed, alpha can come out positive — warn."""
        rng = np.random.default_rng(3)
        N_T, J = 25, 3
        N = N_T * J
        market_ids = np.repeat(np.arange(N_T), J)
        # Construct an upward-sloping demand DGP so the estimator hits alpha > 0.
        prices = rng.uniform(0.5, 1.5, size=N)
        x1 = rng.standard_normal(N)
        z1 = prices + rng.standard_normal(N) * 0.05
        delta = -1.0 + 1.5 * prices + 0.5 * x1 + rng.standard_normal(N) * 0.1
        # Convert delta -> shares via logit form
        shares = np.zeros(N)
        for t in np.unique(market_ids):
            idx = np.where(market_ids == t)[0]
            exp_delta = np.exp(delta[idx])
            shares[idx] = exp_delta / (1 + exp_delta.sum())
        data = pd.DataFrame({
            'market_ids': market_ids,
            'shares': shares,
            'prices': prices,
            'x1': x1,
            'z1': z1,
        })
        est = pyRVtest.LogitEstimator(
            product_data=data,
            formulation_X=pyRVtest.Formulation('1 + x1'),
            formulation_Z=pyRVtest.Formulation('0 + z1'),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            params = est.solve()
        assert params['alpha'] > 0
        assert any('alpha' in str(w.message) and '>= 0' in str(w.message)
                   for w in caught), (
            f"Expected a 'alpha >= 0' warning; got {[str(w.message) for w in caught]}"
        )
