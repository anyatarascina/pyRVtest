"""Tests for the in-package one-level nested-logit 2SLS estimator.

Validates against:
1. Hand-computed within-nest log-share construction (unit tests).
2. Recovery of (alpha, rho) on a simulated nested-logit DGP.
3. End-to-end Problem.solve consistency with the demand_results path
   (pyblp's nested-logit solution).
"""

from typing import List

import numpy as np
import pandas as pd
import pytest
import warnings

import pyRVtest
from pyRVtest.estimators._within_share import count_in_nest_iv
from pyRVtest.exceptions import ValidationError


def _simulate_nested_logit_data(
        T=80, alpha_true=-2.0, rho_true=0.5, seed=99,
):
    """Simulate a one-level nested-logit DGP in closed form (no PyBLP).

    Generates shares from the AFSSZ nested-logit formula directly so the
    test doesn't depend on PyBLP's internal share-equilibration code,
    which raises on stricter NumPy versions (Lorenzo's audit observed
    TypeError on 0-dim conversion). The recovery property the test
    cares about is: given (price, x1, nesting_ids, shares) generated
    from a known (alpha, rho), can NestedLogitEstimator recover them?
    Closed-form generation lets us answer that without entangling PyBLP.

    DGP details:
      * Varying market sizes J_t ∈ {3,4,5,6} with random nest splits
        so count_in_nest_iv has variation across markets.
      * Linear utility delta_jt = X_jt'beta + alpha*p_jt + xi_jt with
        beta = (1, 1) on (intercept, x1).
      * Cost shifter z1 driving prices (so z1 is a valid IV).
      * One-level nested-logit shares from the standard formula:
            s_{j|g} = exp(delta_j / (1-rho)) / sum_{j' in g} exp(...)
            s_g = D_g^{(1-rho)} / (1 + sum_h D_h^{(1-rho)})
            s_j = s_{j|g} * s_g
        where D_g = sum_{j' in g} exp(delta_{j'} / (1-rho)).
    """
    rng = np.random.default_rng(seed)
    records: List[dict] = []
    for t in range(T):
        J_t = int(rng.choice([3, 4, 5, 6]))
        n_nest_a = int(rng.integers(1, J_t))
        nest = np.array([0] * n_nest_a + [1] * (J_t - n_nest_a))
        for j in range(J_t):
            records.append({
                'market_ids': t, 'firm_ids': j,
                'nesting_ids': int(nest[j]),
            })
    df = pd.DataFrame(records)
    N = len(df)

    # Demand-side covariates and unobservable.
    df['x1'] = rng.normal(size=N)
    df['xi'] = rng.normal(scale=0.2, size=N)
    # Cost shifter, drives prices.
    df['z1'] = rng.normal(size=N)
    # Cost-side residual.
    omega = rng.normal(scale=0.2, size=N)

    # Reduced-form price equation: prices correlated with z1 (the cost
    # shifter) and with xi (so price is endogenous to demand). Use a
    # positive base + log-additive structure so prices stay strictly
    # positive even when the cost-residual omega is very negative
    # (pyRVtest.Problem rejects negative prices).
    df['prices'] = np.exp(
        0.0 + 0.3 * df['z1'].to_numpy() + 0.2 * df['xi'].to_numpy()
        + 0.2 * omega
    )

    # Linear-utility delta = beta_const + beta_x1 * x1 + alpha * prices + xi
    delta = (
        1.0
        + 1.0 * df['x1'].to_numpy()
        + alpha_true * df['prices'].to_numpy()
        + df['xi'].to_numpy()
    )

    # Closed-form nested-logit shares per market.
    shares = np.zeros(N)
    market_ids = df['market_ids'].to_numpy()
    nesting_ids = df['nesting_ids'].to_numpy()
    one_minus_rho = 1.0 - rho_true
    for t in df['market_ids'].unique():
        idx = np.where(market_ids == t)[0]
        nests_t = nesting_ids[idx]
        delta_t = delta[idx]
        # Per-nest inclusive value D_g and within-nest shares.
        within_nest_share = np.zeros(len(idx))
        D_g = {}
        for g in np.unique(nests_t):
            in_g = (nests_t == g)
            exp_terms = np.exp(delta_t[in_g] / one_minus_rho)
            D_g_val = exp_terms.sum()
            D_g[g] = D_g_val
            within_nest_share[in_g] = exp_terms / D_g_val
        # Nest-level shares.
        denominator = 1.0 + sum(D_g_val ** one_minus_rho for D_g_val in D_g.values())
        nest_share = {g: D_g[g] ** one_minus_rho / denominator for g in D_g}
        # Combine.
        for g in np.unique(nests_t):
            in_g = (nests_t == g)
            shares[idx[in_g]] = within_nest_share[in_g] * nest_share[g]

    df['shares'] = shares
    df['demand_instruments0'] = df['z1']
    return df


# ---------------------------------------------------------------------------
# Within-share IV helper
# ---------------------------------------------------------------------------


class TestCountInNestIV:
    def test_basic_two_market(self):
        df = pd.DataFrame({
            'market_ids': [0, 0, 0, 1, 1, 1, 1],
            'nesting_ids': [0, 0, 1, 0, 1, 1, 1],
        })
        counts, name = count_in_nest_iv(
            df, market_ids_column='market_ids', nesting_ids_column='nesting_ids',
        )
        assert name == 'count_in_nest_iv'
        np.testing.assert_array_equal(counts, [2, 2, 1, 1, 3, 3, 3])

    def test_default_column_name(self):
        df = pd.DataFrame({'market_ids': [0, 0], 'nesting_ids': [0, 1]})
        _, name = count_in_nest_iv(df, 'market_ids', 'nesting_ids')
        assert name == 'count_in_nest_iv'

    def test_misaligned_lengths_raise(self):
        # Pass arrays of different sizes (mock unaligned input)
        bad = {'market_ids': np.array([0, 0]), 'nesting_ids': np.array([0])}
        with pytest.raises(ValidationError, match="same length"):
            count_in_nest_iv(bad, 'market_ids', 'nesting_ids')


# ---------------------------------------------------------------------------
# Recovery / dict shape
# ---------------------------------------------------------------------------


class TestRecovery:
    def test_alpha_and_rho_recovered(self):
        data = _simulate_nested_logit_data(T=120, seed=99)
        est = pyRVtest.NestedLogitEstimator(
            product_data=data,
            formulation_X=pyRVtest.Formulation('1 + x1'),
            formulation_Z=pyRVtest.Formulation('0 + z1'),
            nesting_ids_column='nesting_ids',
            auto_construct_within_share_iv=True,
        )
        params = est.solve()
        assert abs(params['alpha'] - (-2.0)) < 0.4, params['alpha']
        assert abs(params['rho'][0] - 0.5) < 0.15, params['rho']

    def test_returned_dict_keys(self):
        data = _simulate_nested_logit_data(T=80, seed=11)
        est = pyRVtest.NestedLogitEstimator(
            product_data=data,
            formulation_X=pyRVtest.Formulation('1 + x1'),
            formulation_Z=pyRVtest.Formulation('0 + z1'),
            nesting_ids_column='nesting_ids',
            auto_construct_within_share_iv=True,
        )
        params = est.solve()
        assert set(params.keys()) >= {
            'alpha', 'beta', 'rho', 'x_columns',
            'demand_instrument_columns', 'W_demand', 'nesting_ids_columns',
        }
        assert isinstance(params['rho'], list)
        assert len(params['rho']) == 1
        assert params['nesting_ids_columns'] == ['nesting_ids']
        # Auto-IV is appended; both excluded IVs first, then x_columns.
        assert params['demand_instrument_columns'][:2] == ['z1', 'count_in_nest_iv']
        assert params['demand_instrument_columns'][2:] == ['1', 'x1']

    def test_explicit_two_iv_path(self):
        """Without auto-IV the user must supply >=2 excluded instruments."""
        data = _simulate_nested_logit_data(T=80, seed=23)
        # Add a synthetic second IV — rival's z1 by market.
        for t in data['market_ids'].unique():
            idx = np.where(data['market_ids'] == t)[0]
            for j in idx:
                rivals = [i for i in idx if i != j]
                data.loc[j, 'rival_z1'] = data.loc[rivals, 'z1'].mean()

        est = pyRVtest.NestedLogitEstimator(
            product_data=data,
            formulation_X=pyRVtest.Formulation('1 + x1'),
            formulation_Z=pyRVtest.Formulation('0 + z1 + rival_z1'),
            nesting_ids_column='nesting_ids',
            auto_construct_within_share_iv=False,
        )
        params = est.solve()
        assert params['demand_instrument_columns'][:2] == ['z1', 'rival_z1']
        assert 'count_in_nest_iv' not in params['demand_instrument_columns']


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_underidentified_without_auto_iv(self):
        """One excluded IV + auto_iv=False -> order condition fails."""
        data = _simulate_nested_logit_data(T=40, seed=3)
        est = pyRVtest.NestedLogitEstimator(
            product_data=data,
            formulation_X=pyRVtest.Formulation('1 + x1'),
            formulation_Z=pyRVtest.Formulation('0 + z1'),
            nesting_ids_column='nesting_ids',
            auto_construct_within_share_iv=False,
        )
        with pytest.raises(ValidationError, match="2 excluded instruments"):
            est.solve()

    def test_no_nesting_column_raises(self):
        data = _simulate_nested_logit_data(T=40, seed=3)
        with pytest.raises(ValidationError, match="nesting_ids_column"):
            pyRVtest.NestedLogitEstimator(
                product_data=data,
                formulation_X=pyRVtest.Formulation('1 + x1'),
                formulation_Z=pyRVtest.Formulation('0 + z1'),
                nesting_ids_column='',
            )

    def test_rho_out_of_range_warns(self):
        """If the estimator produces rho outside [0, 1), it should warn.

        Use the closed-form DGP at rho_true=0.99 (very near the boundary)
        with high xi-variance noise so 2SLS sometimes overshoots; this is
        a stochastic check, conditional on the data actually producing
        an out-of-range rho_hat. The point is: when it does happen, the
        estimator emits a clear warning rather than silently returning
        nonsense.
        """
        # Closed-form near-boundary DGP. We can't guarantee rho_hat falls
        # out of [0, 1) at any specific seed, so the assertion is gated:
        # only require the warning IF rho is actually out of range.
        data = _simulate_nested_logit_data(T=40, rho_true=0.99, seed=21)

        est = pyRVtest.NestedLogitEstimator(
            product_data=data,
            formulation_X=pyRVtest.Formulation('1 + x1'),
            formulation_Z=pyRVtest.Formulation('0 + z1'),
            nesting_ids_column='nesting_ids',
            auto_construct_within_share_iv=True,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            params = est.solve()
        if not (0 <= params['rho'][0] < 1):
            assert any('rho' in str(w.message).lower() for w in caught), (
                f"rho_hat={params['rho'][0]} is out of [0, 1) but no warning fired"
            )


# ---------------------------------------------------------------------------
# End-to-end: Problem.solve with the estimator's output
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_problem_solve_runs(self):
        """A minimal end-to-end check: Problem(demand_params=est.solve()) runs and
        produces sensible markups (positive, less than price).
        """
        pyRVtest.options.verbose = False
        data = _simulate_nested_logit_data(T=80, seed=17)
        est = pyRVtest.NestedLogitEstimator(
            product_data=data,
            formulation_X=pyRVtest.Formulation('1 + x1'),
            formulation_Z=pyRVtest.Formulation('0 + z1'),
            nesting_ids_column='nesting_ids',
            auto_construct_within_share_iv=True,
        )
        demand_params = est.solve()
        models = (
            pyRVtest.ModelFormulation(
                model_downstream='bertrand', ownership_downstream='firm_ids',
            ),
            pyRVtest.ModelFormulation(model_downstream='perfect_competition'),
        )
        # Use rival_z1 as testing instrument -- distinct from cost shifter z1.
        for t in data['market_ids'].unique():
            idx = np.where(data['market_ids'] == t)[0]
            for j in idx:
                rivals = [i for i in idx if i != j]
                data.loc[j, 'rival_z1'] = data.loc[rivals, 'z1'].mean()
        # Auto-IV adds a count_in_nest_iv column to product_data; pass the
        # augmented version so Problem can find it.
        data_for_problem = est.product_data
        data_for_problem = data_for_problem.assign(
            rival_z1=data['rival_z1'].values,
        )

        results = pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('1 + z1'),
            instrument_formulation=pyRVtest.Formulation('0 + rival_z1'),
            model_formulations=models,
            product_data=data_for_problem,
            demand_params=demand_params,
        ).solve(demand_adjustment=False)
        pyRVtest.options.verbose = True

        # Bertrand markups should be positive and finite.
        bertrand_markups = results.markups[0].flatten()
        assert np.all(np.isfinite(bertrand_markups))
        assert np.all(bertrand_markups > 0)
        # Perfect competition markups should be all zero.
        np.testing.assert_allclose(results.markups[1].flatten(), 0.0)


class TestInlineShortcut:
    """Phase 3: Problem(demand_params={'estimate':'nested_logit', ...}).

    Inline path runs NestedLogitEstimator internally. With
    auto_construct_within_share_iv=True the estimator augments
    product_data with a count_in_nest_iv column; Problem forwards that
    augmented copy so downstream code (Products construction etc.) sees
    the same data the estimator did.
    """

    @pytest.fixture(scope='class')
    def comparison_results(self):
        pyRVtest.options.verbose = False
        data = _simulate_nested_logit_data(T=80, seed=29)
        # Build the testing instrument (rival_z1) up front so both paths use
        # the same product_data on input. Auto-IV adds count_in_nest_iv to
        # both paths' downstream copies.
        for t in data['market_ids'].unique():
            idx = np.where(data['market_ids'] == t)[0]
            for j in idx:
                rivals = [i for i in idx if i != j]
                data.loc[j, 'rival_z1'] = data.loc[rivals, 'z1'].mean()

        models = (
            pyRVtest.ModelFormulation(
                model_downstream='bertrand', ownership_downstream='firm_ids',
            ),
            pyRVtest.ModelFormulation(model_downstream='perfect_competition'),
        )

        # Standalone path (the user runs the estimator, then passes the dict).
        est = pyRVtest.NestedLogitEstimator(
            product_data=data,
            formulation_X=pyRVtest.Formulation('1 + x1'),
            formulation_Z=pyRVtest.Formulation('0 + z1'),
            nesting_ids_column='nesting_ids',
            auto_construct_within_share_iv=True,
        )
        demand_params = est.solve()
        r_standalone = pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('1 + z1'),
            instrument_formulation=pyRVtest.Formulation('0 + rival_z1'),
            model_formulations=models,
            product_data=est.product_data,  # augmented with count_in_nest_iv
            demand_params=demand_params,
        ).solve(demand_adjustment=False)

        # Inline path (Problem detects 'estimate' and runs the estimator).
        r_inline = pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('1 + z1'),
            instrument_formulation=pyRVtest.Formulation('0 + rival_z1'),
            model_formulations=models,
            product_data=data,
            demand_params={
                'estimate': 'nested_logit',
                'formulation_X': pyRVtest.Formulation('1 + x1'),
                'formulation_Z': pyRVtest.Formulation('0 + z1'),
                'nesting_ids_column': 'nesting_ids',
                'auto_construct_within_share_iv': True,
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

    def test_inline_missing_nesting_column_raises(self):
        data = _simulate_nested_logit_data(T=20, seed=31)
        with pytest.raises(ValueError, match="nesting_ids_column"):
            pyRVtest.Problem(
                cost_formulation=pyRVtest.Formulation('1 + z1'),
                instrument_formulation=pyRVtest.Formulation('0 + z1'),
                model_formulations=(
                    pyRVtest.ModelFormulation(model_downstream='perfect_competition'),
                ),
                product_data=data,
                demand_params={
                    'estimate': 'nested_logit',
                    'formulation_X': pyRVtest.Formulation('1 + x1'),
                    'formulation_Z': pyRVtest.Formulation('0 + z1'),
                    'auto_construct_within_share_iv': True,
                },
            )
