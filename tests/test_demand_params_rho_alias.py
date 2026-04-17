"""v0.4 step 6b: demand_params['rho'] canonical; demand_params['sigma'] is a
deprecated alias.

Pre-v0.4 the nested-logit parameter in ``demand_params`` was named
``sigma``, matching pyRVtest's AFSSZ L-level internal convention.
``pyblp`` calls it ``rho`` for the Cardell-Nevo formulation. Step 6b
canonicalizes the user-facing name as ``rho`` (aligned with pyblp) while
keeping ``sigma`` accepted for back-compat with a once-per-session
``DeprecationWarning``. Internal backend classes (NestedLogitBackend)
keep their ``sigma=...`` kwarg unchanged.

Mutual exclusion: supplying both ``rho`` and ``sigma`` raises TypeError.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import pyRVtest
import pyRVtest.problem as pyrv_problem


@pytest.fixture(autouse=True)
def _reset_deprecation_flag():
    """Reset the once-per-session flag so each test sees a fresh state."""
    saved = pyrv_problem._demand_params_sigma_deprecation_warned
    pyrv_problem._demand_params_sigma_deprecation_warned = False
    yield
    pyrv_problem._demand_params_sigma_deprecation_warned = saved


def _minimal_df():
    rng = np.random.default_rng(seed=42)
    T, J = 6, 3
    N = T * J
    market_ids = np.repeat(np.arange(T), J)
    firm_ids = np.tile(np.arange(J), T)
    x1 = rng.normal(size=N)
    prices = rng.uniform(0.8, 2.0, size=N)
    alpha = -1.5
    u = 0.4 * x1 + rng.normal(scale=0.2, size=N)
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
    return pd.DataFrame({
        'market_ids': market_ids, 'firm_ids': firm_ids,
        'x1': x1, 'intercept': np.ones(N),
        'prices': prices, 'shares': shares,
        'rival_x1': rival_x1, 'z1': z1,
    }), alpha


class TestRhoNormalization:
    """Unit-level tests for the normalization helper itself."""

    def test_rho_canonical_passes_through(self):
        demand_params = {'alpha': -1.5, 'rho': [0.3]}
        out = pyrv_problem._normalize_demand_params_rho(demand_params)
        assert out['rho'] == [0.3]
        assert out['sigma'] == [0.3]  # mirrored for back-compat downstream

    def test_sigma_translated_to_rho_with_warning(self):
        demand_params = {'alpha': -1.5, 'sigma': [0.3]}
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter('always')
            out = pyrv_problem._normalize_demand_params_rho(demand_params)
        assert out['rho'] == [0.3]
        assert out['sigma'] == [0.3]
        dep_warnings = [
            w for w in recorded if issubclass(w.category, DeprecationWarning)
        ]
        assert len(dep_warnings) == 1
        assert "demand_params['sigma'] is deprecated" in str(dep_warnings[0].message)

    def test_sigma_warning_fires_only_once_per_session(self):
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter('always')
            pyrv_problem._normalize_demand_params_rho(
                {'alpha': -1.5, 'sigma': [0.3]},
            )
            pyrv_problem._normalize_demand_params_rho(
                {'alpha': -1.5, 'sigma': [0.4]},
            )
        dep_warnings = [
            w for w in recorded if issubclass(w.category, DeprecationWarning)
        ]
        assert len(dep_warnings) == 1, (
            f"Expected exactly one DeprecationWarning across two calls, "
            f"got {len(dep_warnings)}."
        )

    def test_both_keys_present_raises(self):
        with pytest.raises(TypeError, match="cannot contain both 'rho' and 'sigma'"):
            pyrv_problem._normalize_demand_params_rho(
                {'alpha': -1.5, 'rho': [0.3], 'sigma': [0.4]},
            )

    def test_does_not_mutate_caller_dict(self):
        original = {'alpha': -1.5, 'rho': [0.3]}
        snapshot = dict(original)
        pyrv_problem._normalize_demand_params_rho(original)
        assert original == snapshot

    def test_empty_list_is_handled(self):
        """Plain logit (no nesting): both 'rho': [] and 'sigma': [] work."""
        out_rho = pyrv_problem._normalize_demand_params_rho(
            {'alpha': -1.5, 'rho': []},
        )
        assert out_rho['sigma'] == []
        out_sigma = pyrv_problem._normalize_demand_params_rho(
            {'alpha': -1.5, 'sigma': []},
        )
        assert out_sigma['rho'] == []


class TestRhoIntegration:
    """Integration: Problem(demand_params={'rho': ...}) produces same output as
    Problem(demand_params={'sigma': ...}) and both work through solve().
    """

    def test_rho_and_sigma_equivalent_in_problem(self):
        df, alpha = _minimal_df()
        common = dict(
            cost_formulation=pyRVtest.Formulation('1 + z1'),
            instrument_formulation=pyRVtest.Formulation('0 + rival_x1'),
            product_data=df,
            models=[
                pyRVtest.Bertrand(ownership='firm_ids'),
                pyRVtest.PerfectCompetition(),
            ],
        )
        pyRVtest.options.verbose = False

        r_rho = pyRVtest.Problem(
            **common,
            demand_params={
                'alpha': alpha, 'rho': [],
                'beta': np.array([0.0, 0.4]),
                'x_columns': ['intercept', 'x1'],
                'demand_instrument_columns': ['rival_x1', 'intercept', 'x1'],
            },
        ).solve(demand_adjustment=False)
        r_sigma = pyRVtest.Problem(
            **common,
            demand_params={
                'alpha': alpha, 'sigma': [],
                'beta': np.array([0.0, 0.4]),
                'x_columns': ['intercept', 'x1'],
                'demand_instrument_columns': ['rival_x1', 'intercept', 'x1'],
            },
        ).solve(demand_adjustment=False)

        for m in range(len(r_rho.markups)):
            np.testing.assert_allclose(
                r_rho.markups[m], r_sigma.markups[m], atol=1e-14,
                err_msg=f"rho vs sigma produce different markups[{m}]",
            )

    def test_both_keys_in_problem_raises(self):
        df, alpha = _minimal_df()
        pyRVtest.options.verbose = False
        with pytest.raises(TypeError, match="cannot contain both"):
            pyRVtest.Problem(
                cost_formulation=pyRVtest.Formulation('1 + z1'),
                instrument_formulation=pyRVtest.Formulation('0 + rival_x1'),
                product_data=df,
                models=[
                    pyRVtest.Bertrand(ownership='firm_ids'),
                    pyRVtest.PerfectCompetition(),
                ],
                demand_params={
                    'alpha': alpha, 'rho': [0.3], 'sigma': [0.3],
                    'beta': np.array([0.0, 0.4]),
                    'x_columns': ['intercept', 'x1'],
                    'demand_instrument_columns': [
                        'rival_x1', 'intercept', 'x1',
                    ],
                },
            )
