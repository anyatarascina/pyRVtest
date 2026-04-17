"""Tests for Formulation(known_coefficients=...) cost shifters (v0.4 OQ 14).

Covers:
* Known-coefficient shifters subtracted uniformly from prices_effective.
* Parity with Problem-level ``unit_tax`` when the coefficient is 1.0.
* Multiple shifters stack linearly.
* Validation on ``Formulation(known_coefficients=...)`` at construction
  time (bad type, non-numeric, NaN / inf, column-name already in the
  formula).
* Column-in-product_data check deferred to ``Problem.__init__``.
* Integration with Problem-level taxes in the same solve call.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyRVtest
from pyRVtest import Bertrand, Cournot, Formulation
from pyRVtest.exceptions import ValidationError


# =====================================================================
# Validation at Formulation construction time (no product_data involved).
# =====================================================================


class TestFormulationValidation:
    """``known_coefficients`` is type-checked at Formulation construction."""

    def test_empty_default(self):
        f = Formulation('0 + w')
        assert f.known_coefficients == {}

    def test_dict_stored(self):
        f = Formulation('0 + w', known_coefficients={'a': 0.5, 'b': 1.25})
        assert f.known_coefficients == {'a': 0.5, 'b': 1.25}

    def test_int_coefficient_coerced_to_float(self):
        f = Formulation('0 + w', known_coefficients={'a': 1})
        assert f.known_coefficients == {'a': 1.0}
        assert isinstance(f.known_coefficients['a'], float)

    def test_non_dict_raises(self):
        with pytest.raises(ValidationError, match='dict'):
            Formulation('0 + w', known_coefficients=[('a', 0.5)])  # type: ignore[arg-type]

    def test_non_string_key_raises(self):
        with pytest.raises(ValidationError, match='non-empty column-name'):
            Formulation(
                '0 + w',
                known_coefficients={123: 0.5},  # type: ignore[dict-item]
            )

    def test_empty_string_key_raises(self):
        with pytest.raises(ValidationError, match='non-empty column-name'):
            Formulation('0 + w', known_coefficients={'': 0.5})

    def test_bool_coefficient_raises(self):
        with pytest.raises(ValidationError, match='numeric scalar'):
            Formulation(
                '0 + w',
                known_coefficients={'a': True},  # type: ignore[dict-item]
            )

    def test_string_coefficient_raises(self):
        with pytest.raises(ValidationError, match='numeric scalar'):
            Formulation(
                '0 + w',
                known_coefficients={'a': '0.5'},  # type: ignore[dict-item]
            )

    def test_nan_raises(self):
        with pytest.raises(ValidationError, match='finite'):
            Formulation('0 + w', known_coefficients={'a': float('nan')})

    def test_inf_raises(self):
        with pytest.raises(ValidationError, match='finite'):
            Formulation('0 + w', known_coefficients={'a': float('inf')})

    def test_overlap_with_formula_raises(self):
        with pytest.raises(ValidationError, match='NOT already'):
            Formulation('0 + w + input_price', known_coefficients={'input_price': 0.5})


# =====================================================================
# DGP fixture and end-to-end tests.
# =====================================================================


def _make_dgp(seed: int = 1234, T: int = 10, J: int = 3, alpha: float = -1.5):
    rng = np.random.default_rng(seed=seed)
    N = T * J
    market_ids = np.repeat(np.arange(T), J)
    firm_ids = np.tile(np.arange(J), T)
    x1 = rng.normal(size=N)
    prices = rng.uniform(0.8, 2.0, size=N)
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
    # Known-coefficient shifters: input_price and union_wage.
    input_price = rng.uniform(0.1, 0.3, size=N)
    union_wage = rng.uniform(0.2, 0.4, size=N)
    df = pd.DataFrame({
        'market_ids': market_ids,
        'firm_ids': firm_ids,
        'x1': x1,
        'prices': prices,
        'shares': shares,
        'rival_x1': rival_x1,
        'z1': z1,
        'intercept': np.ones(N),
        'input_price': input_price,
        'union_wage': union_wage,
        'tax_col': np.full(N, 0.05),
    })
    return df, alpha


def _kwargs_without_cost_formulation(df, alpha):
    return dict(
        instrument_formulation=pyRVtest.Formulation('0 + rival_x1'),
        product_data=df,
        demand_params={
            'alpha': alpha, 'sigma': [],
            'beta': np.array([0.0, 0.4]),
            'x_columns': ['intercept', 'x1'],
            'demand_instrument_columns': ['rival_x1', 'intercept', 'x1'],
        },
    )


@pytest.fixture(scope='module')
def dgp():
    return _make_dgp()


# =====================================================================
# End-to-end semantics.
# =====================================================================


class TestKnownCoefficientsApplied:
    """Known-coefficient shifters subtract from effective prices."""

    def test_single_shifter_shifts_marginal_cost(self, dgp):
        df, alpha = dgp
        pyRVtest.options.verbose = False
        gamma = 0.75

        # Baseline: no known-coefficient shifters.
        r_base = pyRVtest.Problem(
            cost_formulation=Formulation('1 + z1'),
            **_kwargs_without_cost_formulation(df, alpha),
            models=[Bertrand(ownership='firm_ids'), Cournot(ownership='firm_ids')],
        ).solve(demand_adjustment=False, clustering_adjustment=False)

        # With known-coef shifter.
        r_shift = pyRVtest.Problem(
            cost_formulation=Formulation(
                '1 + z1',
                known_coefficients={'input_price': gamma},
            ),
            **_kwargs_without_cost_formulation(df, alpha),
            models=[Bertrand(ownership='firm_ids'), Cournot(ownership='firm_ids')],
        ).solve(demand_adjustment=False, clustering_adjustment=False)

        # Markups are unchanged (depend only on demand). Marginal cost
        # shifts down by gamma * input_price.
        np.testing.assert_array_equal(r_base.markups, r_shift.markups)
        expected = (
            r_base.marginal_cost - gamma * np.asarray(df['input_price']).reshape(-1, 1)
        )
        np.testing.assert_allclose(r_shift.marginal_cost, expected, atol=1e-14)


class TestParityWithUnitTax:
    """``known_coefficients={'col': 1.0}`` equals Problem-level ``unit_tax='col'``."""

    def test_coefficient_one_matches_unit_tax(self, dgp):
        df, alpha = dgp
        pyRVtest.options.verbose = False

        # Unit-tax path.
        r_tax = pyRVtest.Problem(
            cost_formulation=Formulation('1 + z1'),
            **_kwargs_without_cost_formulation(df, alpha),
            unit_tax='tax_col',
            models=[Bertrand(ownership='firm_ids'), Cournot(ownership='firm_ids')],
        ).solve(demand_adjustment=False, clustering_adjustment=False)

        # Known-coefficient path with gamma=1.0 on the same column.
        r_kc = pyRVtest.Problem(
            cost_formulation=Formulation(
                '1 + z1', known_coefficients={'tax_col': 1.0},
            ),
            **_kwargs_without_cost_formulation(df, alpha),
            models=[Bertrand(ownership='firm_ids'), Cournot(ownership='firm_ids')],
        ).solve(demand_adjustment=False, clustering_adjustment=False)

        np.testing.assert_allclose(r_tax.marginal_cost, r_kc.marginal_cost, atol=1e-14)
        np.testing.assert_allclose(r_tax.markups, r_kc.markups, atol=1e-14)


class TestMultipleShifters:
    """Multiple known-coefficient shifters stack linearly."""

    def test_two_shifters_stack(self, dgp):
        df, alpha = dgp
        pyRVtest.options.verbose = False
        gamma1, gamma2 = 0.5, 1.25

        r_base = pyRVtest.Problem(
            cost_formulation=Formulation('1 + z1'),
            **_kwargs_without_cost_formulation(df, alpha),
            models=[Bertrand(ownership='firm_ids'), Cournot(ownership='firm_ids')],
        ).solve(demand_adjustment=False, clustering_adjustment=False)

        r_both = pyRVtest.Problem(
            cost_formulation=Formulation(
                '1 + z1',
                known_coefficients={
                    'input_price': gamma1,
                    'union_wage': gamma2,
                },
            ),
            **_kwargs_without_cost_formulation(df, alpha),
            models=[Bertrand(ownership='firm_ids'), Cournot(ownership='firm_ids')],
        ).solve(demand_adjustment=False, clustering_adjustment=False)

        shift = (
            gamma1 * np.asarray(df['input_price']).reshape(-1, 1) +
            gamma2 * np.asarray(df['union_wage']).reshape(-1, 1)
        )
        expected = r_base.marginal_cost - shift
        np.testing.assert_allclose(r_both.marginal_cost, expected, atol=1e-14)


class TestColumnNotInProductData:
    """Missing column raises at ``Problem.__init__``, not at Formulation."""

    def test_missing_column_raises_at_problem(self, dgp):
        df, alpha = dgp
        pyRVtest.options.verbose = False
        f = Formulation('1 + z1', known_coefficients={'nonexistent': 0.5})
        with pytest.raises(ValidationError, match='nonexistent'):
            pyRVtest.Problem(
                cost_formulation=f,
                **_kwargs_without_cost_formulation(df, alpha),
                models=[Bertrand(ownership='firm_ids'), Cournot(ownership='firm_ids')],
            )


class TestIntegrationWithProblemLevelTaxes:
    """Known coefficients + Problem-level ``unit_tax`` compose additively."""

    def test_both_subtract_from_effective_price(self, dgp):
        df, alpha = dgp
        pyRVtest.options.verbose = False
        gamma = 0.3

        r_combo = pyRVtest.Problem(
            cost_formulation=Formulation(
                '1 + z1', known_coefficients={'input_price': gamma},
            ),
            **_kwargs_without_cost_formulation(df, alpha),
            unit_tax='tax_col',
            models=[Bertrand(ownership='firm_ids'), Cournot(ownership='firm_ids')],
        ).solve(demand_adjustment=False, clustering_adjustment=False)

        # Compare against sum of two separate effects (no-tax, no-kc
        # baseline + shift from tax + shift from kc).
        r_base = pyRVtest.Problem(
            cost_formulation=Formulation('1 + z1'),
            **_kwargs_without_cost_formulation(df, alpha),
            models=[Bertrand(ownership='firm_ids'), Cournot(ownership='firm_ids')],
        ).solve(demand_adjustment=False, clustering_adjustment=False)

        expected = (
            r_base.marginal_cost -
            np.asarray(df['tax_col']).reshape(-1, 1) -
            gamma * np.asarray(df['input_price']).reshape(-1, 1)
        )
        np.testing.assert_allclose(r_combo.marginal_cost, expected, atol=1e-14)
