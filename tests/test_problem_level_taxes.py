"""Tests for Problem-level tax kwargs + per-model salience flags (v0.4 OQ 14).

Covers:
* Parity between Problem-level ``unit_tax`` and the legacy per-model
  ``unit_tax`` — identical TRV / F / markups on the same DGP.
* Parity for ``advalorem_tax`` and ``advalorem_payer``.
* Salience flags (``unit_tax_salient=False`` /
  ``advalorem_tax_salient=False``) make individual models opt out.
* Model-level ``unit_tax`` wins when both are set, with a
  ``DeprecationWarning`` that calls out the conflict.
* Input validation on ``Problem(unit_tax=...)`` and
  ``ConductModel(unit_tax_salient=...)``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import pyRVtest
from pyRVtest import Bertrand, Cournot
from pyRVtest.exceptions import ValidationError


# =====================================================================
# Minimal DGP fixture with a unit-tax and an ad-valorem-tax column.
# =====================================================================


def _make_dgp(seed: int = 1234, T: int = 10, J: int = 3, alpha: float = -1.5):
    """Plain-logit fixture with two tax columns for parity tests."""
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
    df = pd.DataFrame({
        'market_ids': market_ids,
        'firm_ids': firm_ids,
        'x1': x1,
        'prices': prices,
        'shares': shares,
        'rival_x1': rival_x1,
        'z1': z1,
        'intercept': np.ones(N),
        # Two tax columns:
        'tax_col': np.full(N, 0.05),
        'vat_col': np.full(N, 0.10),
    })
    return df, alpha


def _common_kwargs(df, alpha):
    return dict(
        cost_formulation=pyRVtest.Formulation('1 + z1'),
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
# Parity: Problem-level vs model-level taxes.
# =====================================================================


class TestUnitTaxParity:
    """Problem-level unit_tax produces identical results to legacy per-model unit_tax."""

    def test_marginal_cost_bit_identical(self, dgp):
        df, alpha = dgp
        common = _common_kwargs(df, alpha)
        pyRVtest.options.verbose = False

        # Legacy path: unit_tax on each model.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            r_legacy = pyRVtest.Problem(
                **common,
                models=[
                    Bertrand(ownership='firm_ids', unit_tax='tax_col'),
                    Cournot(ownership='firm_ids', unit_tax='tax_col'),
                ],
            ).solve(demand_adjustment=False, clustering_adjustment=False)

        # New path: Problem-level unit_tax, salience default True.
        r_new = pyRVtest.Problem(
            **common, unit_tax='tax_col',
            models=[
                Bertrand(ownership='firm_ids'),
                Cournot(ownership='firm_ids'),
            ],
        ).solve(demand_adjustment=False, clustering_adjustment=False)

        np.testing.assert_array_equal(r_legacy.marginal_cost, r_new.marginal_cost)
        np.testing.assert_array_equal(r_legacy.markups, r_new.markups)
        np.testing.assert_array_equal(
            np.asarray(r_legacy.TRV), np.asarray(r_new.TRV), strict=False,
        )
        np.testing.assert_array_equal(
            np.asarray(r_legacy.F), np.asarray(r_new.F), strict=False,
        )


class TestAdValoremTaxParity:
    """Parity on ad-valorem tax with payer='consumer'."""

    def test_marginal_cost_bit_identical(self, dgp):
        df, alpha = dgp
        common = _common_kwargs(df, alpha)
        pyRVtest.options.verbose = False

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            r_legacy = pyRVtest.Problem(
                **common,
                models=[
                    Bertrand(
                        ownership='firm_ids',
                        advalorem_tax='vat_col', advalorem_payer='consumer',
                    ),
                    Cournot(
                        ownership='firm_ids',
                        advalorem_tax='vat_col', advalorem_payer='consumer',
                    ),
                ],
            ).solve(demand_adjustment=False, clustering_adjustment=False)

        r_new = pyRVtest.Problem(
            **common,
            advalorem_tax='vat_col', advalorem_payer='consumer',
            models=[
                Bertrand(ownership='firm_ids'),
                Cournot(ownership='firm_ids'),
            ],
        ).solve(demand_adjustment=False, clustering_adjustment=False)

        np.testing.assert_array_equal(r_legacy.marginal_cost, r_new.marginal_cost)
        np.testing.assert_array_equal(r_legacy.markups, r_new.markups)


# =====================================================================
# Salience flag.
# =====================================================================


class TestSalienceFlag:
    """``unit_tax_salient=False`` makes a model opt out of Problem-level tax."""

    def test_salient_vs_nonsalient_markups_differ(self, dgp):
        df, alpha = dgp
        common = _common_kwargs(df, alpha)
        pyRVtest.options.verbose = False

        r = pyRVtest.Problem(
            **common, unit_tax='tax_col',
            models=[
                Bertrand(ownership='firm_ids'),  # salient (default)
                Bertrand(ownership='firm_ids', unit_tax_salient=False),
            ],
        ).solve(demand_adjustment=False, clustering_adjustment=False)

        # Same raw markups (tax does not enter Bertrand FOC); different
        # effective marginal cost because the non-salient model doesn't
        # subtract the tax.
        np.testing.assert_array_equal(r.markups[0], r.markups[1])
        assert not np.allclose(r.marginal_cost[0], r.marginal_cost[1])
        # Quantify: non-salient mc is salient mc + tax (because
        # salient path subtracts unit_tax, non-salient does not).
        tax = np.asarray(df['tax_col']).reshape(-1, 1)
        np.testing.assert_allclose(
            r.marginal_cost[1] - r.marginal_cost[0], tax, atol=1e-14,
        )

    def test_advalorem_salience(self, dgp):
        df, alpha = dgp
        common = _common_kwargs(df, alpha)
        pyRVtest.options.verbose = False

        r = pyRVtest.Problem(
            **common,
            advalorem_tax='vat_col', advalorem_payer='firm',
            models=[
                Bertrand(ownership='firm_ids'),
                Bertrand(ownership='firm_ids', advalorem_tax_salient=False),
            ],
        ).solve(demand_adjustment=False, clustering_adjustment=False)

        # Salient path applies (1 - vat) factor; non-salient does not.
        assert not np.allclose(r.marginal_cost[0], r.marginal_cost[1])

    def test_nonsalient_with_no_problem_tax_is_noop(self, dgp):
        """Flag alone is a no-op when no Problem-level tax is set."""
        df, alpha = dgp
        common = _common_kwargs(df, alpha)
        pyRVtest.options.verbose = False
        r_flag = pyRVtest.Problem(
            **common,
            models=[
                Bertrand(ownership='firm_ids', unit_tax_salient=False),
                Cournot(ownership='firm_ids'),
            ],
        ).solve(demand_adjustment=False, clustering_adjustment=False)
        r_plain = pyRVtest.Problem(
            **common,
            models=[Bertrand(ownership='firm_ids'), Cournot(ownership='firm_ids')],
        ).solve(demand_adjustment=False, clustering_adjustment=False)
        np.testing.assert_array_equal(r_flag.marginal_cost, r_plain.marginal_cost)

    def test_default_is_true(self):
        """Default ``unit_tax_salient`` is ``True``; explicit True matches."""
        m_default = Bertrand(ownership='firm_ids')
        m_explicit = Bertrand(ownership='firm_ids', unit_tax_salient=True)
        assert m_default.unit_tax_salient is True
        assert m_explicit.unit_tax_salient is True


# =====================================================================
# Precedence and DeprecationWarning.
# =====================================================================


class TestOverridePrecedence:
    """Model-level unit_tax wins over Problem-level, and emits DeprecationWarning."""

    def test_model_level_wins_and_warns(self, dgp):
        df, alpha = dgp
        # Same tax column set at both levels; model-level wins.
        common = _common_kwargs(df, alpha)
        # Make two distinct columns: 'tax_col' (0.05) at model level,
        # 'vat_col' (0.10) at problem level (re-purposed as a unit tax).
        pyRVtest.options.verbose = False
        # Reset the once-per-session warning-fired flag so this test
        # can observe the DeprecationWarning.
        from pyRVtest import problem as _prob
        _prob._legacy_tax_deprecation_warned.clear()
        with pytest.warns(DeprecationWarning, match='unit_tax'):
            pyRVtest.Problem(
                **common,
                unit_tax='vat_col',  # Problem level
                models=[
                    Bertrand(ownership='firm_ids', unit_tax='tax_col'),  # legacy wins
                    Cournot(ownership='firm_ids'),
                ],
            )

    def test_model_level_value_used_on_conflict(self, dgp):
        """Conflicting spec: model-level tax column is what enters mc."""
        df, alpha = dgp
        common = _common_kwargs(df, alpha)
        pyRVtest.options.verbose = False
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            r = pyRVtest.Problem(
                **common,
                unit_tax='vat_col',  # 0.10; Problem-level, should be ignored
                models=[
                    Bertrand(ownership='firm_ids', unit_tax='tax_col'),  # 0.05 wins
                    Cournot(ownership='firm_ids'),  # no model-level; Problem-level applies
                ],
            ).solve(demand_adjustment=False, clustering_adjustment=False)
        # Bertrand subtracts 0.05, Cournot subtracts 0.10 — so the
        # difference between their marginal costs should NOT include
        # the peers' raw markups, only the tax differential plus the
        # different FOC markups.
        # Easier assertion: the Models recarray should store the
        # model-level column name.
        assert r.problem.models['unit_tax_name'][0] == 'tax_col'
        assert r.problem.models['unit_tax_name'][1] == 'vat_col'

    def test_tiebreaker_policy_pin(self, dgp):
        """Tax-spec tiebreaking policy pin (Lorenzo review 2026-04-18).

        Pins the exact behavior when both Problem-level and per-model
        ``unit_tax`` / ``advalorem_tax`` are supplied:

        1. Per-model wins: the model-level column is what enters the
           effective price for that model.
        2. Both a generic per-model-tax ``DeprecationWarning`` AND a
           conflict-specific ``DeprecationWarning`` fire. The conflict
           warning names BOTH columns so the user can see the silent
           override in logs.
        3. Other models (no per-model tax) still inherit the
           Problem-level value.

        See :ref:`tax-precedence-tiebreaker` in
        ``docs/migrating_to_v0.4.rst`` for the user-facing statement.
        """
        df, alpha = dgp
        common = _common_kwargs(df, alpha)
        pyRVtest.options.verbose = False
        from pyRVtest import problem as _prob
        _prob._legacy_tax_deprecation_warned.clear()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            r = pyRVtest.Problem(
                **common,
                unit_tax='vat_col',  # Problem level: 0.10
                advalorem_tax='vat_col',  # Problem level for ad valorem too
                advalorem_payer='firm',
                models=[
                    Bertrand(
                        ownership='firm_ids',
                        unit_tax='tax_col',  # per-model: 0.05, should win
                        advalorem_tax='tax_col',  # per-model wins for ad-valorem too
                        advalorem_payer='consumer',
                    ),
                    Cournot(ownership='firm_ids'),  # no per-model tax; inherits Problem-level
                ],
            ).solve(demand_adjustment=False, clustering_adjustment=False)

        # (1) Per-model column wins for the first model.
        assert r.problem.models['unit_tax_name'][0] == 'tax_col', (
            "Per-model unit_tax must win on conflict."
        )
        assert r.problem.models['advalorem_tax_name'][0] == 'tax_col', (
            "Per-model advalorem_tax must win on conflict."
        )
        # The second model (no per-model tax) inherits the Problem-level value.
        assert r.problem.models['unit_tax_name'][1] == 'vat_col', (
            "Second model without per-model unit_tax inherits Problem-level."
        )
        assert r.problem.models['advalorem_tax_name'][1] == 'vat_col', (
            "Second model without per-model advalorem_tax inherits Problem-level."
        )

        # (2) Both a generic deprecation AND a conflict-specific
        # deprecation fire. The conflict warning names BOTH columns so
        # the user cannot silently miss the override.
        dep_msgs = [
            str(w.message) for w in caught
            if issubclass(w.category, DeprecationWarning)
        ]
        # Generic per-model-tax deprecation (pointing to migration doc).
        assert any('deprecated' in m for m in dep_msgs), (
            "Expected at least one generic deprecation message."
        )
        # Conflict-specific warning mentions the losing Problem-level
        # column ('vat_col') by name so the user sees what was ignored.
        conflict_msgs = [
            m for m in dep_msgs
            if 'Conflicting' in m and "'tax_col'" in m and "'vat_col'" in m
        ]
        assert conflict_msgs, (
            f"Expected a conflict-specific warning naming both columns. "
            f"Got:\n" + "\n---\n".join(dep_msgs)
        )


# =====================================================================
# No-op defaults.
# =====================================================================


class TestNoOpDefaults:
    """No tax specified anywhere: marginal cost is ``p - markup``."""

    def test_no_tax_anywhere(self, dgp):
        df, alpha = dgp
        common = _common_kwargs(df, alpha)
        pyRVtest.options.verbose = False
        r = pyRVtest.Problem(
            **common,
            models=[Bertrand(ownership='firm_ids'), Cournot(ownership='firm_ids')],
        ).solve(demand_adjustment=False, clustering_adjustment=False)
        expected = np.asarray(df['prices']).reshape(-1, 1) - r.markups[0]
        np.testing.assert_allclose(r.marginal_cost[0], expected, atol=1e-14)


# =====================================================================
# Validation errors.
# =====================================================================


class TestValidation:

    def test_problem_unit_tax_non_string_raises(self, dgp):
        df, alpha = dgp
        common = _common_kwargs(df, alpha)
        with pytest.raises(ValidationError, match='unit_tax'):
            pyRVtest.Problem(
                **common, unit_tax=123,  # type: ignore[arg-type]
                models=[Bertrand(ownership='firm_ids'), Cournot(ownership='firm_ids')],
            )

    def test_problem_advalorem_payer_invalid_raises(self, dgp):
        df, alpha = dgp
        common = _common_kwargs(df, alpha)
        with pytest.raises(ValidationError, match="advalorem_payer"):
            pyRVtest.Problem(
                **common, advalorem_tax='vat_col', advalorem_payer='banker',
                models=[Bertrand(ownership='firm_ids'), Cournot(ownership='firm_ids')],
            )

    def test_problem_advalorem_tax_without_payer_raises(self, dgp):
        df, alpha = dgp
        common = _common_kwargs(df, alpha)
        with pytest.raises(ValidationError, match="advalorem_payer"):
            pyRVtest.Problem(
                **common, advalorem_tax='vat_col',
                models=[Bertrand(ownership='firm_ids'), Cournot(ownership='firm_ids')],
            )

    def test_problem_advalorem_payer_without_tax_raises(self, dgp):
        df, alpha = dgp
        common = _common_kwargs(df, alpha)
        with pytest.raises(ValidationError, match="advalorem_tax"):
            pyRVtest.Problem(
                **common, advalorem_payer='firm',
                models=[Bertrand(ownership='firm_ids'), Cournot(ownership='firm_ids')],
            )

    def test_problem_unit_tax_missing_column_raises(self, dgp):
        df, alpha = dgp
        common = _common_kwargs(df, alpha)
        with pytest.raises(ValidationError, match="missing_col"):
            pyRVtest.Problem(
                **common, unit_tax='missing_col',
                models=[Bertrand(ownership='firm_ids'), Cournot(ownership='firm_ids')],
            )

    def test_salience_flag_non_bool_raises(self):
        with pytest.raises(TypeError, match='unit_tax_salient'):
            Bertrand(ownership='firm_ids', unit_tax_salient='yes')  # type: ignore[arg-type]

    def test_advalorem_salience_flag_non_bool_raises(self):
        with pytest.raises(TypeError, match='advalorem_tax_salient'):
            Bertrand(
                ownership='firm_ids',
                advalorem_tax_salient='true',  # type: ignore[arg-type]
            )

    def test_salience_flag_on_vertical(self):
        from pyRVtest import Monopoly, Vertical
        with pytest.raises(TypeError, match='unit_tax_salient'):
            Vertical(
                downstream=Bertrand(ownership='firm_ids'),
                upstream=Monopoly(ownership='firm_ids'),
                unit_tax_salient='no',  # type: ignore[arg-type]
            )


# =====================================================================
# Interaction with legacy ModelFormulation.
# =====================================================================


class TestLegacyModelFormulationWithSalience:
    """Legacy ``ModelFormulation`` accepts the salience flags too."""

    def test_model_formulation_carries_flags(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            mf = pyRVtest.ModelFormulation(
                model_downstream='bertrand',
                ownership_downstream='firm_ids',
                unit_tax_salient=False,
            )
        assert mf._unit_tax_salient is False
        assert mf._advalorem_tax_salient is True
