"""Tests for Dearing et al. (2024) simple-markup conduct models.

Covers :class:`pyRVtest.RuleOfThumb` and :class:`pyRVtest.ConstantMarkup`,
shipped in v0.4 step 12.

Source: Dearing, Magnolfi, Quint, Sullivan, and Waldfogel (2024),
"Learning Firm Conduct: Pass-Through as a Foundation for Instrument
Relevance," Example 1 (RuleOfThumb, pp. 7-8) and Example 7
(ConstantMarkup, pp. 23-24).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyRVtest
from pyRVtest import (
    Bertrand,
    ConstantMarkup,
    PerfectCompetition,
    RuleOfThumb,
)
from pyRVtest.exceptions import ValidationError


# =====================================================================
# Validation
# =====================================================================


class TestRuleOfThumbValidation:
    """Input-validation guards for RuleOfThumb."""

    def test_phi_below_one_raises(self):
        with pytest.raises(ValidationError, match="phi >= 1.0"):
            RuleOfThumb(phi=0.5)

    def test_phi_negative_raises(self):
        with pytest.raises(ValidationError, match="phi >= 1.0"):
            RuleOfThumb(phi=-1.0)

    def test_phi_nan_raises(self):
        with pytest.raises(ValidationError, match="phi >= 1.0"):
            RuleOfThumb(phi=float('nan'))

    def test_phi_inf_rejected(self):
        # Infinite phi is finite-check-rejected (division-by-(1+inf) is
        # degenerate; users should express p=0 by a different path).
        with pytest.raises(ValidationError, match="phi >= 1.0"):
            RuleOfThumb(phi=float('inf'))

    def test_phi_string_raises(self):
        with pytest.raises(ValidationError, match="numeric scalar"):
            RuleOfThumb(phi="2.0")

    def test_phi_bool_raises(self):
        # isinstance(True, int) is True; reject explicitly.
        with pytest.raises(ValidationError, match="numeric scalar"):
            RuleOfThumb(phi=True)

    def test_cost_scaling_conflict_raises(self):
        with pytest.raises(ValidationError, match="cost_scaling"):
            RuleOfThumb(phi=2.0, cost_scaling=0.5)


class TestConstantMarkupValidation:
    def test_scalar_float(self):
        c = ConstantMarkup(markup=0.5)
        assert c.markup == 0.5
        assert c._model_name == 'constant_markup'

    def test_scalar_int(self):
        c = ConstantMarkup(markup=1)
        assert c.markup == 1

    def test_column_name(self):
        c = ConstantMarkup(markup='eta_col')
        assert c.markup == 'eta_col'

    def test_list_markup_raises(self):
        with pytest.raises(ValidationError, match="scalar or a column-name"):
            ConstantMarkup(markup=[1.0, 2.0])  # type: ignore[arg-type]

    def test_none_markup_raises(self):
        with pytest.raises(ValidationError, match="scalar or a column-name"):
            ConstantMarkup(markup=None)  # type: ignore[arg-type]

    def test_empty_column_name_raises(self):
        with pytest.raises(ValidationError, match="non-empty"):
            ConstantMarkup(markup='')

    def test_nan_scalar_raises(self):
        with pytest.raises(ValidationError, match="finite"):
            ConstantMarkup(markup=float('nan'))

    def test_user_supplied_conflict_raises(self):
        with pytest.raises(ValidationError, match="user_supplied_markups"):
            ConstantMarkup(markup=0.5, user_supplied_markups='other_col')

    def test_compute_markup_hook_raises(self):
        """Direct invocation of the abstract hook is unsupported: the
        markup is a model primitive and does not come from (O, D, s).
        """
        c = ConstantMarkup(markup=0.5)
        with pytest.raises(NotImplementedError, match="model primitive"):
            c._compute_markup(np.eye(2), np.eye(2), np.array([0.3, 0.3]))


# =====================================================================
# Class-level math hooks
# =====================================================================


class TestRuleOfThumbHook:
    """RuleOfThumb hook contract: _compute_markup raises NotImplementedError
    (markup requires prices, not in the (O, D, s) triple); the Dearing math
    is delivered by Problem.__init__ pre-computing (phi-1)/phi * prices.
    """

    def test_compute_markup_raises(self):
        r = RuleOfThumb(phi=2.0)
        with pytest.raises(NotImplementedError, match="prices"):
            r._compute_markup(np.eye(3), np.eye(3), np.array([0.2, 0.3, 0.4]))

    def test_markup_derivative_returns_zero(self):
        r = RuleOfThumb(phi=2.0)
        d = r._markup_derivative(
            np.eye(3), np.eye(3), np.eye(3),
            np.array([0.2, 0.3, 0.4]), np.zeros(3),
        )
        np.testing.assert_array_equal(d, np.zeros(3))

    def test_model_name_is_constant_markup(self):
        r = RuleOfThumb(phi=1.0)
        assert r._model_name == 'constant_markup'


# =====================================================================
# End-to-end Problem.solve integration
# =====================================================================


def _make_dgp(seed: int = 1234, T: int = 10, J: int = 3, alpha: float = -1.5):
    """Minimal plain-logit fixture for Problem.solve integration tests.

    Mirrors ``tests/test_models_integration.py::_make_dgp`` but with a
    smaller grid and an extra ``eta_col`` column for ConstantMarkup's
    column-name interface.
    """
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
    # Per-product dollar markup for the column-name ConstantMarkup test:
    # one value per product repeated across markets.
    eta_per_product = np.array([0.10, 0.20, 0.30])[:J]
    eta_col = np.tile(eta_per_product, T)
    df = pd.DataFrame({
        'market_ids': market_ids,
        'firm_ids': firm_ids,
        'x1': x1,
        'prices': prices,
        'shares': shares,
        'rival_x1': rival_x1,
        'z1': z1,
        'intercept': np.ones(N),
        'eta_col': eta_col,
    })
    return df, alpha


def _common_problem_kwargs(df, alpha):
    return dict(
        cost_formulation=pyRVtest.Formulation('1 + z1'),
        instrument_formulation=pyRVtest.Formulation('0 + rival_x1'),
        product_data=df,
        demand_params={
            'alpha': alpha, 'rho': [],
            'beta': np.array([0.0, 0.4]),
            'x_columns': ['intercept', 'x1'],
            'demand_instrument_columns': ['rival_x1', 'intercept', 'x1'],
        },
    )


class TestRuleOfThumbEndToEnd:
    """RuleOfThumb through Problem.solve — marginal cost equals p / phi."""

    @pytest.fixture(scope='class')
    def dgp(self):
        return _make_dgp()

    def test_phi_one_matches_perfect_competition(self, dgp):
        """phi=1 degenerates to marginal-cost pricing: mc == price, matching
        PerfectCompetition byte-for-byte.
        """
        df, alpha = dgp
        common = _common_problem_kwargs(df, alpha)
        pyRVtest.options.verbose = False
        r_rot = pyRVtest.Problem(
            **common,
            models=[
                RuleOfThumb(phi=1.0),
                Bertrand(ownership='firm_ids'),
            ],
        ).solve(demand_adjustment=False, clustering_adjustment=False)
        r_pc = pyRVtest.Problem(
            **common,
            models=[
                PerfectCompetition(),
                Bertrand(ownership='firm_ids'),
            ],
        ).solve(demand_adjustment=False, clustering_adjustment=False)
        # mc[0] is the implied marginal cost vector for the first model.
        np.testing.assert_allclose(
            r_rot.markups[0], r_pc.markups[0], atol=1e-14,
            err_msg="RuleOfThumb(phi=1) should have zero raw markup",
        )
        # Effective marginal_cost matches price exactly in both cases.
        np.testing.assert_allclose(
            r_rot.marginal_cost[0], r_pc.marginal_cost[0], atol=1e-14,
            err_msg="RuleOfThumb(phi=1) should yield mc == price",
        )
        np.testing.assert_allclose(
            r_rot.marginal_cost[0].flatten(),
            np.asarray(df['prices']),
            atol=1e-14,
        )

    def test_phi_two_is_half_price_marginal_cost(self, dgp):
        """phi=2: mc = price / 2 (50%-of-price markup)."""
        df, alpha = dgp
        common = _common_problem_kwargs(df, alpha)
        pyRVtest.options.verbose = False
        r = pyRVtest.Problem(
            **common,
            models=[
                RuleOfThumb(phi=2.0),
                Bertrand(ownership='firm_ids'),
            ],
        ).solve(demand_adjustment=False, clustering_adjustment=False)
        expected_mc = np.asarray(df['prices']) / 2.0
        np.testing.assert_allclose(
            r.marginal_cost[0].flatten(), expected_mc, atol=1e-14,
        )

    def test_phi_two_markup_equals_half_price(self, dgp):
        """phi=2: markup = (phi-1)/phi * p = p/2 (not zero)."""
        df, alpha = dgp
        common = _common_problem_kwargs(df, alpha)
        pyRVtest.options.verbose = False
        r = pyRVtest.Problem(
            **common,
            models=[
                RuleOfThumb(phi=2.0),
                Bertrand(ownership='firm_ids'),
            ],
        ).solve(demand_adjustment=False, clustering_adjustment=False)
        expected_markup = np.asarray(df['prices']) / 2.0
        np.testing.assert_allclose(
            r.markups[0].flatten(), expected_markup, atol=1e-14,
        )

    def test_phi_general_markup_formula(self, dgp):
        """markup = (phi-1)/phi * p for arbitrary phi."""
        df, alpha = dgp
        common = _common_problem_kwargs(df, alpha)
        pyRVtest.options.verbose = False
        phi = 1.5
        r = pyRVtest.Problem(
            **common,
            models=[
                RuleOfThumb(phi=phi),
                Bertrand(ownership='firm_ids'),
            ],
        ).solve(demand_adjustment=False, clustering_adjustment=False)
        expected_markup = (phi - 1) / phi * np.asarray(df['prices'])
        np.testing.assert_allclose(
            r.markups[0].flatten(), expected_markup, atol=1e-14,
        )
        np.testing.assert_allclose(
            r.marginal_cost[0].flatten(),
            np.asarray(df['prices']) / phi,
            atol=1e-14,
        )

    def test_rule_of_thumb_matches_legacy_cost_scaling(self, dgp):
        """RuleOfThumb(phi=2.0) reproduces the legacy v0.3 pattern of
        PerfectCompetition with a per-row cost_scaling column whose values
        all equal 1.0 (= phi - 1). This is the backward-compatibility
        guarantee from the plan.
        """
        df, alpha = dgp
        df_with_lmbda = df.copy()
        df_with_lmbda['lmbda_col'] = np.ones(len(df))
        common_rot = _common_problem_kwargs(df, alpha)
        common_leg = _common_problem_kwargs(df_with_lmbda, alpha)
        pyRVtest.options.verbose = False
        r_new = pyRVtest.Problem(
            **common_rot,
            models=[RuleOfThumb(phi=2.0), Bertrand(ownership='firm_ids')],
        ).solve(demand_adjustment=False, clustering_adjustment=False)
        r_old = pyRVtest.Problem(
            **common_leg,
            models=[
                PerfectCompetition(cost_scaling='lmbda_col'),
                Bertrand(ownership='firm_ids'),
            ],
        ).solve(demand_adjustment=False, clustering_adjustment=False)
        np.testing.assert_allclose(
            r_new.marginal_cost, r_old.marginal_cost, atol=1e-14,
        )
        np.testing.assert_allclose(r_new.TRV, r_old.TRV, atol=1e-14)
        np.testing.assert_allclose(r_new.F, r_old.F, atol=1e-14)

    def test_rule_of_thumb_demand_adjustment_runs(self, dgp):
        """RuleOfThumb combines with demand_adjustment=True (first-stage
        correction). The raw markup is zero so the gradient is zero, which
        is consistent with the perfect_competition skip in
        solve/demand_adjustment.py.
        """
        df, alpha = dgp
        common = _common_problem_kwargs(df, alpha)
        pyRVtest.options.verbose = False
        r = pyRVtest.Problem(
            **common,
            models=[RuleOfThumb(phi=2.0), Bertrand(ownership='firm_ids')],
        ).solve(demand_adjustment=True, clustering_adjustment=False)
        expected_mc = np.asarray(df['prices']) / 2.0
        np.testing.assert_allclose(
            r.marginal_cost[0].flatten(), expected_mc, atol=1e-14,
        )
        # Off-diagonal entries carry the pairwise test statistic; the
        # self-comparison diagonals are NaN by construction.
        trv = np.asarray(r.TRV)
        off_diag = trv[~np.isnan(trv)]
        assert off_diag.size > 0
        assert np.isfinite(off_diag).all()


class TestConstantMarkupEndToEnd:
    """ConstantMarkup through Problem.solve — markup vector equals the
    specified scalar or column.
    """

    @pytest.fixture(scope='class')
    def dgp(self):
        return _make_dgp()

    def test_scalar_markup(self, dgp):
        """ConstantMarkup(markup=0.25): markup == 0.25 * ones(N)."""
        df, alpha = dgp
        common = _common_problem_kwargs(df, alpha)
        pyRVtest.options.verbose = False
        r = pyRVtest.Problem(
            **common,
            models=[ConstantMarkup(markup=0.25), Bertrand(ownership='firm_ids')],
        ).solve(demand_adjustment=False, clustering_adjustment=False)
        np.testing.assert_allclose(
            r.markups[0].flatten(), np.full(len(df), 0.25), atol=1e-14,
        )
        np.testing.assert_allclose(
            r.marginal_cost[0].flatten(),
            np.asarray(df['prices']) - 0.25,
            atol=1e-14,
        )

    def test_column_name_markup(self, dgp):
        """ConstantMarkup(markup='eta_col'): markup == per-product column."""
        df, alpha = dgp
        common = _common_problem_kwargs(df, alpha)
        pyRVtest.options.verbose = False
        r = pyRVtest.Problem(
            **common,
            models=[
                ConstantMarkup(markup='eta_col'),
                Bertrand(ownership='firm_ids'),
            ],
        ).solve(demand_adjustment=False, clustering_adjustment=False)
        np.testing.assert_allclose(
            r.markups[0].flatten(),
            np.asarray(df['eta_col']),
            atol=1e-14,
        )
        np.testing.assert_allclose(
            r.marginal_cost[0].flatten(),
            np.asarray(df['prices']) - np.asarray(df['eta_col']),
            atol=1e-14,
        )

    def test_missing_column_raises(self, dgp):
        """ConstantMarkup referencing a column not in product_data fails
        loudly at Problem.__init__ with a helpful error.
        """
        df, alpha = dgp
        common = _common_problem_kwargs(df, alpha)
        pyRVtest.options.verbose = False
        with pytest.raises(ValidationError, match="missing_col"):
            pyRVtest.Problem(
                **common,
                models=[
                    ConstantMarkup(markup='missing_col'),
                    Bertrand(ownership='firm_ids'),
                ],
            )

    def test_constant_markup_demand_adjustment_runs(self, dgp):
        """ConstantMarkup combines with demand_adjustment=True; markup is
        a model primitive, so the gradient w.r.t. theta is zero.
        """
        df, alpha = dgp
        common = _common_problem_kwargs(df, alpha)
        pyRVtest.options.verbose = False
        r = pyRVtest.Problem(
            **common,
            models=[ConstantMarkup(markup=0.25), Bertrand(ownership='firm_ids')],
        ).solve(demand_adjustment=True, clustering_adjustment=False)
        np.testing.assert_allclose(
            r.markups[0].flatten(), np.full(len(df), 0.25), atol=1e-14,
        )
        # Off-diagonal entries carry the pairwise test statistic; the
        # self-comparison diagonals are NaN by construction.
        trv = np.asarray(r.TRV)
        off_diag = trv[~np.isnan(trv)]
        assert off_diag.size > 0
        assert np.isfinite(off_diag).all()


class TestCombinationWithTaxes:
    """RuleOfThumb and ConstantMarkup combined with the tax machinery."""

    @pytest.fixture(scope='class')
    def dgp_with_taxes(self):
        df, alpha = _make_dgp()
        df = df.copy()
        # Flat per-unit tax so all post-tax effects are deterministic and
        # easy to assert. Small value relative to prices to keep mc > 0.
        df['tax_col'] = np.full(len(df), 0.05)
        return df, alpha

    def test_rule_of_thumb_with_unit_tax(self, dgp_with_taxes):
        """RuleOfThumb(phi=2) + unit_tax: mc = p / phi - unit_tax."""
        df, alpha = dgp_with_taxes
        common = _common_problem_kwargs(df, alpha)
        pyRVtest.options.verbose = False
        r = pyRVtest.Problem(
            **common,
            models=[
                RuleOfThumb(phi=2.0, unit_tax='tax_col'),
                Bertrand(ownership='firm_ids'),
            ],
        ).solve(demand_adjustment=False, clustering_adjustment=False)
        expected_mc = np.asarray(df['prices']) / 2.0 - 0.05
        np.testing.assert_allclose(
            r.marginal_cost[0].flatten(), expected_mc, atol=1e-14,
        )

    def test_constant_markup_with_unit_tax(self, dgp_with_taxes):
        """ConstantMarkup(markup=0.25) + unit_tax:
            prices_effective = p - unit_tax
            markups_effective = 0.25
            mc = (p - unit_tax) - 0.25
        """
        df, alpha = dgp_with_taxes
        common = _common_problem_kwargs(df, alpha)
        pyRVtest.options.verbose = False
        r = pyRVtest.Problem(
            **common,
            models=[
                ConstantMarkup(markup=0.25, unit_tax='tax_col'),
                Bertrand(ownership='firm_ids'),
            ],
        ).solve(demand_adjustment=False, clustering_adjustment=False)
        expected_mc = np.asarray(df['prices']) - 0.05 - 0.25
        np.testing.assert_allclose(
            r.marginal_cost[0].flatten(), expected_mc, atol=1e-14,
        )
