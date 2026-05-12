"""Tests for the rc14 PT reliability vocabulary (Audit 2 Findings C2 + C3).

``compute_passthrough_reliability`` is a passive diagnostic — it does
NOT change any computed pass-through value. These tests verify:

1. The function returns the documented column set and dtype contract.
2. ``pt_method`` correctly classifies each conduct (analytical_trivial /
   analytical_vertical / numerical_central_difference).
3. ``pt_status`` threshold logic maps condition numbers to the four
   states (robust / ill-conditioned / near-degenerate / undefined).
4. The reported ``pt_condition_number`` matches a direct
   ``np.linalg.cond`` call on the same P matrix returned by
   ``build_passthrough`` (i.e., it doesn't drift).
5. The pass-through values themselves are unchanged by adding the
   diagnostic — pinned via the existing ``test_compute_markups_direct``
   suite, plus an explicit value-parity check here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyRVtest
from pyRVtest.solve.passthrough import (
    _classify_condition_number,
    _classify_pt_method,
    build_passthrough,
    compute_passthrough_reliability,
)


@pytest.fixture(scope='module')
def synthetic_problem():
    df = pyRVtest.data.load_example()
    pyRVtest.options.verbose = False
    return pyRVtest.Problem(
        cost_formulation=pyRVtest.Formulation('1 + z1 + z2'),
        instrument_formulation=pyRVtest.Formulation('0 + rival_z1 + rival_z2'),
        product_data=df,
        demand_results=None,
        demand_params={
            'alpha': -2.0, 'beta': [0.0], 'sigma': [],
            'x_columns': ['x1'],
            'demand_instrument_columns': ['rival_z1', 'rival_z2', 'x1'],
        },
        models=[
            pyRVtest.Bertrand(ownership='firm_ids'),
            pyRVtest.Cournot(ownership='firm_ids'),
            pyRVtest.Monopoly(),
            pyRVtest.PerfectCompetition(),
        ],
    )


class TestColumnContract:
    def test_columns_match_spec(self, synthetic_problem):
        out = synthetic_problem.passthrough_reliability()
        expected = [
            'model_index', 'model', 'market_id',
            'pt_method', 'pt_condition_number', 'pt_rank',
            'pt_status', 'pt_warning',
        ]
        assert list(out.columns) == expected

    def test_one_row_per_model_market(self, synthetic_problem):
        out = synthetic_problem.passthrough_reliability()
        n_models = len(synthetic_problem._models)
        n_markets = len(synthetic_problem.unique_market_ids)
        assert len(out) == n_models * n_markets


class TestMethodClassification:
    def test_perfect_competition_is_analytical_trivial(self, synthetic_problem):
        out = synthetic_problem.passthrough_reliability()
        pc_rows = out[out['model'] == 'PerfectCompetition']
        assert (pc_rows['pt_method'] == 'analytical_trivial').all()

    def test_bertrand_cournot_monopoly_are_numerical(self, synthetic_problem):
        out = synthetic_problem.passthrough_reliability()
        for label in ('Bertrand', 'Cournot', 'Monopoly'):
            rows = out[out['model'] == label]
            assert (rows['pt_method'] == 'numerical_central_difference').all(), (
                f"{label} should use numerical central difference"
            )


class TestConditionNumberValues:
    def test_analytical_trivial_cond_is_one(self, synthetic_problem):
        """PC short-circuits to P = I, so cond = 1 exactly."""
        out = synthetic_problem.passthrough_reliability()
        pc_rows = out[out['model'] == 'PerfectCompetition']
        np.testing.assert_allclose(
            pc_rows['pt_condition_number'], 1.0, atol=1e-12,
        )

    def test_reported_cond_matches_direct_call(self, synthetic_problem):
        """Spot-check: reported pt_condition_number equals np.linalg.cond
        of the same per-market P that build_passthrough returns. Pins
        the diagnostic to the live PT matrices."""
        # Pick the first market and the Bertrand model.
        m = 0
        t = next(iter(synthetic_problem.unique_market_ids))
        pt_dict = build_passthrough(synthetic_problem, model_index=m)
        P_t = np.asarray(pt_dict[t], dtype=float)
        cond_direct = float(np.linalg.cond(P_t))

        out = synthetic_problem.passthrough_reliability()
        cond_reported = out[
            (out['model_index'] == m) & (out['market_id'] == t)
        ]['pt_condition_number'].iloc[0]
        np.testing.assert_allclose(
            cond_reported, cond_direct, atol=1e-12,
        )


class TestStatusThresholds:
    """Pin the four-level status logic by feeding _classify_condition_number
    directly."""

    def test_robust_below_warn(self):
        status, warn = _classify_condition_number(
            cond_val=1e3, rank=2, J=2,
            cond_warn=1e6, cond_severe=1e12, cond_undefined=1e16,
        )
        assert status == 'robust'
        assert warn == ''

    def test_ill_conditioned_above_warn(self):
        status, warn = _classify_condition_number(
            cond_val=1e8, rank=2, J=2,
            cond_warn=1e6, cond_severe=1e12, cond_undefined=1e16,
        )
        assert status == 'ill-conditioned'
        assert 'ill-conditioned' in warn

    def test_near_degenerate_above_severe(self):
        status, warn = _classify_condition_number(
            cond_val=1e14, rank=2, J=2,
            cond_warn=1e6, cond_severe=1e12, cond_undefined=1e16,
        )
        assert status == 'near-degenerate'
        assert 'near-singular' in warn

    def test_undefined_above_undefined_threshold(self):
        status, warn = _classify_condition_number(
            cond_val=1e18, rank=2, J=2,
            cond_warn=1e6, cond_severe=1e12, cond_undefined=1e16,
        )
        assert status == 'undefined'

    def test_undefined_when_rank_deficient(self):
        status, warn = _classify_condition_number(
            cond_val=1e3, rank=1, J=2,  # cond fine but rank-deficient
            cond_warn=1e6, cond_severe=1e12, cond_undefined=1e16,
        )
        assert status == 'undefined'
        assert 'rank' in warn

    def test_undefined_when_cond_non_finite(self):
        status, warn = _classify_condition_number(
            cond_val=float('inf'), rank=2, J=2,
            cond_warn=1e6, cond_severe=1e12, cond_undefined=1e16,
        )
        assert status == 'undefined'

    def test_thresholds_configurable(self, synthetic_problem):
        """Lowering cond_warn must reclassify a non-trivial set of rows
        as ill-conditioned. PerfectCompetition (cond exactly 1.0) stays
        robust at the boundary; Bertrand/Cournot/Monopoly typically run
        with cond in the 2-5 range but a handful of markets sit very
        close to 1 (Bertrand approaches PC in the small-share limit),
        so we only require *most* non-PC rows to flip, not all."""
        out_default = synthetic_problem.passthrough_reliability()
        assert (out_default['pt_status'] == 'robust').all()

        # cond_warn just above the absolute minimum (cond is always >= 1):
        # PC stays exactly 1.0 -> robust; the bulk of non-PC markets have
        # cond > 1.001 and should flip.
        out_tight = synthetic_problem.passthrough_reliability(cond_warn=1.001)
        pc_rows = out_tight[out_tight['model'] == 'PerfectCompetition']
        assert (pc_rows['pt_status'] == 'robust').all()
        non_pc = out_tight[out_tight['model'] != 'PerfectCompetition']
        flipped = (non_pc['pt_status'] == 'ill-conditioned')
        assert flipped.mean() > 0.95, (
            f"Expected the cond_warn=1.001 threshold to flip the vast "
            f"majority of non-PC rows to ill-conditioned. "
            f"Got flip rate {flipped.mean():.3f}."
        )


class TestMarketIdFiltering:
    def test_single_market_returns_one_row_per_model(self, synthetic_problem):
        t = next(iter(synthetic_problem.unique_market_ids))
        out = synthetic_problem.passthrough_reliability(market_id=t)
        n_models = len(synthetic_problem._models)
        assert len(out) == n_models
        assert (out['market_id'] == t).all()

    def test_invalid_market_id_raises(self, synthetic_problem):
        with pytest.raises(ValueError, match='unique_market_ids'):
            synthetic_problem.passthrough_reliability(market_id=999_999)


class TestValuePreservation:
    """The diagnostic must not change any computed pass-through value.

    Compares ``passthrough_summary`` output before / after calling
    ``passthrough_reliability`` and confirms the per-pair distance
    metrics are bit-identical (i.e., no shared state pollution).
    """

    def test_passthrough_summary_unchanged_after_reliability_call(
            self, synthetic_problem,
    ):
        s1 = synthetic_problem.passthrough_summary().to_dataframe()
        synthetic_problem.passthrough_reliability()
        s2 = synthetic_problem.passthrough_summary().to_dataframe()
        pd.testing.assert_frame_equal(s1, s2, check_exact=True)


class TestProblemResultsMirror:
    """ProblemResults.passthrough_reliability should match the
    Problem-level version exactly (it's just a delegating wrapper)."""

    def test_results_mirror_matches_problem(self, synthetic_problem):
        results = synthetic_problem.solve(demand_adjustment=False)
        from_problem = synthetic_problem.passthrough_reliability()
        from_results = results.passthrough_reliability()
        # Drop and re-sort for comparison (DataFrame may emerge in
        # different order but content should match).
        a = from_problem.sort_values(['model_index', 'market_id']).reset_index(drop=True)
        b = from_results.sort_values(['model_index', 'market_id']).reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, check_exact=True)


class TestPtMethodClassifier:
    """Unit-test _classify_pt_method directly to lock the decision tree."""

    def test_perfect_competition(self):
        assert _classify_pt_method(
            pyRVtest.PerfectCompetition(), is_vertical=False,
        ) == 'analytical_trivial'

    def test_constant_markup(self):
        assert _classify_pt_method(
            pyRVtest.ConstantMarkup(markup=0.5), is_vertical=False,
        ) == 'analytical_trivial'

    def test_rule_of_thumb(self):
        assert _classify_pt_method(
            pyRVtest.RuleOfThumb(phi=2.0), is_vertical=False,
        ) == 'analytical_trivial'

    def test_bertrand(self):
        assert _classify_pt_method(
            pyRVtest.Bertrand(ownership='firm_ids'), is_vertical=False,
        ) == 'numerical_central_difference'

    def test_cournot(self):
        assert _classify_pt_method(
            pyRVtest.Cournot(ownership='firm_ids'), is_vertical=False,
        ) == 'numerical_central_difference'

    def test_monopoly(self):
        assert _classify_pt_method(
            pyRVtest.Monopoly(), is_vertical=False,
        ) == 'numerical_central_difference'

    def test_vertical(self):
        # is_vertical=True is what build_passthrough computes; just
        # pass a Bertrand here as the placeholder candidate.
        assert _classify_pt_method(
            pyRVtest.Bertrand(ownership='firm_ids'), is_vertical=True,
        ) == 'analytical_vertical'
