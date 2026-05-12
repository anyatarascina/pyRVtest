"""DMQSW Phase 2: pair × pass-through-feature distance summary.

Exercises :meth:`pyRVtest.Problem.passthrough_summary` and
:meth:`pyRVtest.ProblemResults.passthrough_summary` on the shipped
2-firm 3000-market synthetic example, verifying:

1. The four DMQSW-keyed feature distances (``offdiag_ratio``,
   ``full_pass``, ``row_sum``, ``level_adj``) are populated for every
   candidate pair.
2. The headline DMQSW Example 2 result for (Cournot, PerfectCompetition):
   under logit demand both candidates have diagonal pass-through, so
   ``offdiag_ratio`` collapses to ~0 — rival cost shifters cannot
   distinguish the pair regardless of γ.
3. Other features (``row_sum``, ``level_adj``) are nonzero for the same
   pair: switching to unit tax or ad valorem tax instruments breaks the
   degeneracy.
4. ``with_models=True`` adds a per-model structural block.
5. ``detail='full'`` returns one row per (market, pair).
6. ``ProblemResults.passthrough_summary`` matches
   ``Problem.passthrough_summary``.
7. The methodology line records how the pass-through matrices were
   computed conditional on the candidate-set composition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyRVtest
from pyRVtest import Bertrand, Cournot, Monopoly, PerfectCompetition
from pyRVtest.solve.passthrough import (
    PassthroughSummary,
    compute_passthrough_summary,
)


@pytest.fixture(scope='module')
def synthetic_problem():
    data = pyRVtest.data.load_example()
    return pyRVtest.Problem(
        cost_formulation=pyRVtest.Formulation('1 + z1 + z2'),
        instrument_formulation=pyRVtest.Formulation('0 + rival_z1 + rival_z2'),
        models=[
            Bertrand(ownership='firm_ids'),
            Cournot(ownership='firm_ids'),
            Monopoly(),
            PerfectCompetition(),
        ],
        product_data=data,
        demand_params={
            'estimate': 'logit',
            'formulation_X': pyRVtest.Formulation('1 + x1'),
            'formulation_Z': pyRVtest.Formulation('0 + z1'),
        },
    )


class TestPairDistanceTable:
    """The pair-distance DataFrame has the right shape and columns."""

    def test_returns_passthrough_summary_object(self, synthetic_problem):
        summary = synthetic_problem.passthrough_summary()
        assert isinstance(summary, PassthroughSummary)

    def test_pair_distances_has_one_row_per_pair(self, synthetic_problem):
        summary = synthetic_problem.passthrough_summary()
        # 4 candidates → 4 choose 2 = 6 pairs.
        assert len(summary.pair_distances) == 6

    def test_pair_distances_has_four_features(self, synthetic_problem):
        summary = synthetic_problem.passthrough_summary()
        for col in ('offdiag_ratio', 'full_pass', 'row_sum', 'level_adj'):
            assert col in summary.pair_distances.columns

    def test_pair_label_matches_class_names(self, synthetic_problem):
        summary = synthetic_problem.passthrough_summary()
        labels = summary.pair_distances['pair'].tolist()
        # Confirm Cournot / PerfectCompetition pair is present.
        cournot_pc_label = '(Cournot, PerfectCompetition)'
        assert cournot_pc_label in labels


class TestCournotPCDegeneracy:
    """The headline DMQSW Example 2 result: rival cost shifters cannot
    distinguish (Cournot, PerfectCompetition) under logit demand."""

    def test_cournot_pc_offdiag_ratio_is_near_zero(self, synthetic_problem):
        summary = synthetic_problem.passthrough_summary()
        df = summary.pair_distances
        cournot_pc = df[df['pair'] == '(Cournot, PerfectCompetition)']
        assert len(cournot_pc) == 1
        offdiag = float(cournot_pc['offdiag_ratio'].iloc[0])
        # Numerical floor — Cournot and PC both have diagonal pass-through
        # under logit, so column-normalized off-diagonals are exactly zero
        # in both. Numerical noise from finite-difference perturbation
        # leaves a residual of ~1e-9.
        assert offdiag < 1e-6, (
            f"(Cournot, PC) offdiag_ratio should be ~0 (Remark 1 degeneracy "
            f"under logit), got {offdiag:.3e}"
        )

    def test_cournot_pc_other_features_nonzero(self, synthetic_problem):
        """Other instrument types (unit tax, ad valorem) DO distinguish
        the pair — their feature distances should be substantially nonzero."""
        summary = synthetic_problem.passthrough_summary()
        df = summary.pair_distances
        cournot_pc = df[df['pair'] == '(Cournot, PerfectCompetition)'].iloc[0]
        assert float(cournot_pc['row_sum']) > 0.1, (
            "Unit tax should distinguish (Cournot, PC) — row sums of P "
            "differ between Cournot's diag(s_0/(1-s_2), s_0/(1-s_1)) and "
            "PC's identity."
        )
        assert float(cournot_pc['level_adj']) > 0.1, (
            "Ad valorem tax should distinguish (Cournot, PC)."
        )
        assert float(cournot_pc['full_pass']) > 0.1, (
            "Own+rival cost shifters should distinguish (Cournot, PC) "
            "(Remark 2 — full pass-through matrices differ)."
        )


class TestWithModels:
    """``with_models=True`` adds a per-model structural block."""

    def test_with_models_false_has_no_per_model(self, synthetic_problem):
        summary = synthetic_problem.passthrough_summary(with_models=False)
        assert summary.per_model is None

    def test_with_models_true_has_per_model_block(self, synthetic_problem):
        summary = synthetic_problem.passthrough_summary(with_models=True)
        assert isinstance(summary.per_model, pd.DataFrame)
        # One row per candidate.
        assert len(summary.per_model) == 4
        for col in ('model', 'diag_avg', 'max_offdiag', 'row_sum_avg'):
            assert col in summary.per_model.columns

    def test_per_model_pc_has_identity_pass_through(self, synthetic_problem):
        """PerfectCompetition has P = I exactly: diag_avg = 1, max_offdiag = 0,
        row_sum_avg = 1."""
        summary = synthetic_problem.passthrough_summary(with_models=True)
        pc = summary.per_model[summary.per_model['model'] == 'PerfectCompetition']
        assert len(pc) == 1
        np.testing.assert_allclose(float(pc['diag_avg'].iloc[0]), 1.0, atol=1e-12)
        np.testing.assert_allclose(float(pc['max_offdiag'].iloc[0]), 0.0, atol=1e-12)
        np.testing.assert_allclose(float(pc['row_sum_avg'].iloc[0]), 1.0, atol=1e-12)

    def test_per_model_cournot_has_zero_offdiag(self, synthetic_problem):
        """Cournot under logit has diagonal pass-through: max_offdiag = 0."""
        summary = synthetic_problem.passthrough_summary(with_models=True)
        cournot = summary.per_model[summary.per_model['model'] == 'Cournot']
        assert len(cournot) == 1
        # Numerical floor; Cournot's off-diagonals are zero analytically but
        # the numerical core gives ~1e-9.
        assert abs(float(cournot['max_offdiag'].iloc[0])) < 1e-6


class TestDetailModes:
    """``detail='median'`` aggregates; ``detail='full'`` returns per-market."""

    def test_median_default(self, synthetic_problem):
        summary = synthetic_problem.passthrough_summary()
        assert summary.detail_mode == 'median'

    def test_full_mode_returns_per_market_rows(self, synthetic_problem):
        summary = synthetic_problem.passthrough_summary(detail='full')
        # 6 pairs × 3000 markets = 18000 rows.
        assert len(summary.pair_distances) == 6 * 3000
        assert 'market_id' in summary.pair_distances.columns

    def test_invalid_detail_raises(self, synthetic_problem):
        with pytest.raises(ValueError, match="detail"):
            synthetic_problem.passthrough_summary(detail='quartile')


class TestProblemResultsWrapper:
    """``ProblemResults.passthrough_summary`` delegates to ``Problem``."""

    @pytest.fixture(scope='class')
    def synthetic_results(self):
        data = pyRVtest.data.load_example()
        problem = pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('1 + z1 + z2'),
            instrument_formulation=pyRVtest.Formulation('0 + rival_z1 + rival_z2'),
            models=[
                Bertrand(ownership='firm_ids'),
                Cournot(ownership='firm_ids'),
                Monopoly(),
                PerfectCompetition(),
            ],
            product_data=data,
            demand_params={
                'estimate': 'logit',
                'formulation_X': pyRVtest.Formulation('1 + x1'),
                'formulation_Z': pyRVtest.Formulation('0 + z1'),
            },
        )
        return problem.solve(demand_adjustment=False)

    def test_results_method_returns_same_summary(
        self, synthetic_results, synthetic_problem,
    ):
        problem_summary = synthetic_problem.passthrough_summary()
        results_summary = synthetic_results.passthrough_summary()
        # Both should produce identical pair-distance frames (same Problem
        # underlying).
        pd.testing.assert_frame_equal(
            problem_summary.pair_distances.sort_values(['model_i', 'model_j']).reset_index(drop=True),
            results_summary.pair_distances.sort_values(['model_i', 'model_j']).reset_index(drop=True),
        )


class TestMethodologyLine:
    """The methodology line reflects the candidate-set composition."""

    def test_mixed_set_mentions_numerical_and_short_circuit(
        self, synthetic_problem,
    ):
        """Synthetic example mixes numerical conducts (Bertrand, Cournot,
        Monopoly) with a trivial short-circuit (PerfectCompetition)."""
        summary = synthetic_problem.passthrough_summary()
        line = summary.methodology_line
        assert 'central-difference numerical' in line
        assert 'short-circuit' in line
        assert 'docs/math.rst' in line

    def test_repr_contains_methodology_line(self, synthetic_problem):
        summary = synthetic_problem.passthrough_summary()
        rendered = repr(summary)
        assert 'Methodology' in rendered
        assert 'central-difference' in rendered


class TestRepr:
    """The ``__repr__`` produces a readable formatted view."""

    def test_repr_includes_pair_table_header(self, synthetic_problem):
        summary = synthetic_problem.passthrough_summary()
        rendered = repr(summary)
        assert 'Per-pair pass-through-feature distances' in rendered

    def test_repr_includes_per_feature_notes(self, synthetic_problem):
        summary = synthetic_problem.passthrough_summary()
        rendered = repr(summary)
        assert 'Per-feature notes' in rendered
        assert 'offdiag_ratio' in rendered
        assert 'row_sum' in rendered

    def test_repr_with_models_includes_per_model_block(self, synthetic_problem):
        summary = synthetic_problem.passthrough_summary(with_models=True)
        rendered = repr(summary)
        assert 'Per-model pass-through structure' in rendered
        assert 'diag_avg' in rendered

    def test_to_dataframe_returns_copy(self, synthetic_problem):
        summary = synthetic_problem.passthrough_summary()
        df = summary.to_dataframe()
        # Modifying the returned frame should not mutate the summary.
        df['offdiag_ratio'] = 999
        assert (summary.pair_distances['offdiag_ratio'] != 999).all()


class TestMarkupsAssemblyCallCount:
    """Pin the markups-assembly call count.

    rc6 audit follow-through (Finding 4): compute_passthrough_summary
    used to call build_passthrough(model_index=m) for each model, and
    each build_passthrough internally called _perturb_and_build_markups.
    For ``n_models`` candidates this multiplied the markups assembly
    cost by ``1 + n_models``. The precomputed-markups hook threads a
    single assembly result through all per-model build_passthrough
    calls; total assembly count is now ``1`` regardless of n_models.
    """

    def test_passthrough_summary_calls_markups_assembly_once(self, synthetic_problem):
        """Calling passthrough_summary on a problem with N models should
        trigger exactly one _perturb_and_build_markups call, not N+1."""
        problem = synthetic_problem
        n_calls = {'count': 0}
        orig = type(problem)._perturb_and_build_markups

        def counter(self):
            n_calls['count'] += 1
            return orig(self)

        type(problem)._perturb_and_build_markups = counter
        try:
            problem.passthrough_summary()
        finally:
            type(problem)._perturb_and_build_markups = orig

        assert n_calls['count'] == 1, (
            f"Expected exactly 1 markups-assembly call across the whole "
            f"passthrough_summary path (cached). "
            f"Received {n_calls['count']} calls — the precomputed-markups "
            f"hook in build_passthrough has regressed."
        )

    def test_instrument_channels_calls_markups_assembly_once(self, synthetic_problem):
        """Same pinning for the post-solve channels diagnostic."""
        problem = synthetic_problem
        n_calls = {'count': 0}
        orig = type(problem)._perturb_and_build_markups

        def counter(self):
            n_calls['count'] += 1
            return orig(self)

        type(problem)._perturb_and_build_markups = counter
        try:
            problem.instrument_channels(column='rival_z2', instrument=0)
        finally:
            type(problem)._perturb_and_build_markups = orig

        assert n_calls['count'] == 1, (
            f"Expected exactly 1 markups-assembly call across the whole "
            f"instrument_channels path (cached). "
            f"Received {n_calls['count']} calls — the precomputed-markups "
            f"hook in build_passthrough has regressed."
        )


class TestEdgeCases:
    """Edge-case input handling."""

    def test_one_model_raises(self):
        """Need at least 2 candidates for per-pair distances."""
        data = pyRVtest.data.load_example()
        problem = pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('1 + z1 + z2'),
            instrument_formulation=pyRVtest.Formulation('0 + rival_z1 + rival_z2'),
            models=[Bertrand(ownership='firm_ids')],
            product_data=data,
            demand_params={
                'estimate': 'logit',
                'formulation_X': pyRVtest.Formulation('1 + x1'),
                'formulation_Z': pyRVtest.Formulation('0 + z1'),
            },
        )
        with pytest.raises(ValueError, match="at least 2"):
            problem.passthrough_summary()
