"""DMQSW Phase 3: per-pair channel decomposition for one IV column.

Exercises :meth:`pyRVtest.Problem.instrument_channels` and
:meth:`pyRVtest.ProblemResults.instrument_channels` on the shipped 2-firm
3000-market synthetic example, verifying:

1. The result reports four named components: data-side ``dp_0/dz_obs``,
   per-candidate direct ``β_m``, per-pair ``structural`` (γ-free
   inverse-pass-through difference), and per-pair ``direct`` (empirical
   ``|β_m − β_m'|``).
2. PerfectCompetition's direct ``β_m`` is exactly zero (``Δ_PC ≡ 0``).
3. Per-pair structural distances are non-negative and reflect the
   candidate's inverse-pass-through differences.
4. ProblemResults.instrument_channels matches Problem.instrument_channels.
5. The methodology line records the instrument type and the empirical
   nature of the direct-channel regression.
6. Edge cases: invalid column name raises a clear error; a Problem with
   only one model raises.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyRVtest
from pyRVtest import Bertrand, Cournot, Monopoly, PerfectCompetition
from pyRVtest.solve.passthrough import (
    InstrumentChannels,
    compute_instrument_channels,
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


class TestStructuredOutput:
    """Result has the expected attributes and DataFrame shapes."""

    def test_returns_instrument_channels_object(self, synthetic_problem):
        result = synthetic_problem.instrument_channels(column='rival_z2')
        assert isinstance(result, InstrumentChannels)

    def test_data_side_attributes(self, synthetic_problem):
        result = synthetic_problem.instrument_channels(column='rival_z2')
        assert isinstance(result.dp0_dz_obs, float)
        assert result.sd_z > 0
        assert result.z_min < result.z_max

    def test_per_candidate_has_one_row_per_model(self, synthetic_problem):
        result = synthetic_problem.instrument_channels(column='rival_z2')
        assert len(result.per_candidate) == 4
        for col in ('model', 'beta_m'):
            assert col in result.per_candidate.columns

    def test_pair_distances_has_one_row_per_pair(self, synthetic_problem):
        result = synthetic_problem.instrument_channels(column='rival_z2')
        # 4 candidates → 4 choose 2 = 6 pairs.
        assert len(result.pair_distances) == 6
        for col in ('pair', 'structural', 'direct'):
            assert col in result.pair_distances.columns

    def test_to_dataframe_returns_copy(self, synthetic_problem):
        result = synthetic_problem.instrument_channels(column='rival_z2')
        df = result.to_dataframe()
        df['structural'] = 999
        assert (result.pair_distances['structural'] != 999).all()


class TestPerfectCompetitionDirectChannel:
    """PC has Δ = 0 always, so β_m is exactly zero (no empirical noise)."""

    def test_pc_beta_m_is_exactly_zero(self, synthetic_problem):
        result = synthetic_problem.instrument_channels(column='rival_z2')
        pc_row = result.per_candidate[
            result.per_candidate['model'] == 'PerfectCompetition'
        ]
        assert len(pc_row) == 1
        np.testing.assert_array_equal(float(pc_row['beta_m'].iloc[0]), 0.0)


class TestStructuralComponent:
    """Per-pair structural distances are non-negative and reflect
    inverse-pass-through differences."""

    def test_structural_distances_nonnegative(self, synthetic_problem):
        result = synthetic_problem.instrument_channels(column='rival_z2')
        assert (result.pair_distances['structural'] >= 0).all()

    def test_pc_self_pair_excluded(self, synthetic_problem):
        """The diagonal isn't computed: only i < j pairs."""
        result = synthetic_problem.instrument_channels(column='rival_z2')
        # No self-pair like (Bertrand, Bertrand).
        for pair_label in result.pair_distances['pair']:
            label_inside = pair_label.strip('()').split(', ')
            assert label_inside[0] != label_inside[1]


class TestProblemResultsWrapper:
    """ProblemResults.instrument_channels delegates to Problem."""

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

    def test_results_method_returns_same_pair_distances(
        self, synthetic_results, synthetic_problem,
    ):
        problem_result = synthetic_problem.instrument_channels(column='rival_z2')
        results_result = synthetic_results.instrument_channels(column='rival_z2')
        pd.testing.assert_frame_equal(
            problem_result.pair_distances.sort_values(['model_i', 'model_j']).reset_index(drop=True),
            results_result.pair_distances.sort_values(['model_i', 'model_j']).reset_index(drop=True),
        )


class TestMethodologyLine:
    """Methodology line covers pass-through computation and
    regression-based components."""

    def test_methodology_includes_pass_through_method(self, synthetic_problem):
        result = synthetic_problem.instrument_channels(column='rival_z2')
        assert 'central-difference numerical' in result.methodology_line
        assert 'short-circuit' in result.methodology_line

    def test_methodology_describes_regression_components(self, synthetic_problem):
        result = synthetic_problem.instrument_channels(column='rival_z2')
        line = result.methodology_line
        assert 'conditional regression' in line
        assert 'cost-formulation controls' in line
        assert "P_m^{-1} − P_m'^{-1}" in line

    def test_declared_type_appears_in_methodology(self, synthetic_problem):
        result = synthetic_problem.instrument_channels(
            column='rival_z2', instrument='rival_cost',
        )
        assert "'rival_cost'" in result.methodology_line

    def test_no_declared_type_omits_targeting_note(self, synthetic_problem):
        result = synthetic_problem.instrument_channels(column='rival_z2')
        # When no instrument= kwarg, methodology line shouldn't include
        # the "Declared instrument type" sub-note.
        assert "Declared instrument type" not in result.methodology_line


class TestRepr:
    """``__repr__`` produces a readable formatted view."""

    def test_repr_includes_data_side_block(self, synthetic_problem):
        result = synthetic_problem.instrument_channels(column='rival_z2')
        rendered = repr(result)
        assert 'Data-side' in rendered
        assert '‖dp_0/dz‖_obs' in rendered

    def test_repr_includes_per_candidate_block(self, synthetic_problem):
        result = synthetic_problem.instrument_channels(column='rival_z2')
        rendered = repr(result)
        assert 'Direct channel' in rendered
        assert 'beta_m' in rendered

    def test_repr_includes_pair_table(self, synthetic_problem):
        result = synthetic_problem.instrument_channels(column='rival_z2')
        rendered = repr(result)
        assert 'Per-pair channel components' in rendered
        assert 'structural' in rendered

    def test_repr_includes_methodology(self, synthetic_problem):
        result = synthetic_problem.instrument_channels(column='rival_z2')
        rendered = repr(result)
        assert 'Methodology' in rendered

    def test_repr_with_declared_type_shown_in_header(self, synthetic_problem):
        result = synthetic_problem.instrument_channels(
            column='rival_z2', instrument='rival_cost',
        )
        rendered = repr(result)
        assert "declared type: 'rival_cost'" in rendered


class TestEdgeCases:

    def test_invalid_column_raises_with_helpful_message(self, synthetic_problem):
        with pytest.raises(ValueError, match="not_a_column"):
            synthetic_problem.instrument_channels(column='not_a_column')

    def test_one_model_raises(self):
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
            problem.instrument_channels(column='rival_z2')


class TestDataSideMatchesIndependentCalculation:
    """The data-side ``dp_0/dz_obs`` matches an independent OLS
    computation."""

    def test_dp0_dz_obs_matches_manual_regression(self, synthetic_problem):
        result = synthetic_problem.instrument_channels(column='rival_z2')

        # Reproduce the regression independently: prices on rival_z2 with
        # 1 + z1 + z2 (cost formulation) as controls.
        z = np.asarray(synthetic_problem.products['rival_z2'], dtype=float).flatten()
        prices = np.asarray(synthetic_problem.products['prices'], dtype=float).flatten()
        # Cost formulation columns: const + z1 + z2.
        z1 = np.asarray(synthetic_problem.products['z1'], dtype=float).flatten()
        z2 = np.asarray(synthetic_problem.products['z2'], dtype=float).flatten()
        n = len(z)
        X = np.column_stack([np.ones(n), z1, z2])
        # FWL: residualize prices and z on X, then take simple slope.
        beta_p, *_ = np.linalg.lstsq(X, prices, rcond=None)
        p_resid = prices - X @ beta_p
        beta_z, *_ = np.linalg.lstsq(X, z, rcond=None)
        z_resid = z - X @ beta_z
        manual_slope = np.cov(p_resid, z_resid, ddof=0)[0, 1] / np.var(z_resid)

        np.testing.assert_allclose(
            result.dp0_dz_obs, manual_slope, atol=1e-10,
        )


class TestZeResidualizationUnderNonConstantMC:
    """Phase 3: when ``endogenous_cost_component`` is set, the data-side
    regression and FWL partialling use ``(g(q_tilde), w_exog)`` instead
    of raw ``(q, w)``. This is DMQSS Appendix B's z^e residualization
    and collapses the Dearing condition + A.4 distinctness check into a
    single unified diagnostic.

    Constant-MC behavior (``endogenous_cost_component is None``) is
    unchanged and covered by the rest of this module — these tests
    only exercise the new non-constant-MC branch.
    """

    @pytest.fixture(scope='class')
    def endog_problem(self):
        data = pyRVtest.data.load_example().copy()
        data['log_q'] = np.log(np.maximum(data['shares'] * 1000.0, 1e-3))
        return pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('1 + z1 + log_q'),
            instrument_formulation=pyRVtest.Formulation('0 + rival_z1 + rival_z2'),
            models=[
                Bertrand(ownership='firm_ids'),
                PerfectCompetition(),
            ],
            product_data=data,
            demand_params={
                'estimate': 'logit',
                'formulation_X': pyRVtest.Formulation('1 + x1'),
                'formulation_Z': pyRVtest.Formulation('0 + z1'),
            },
            endogenous_cost_component='log_q',
        )

    def test_dp0_dz_uses_q_tilde_residualization(self, endog_problem):
        """Data-side dp_0/dz_obs differs from the raw-(q, w) projection
        when endog is set. The package projects on (q_tilde, w_exog),
        where q_tilde uses the FULL declared instrument set (rival_z1
        + rival_z2 here) for the first stage — not just the diagnostic
        column.
        """
        from pyRVtest.solve.passthrough import _ols_slope_partialled

        result = endog_problem.instrument_channels(column='rival_z2')

        # Reproduce the package's z^e regression independently.
        z = np.asarray(endog_problem.products['rival_z2'], dtype=float).flatten()
        prices = np.asarray(endog_problem.products['prices'], dtype=float).flatten()
        z1 = np.asarray(endog_problem.products['z1'], dtype=float).flatten()
        rival_z1 = np.asarray(endog_problem.products['rival_z1'], dtype=float).flatten()
        rival_z2 = np.asarray(endog_problem.products['rival_z2'], dtype=float).flatten()
        log_q = np.asarray(endog_problem.products['log_q'], dtype=float).flatten()
        n = len(z)
        ones = np.ones(n)

        # First stage uses the FULL test IV bundle (rival_z1 + rival_z2)
        # plus w_exog (intercept, z1) — NOT just the diagnostic column.
        first_stage_X = np.column_stack([rival_z1, rival_z2, ones, z1])
        beta_fs, *_ = np.linalg.lstsq(first_stage_X, log_q, rcond=None)
        log_q_tilde = first_stage_X @ beta_fs

        # Second stage: pass (q_tilde, w_exog) as controls to the package's
        # _ols_slope_partialled helper (which adds an implicit intercept).
        controls_for_zE = np.column_stack([log_q_tilde, ones, z1])
        manual_zE_slope = _ols_slope_partialled(prices, z, controls_for_zE)

        # Package's slope should match.
        np.testing.assert_allclose(result.dp0_dz_obs, manual_zE_slope, atol=1e-8)

        # And it should DIFFER from the raw-(q, w) projection that the
        # constant-MC branch would have used.
        controls_raw = np.column_stack([log_q, ones, z1])
        raw_slope = _ols_slope_partialled(prices, z, controls_raw)

        # The two should differ; on this fixture they genuinely do.
        assert not np.isclose(manual_zE_slope, raw_slope, atol=1e-8), (
            "z^e and raw-q residualization produced identical data-side "
            "slopes; the z^e generalization may not be active."
        )

    def test_methodology_line_mentions_z_e(self, endog_problem):
        """The methodology footer documents the z^e residualization
        when endog is set.
        """
        result = endog_problem.instrument_channels(column='rival_z2')
        line = result.methodology_line
        assert 'z^e' in line or 'DMQSS' in line, (
            f"Expected methodology footer to mention z^e or DMQSS when "
            f"endogenous_cost_component is set; got: {line!r}"
        )

    def test_methodology_line_unchanged_for_constant_mc(self, synthetic_problem):
        """The constant-MC methodology footer keeps its prior wording."""
        result = synthetic_problem.instrument_channels(column='rival_z2')
        line = result.methodology_line
        # Constant-MC footer mentions the standard FWL / cost-formulation
        # controls language and should NOT mention z^e or DMQSS A.4.
        assert 'cost-formulation controls' in line
        assert 'z^e' not in line
        assert 'A.4' not in line
