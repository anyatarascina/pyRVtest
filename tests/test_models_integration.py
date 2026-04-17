"""Integration tests for v0.4 step 5b: Problem(models=[...]) byte-identical to
Problem(model_formulations=(ModelFormulation(...),)).

Each test builds the same DGP twice — once with the new class-based API
(Problem models=[Bertrand(...)]) and once with the legacy API
(Problem model_formulations=(ModelFormulation(model_downstream='bertrand',
...),)) — and asserts markups, TRV, F, and MCS_pvalues agree at machine
precision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyRVtest
from pyRVtest import (
    Bertrand,
    Cournot,
    CustomConductModel,
    MixCournotBertrand,
    Monopoly,
    PartialCollusion,
    PerfectCompetition,
    Vertical,
)


def _make_dgp(seed: int = 1234, T: int = 12, J: int = 4, alpha: float = -1.5):
    """Minimal plain-logit fixture with enough structure for downstream tests."""
    rng = np.random.default_rng(seed=seed)
    N = T * J
    market_ids = np.repeat(np.arange(T), J)
    firm_ids = np.tile(np.arange(J), T)
    upstream_firm_ids = np.tile(['m1', 'm1', 'm2', 'm2'], T)
    mix_flag_col = np.tile([True, True, False, False], T)
    vi_col = np.zeros(N, dtype=int)
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
    return pd.DataFrame({
        'market_ids': market_ids,
        'firm_ids': firm_ids,
        'upstream_firm_ids': upstream_firm_ids,
        'mix_flag_col': mix_flag_col,
        'vi_col': vi_col,
        'x1': x1,
        'prices': prices,
        'shares': shares,
        'rival_x1': rival_x1,
        'z1': z1,
        'intercept': np.ones(N),
    }), alpha


def _assert_same_markups(r_old, r_new, context: str):
    for m in range(len(r_old.markups)):
        np.testing.assert_allclose(
            r_new.markups[m], r_old.markups[m], atol=1e-14,
            err_msg=f"markups[{m}] diverge for {context}",
        )


def _assert_same_trv_f_mcs(r_old, r_new, context: str):
    np.testing.assert_allclose(
        r_new.TRV, r_old.TRV, atol=1e-14, equal_nan=True,
        err_msg=f"TRV diverges for {context}",
    )
    np.testing.assert_allclose(
        r_new.F, r_old.F, atol=1e-14, equal_nan=True,
        err_msg=f"F diverges for {context}",
    )
    np.testing.assert_allclose(
        r_new.MCS_pvalues, r_old.MCS_pvalues, atol=1e-14, equal_nan=True,
        err_msg=f"MCS diverges for {context}",
    )


class TestSingleConductModel:
    """Basic cases: single-tier models with plain demand_params."""

    @pytest.fixture(scope='class')
    def dgp(self):
        return _make_dgp()

    @staticmethod
    def _pc_formulation():
        return pyRVtest.ModelFormulation(model_downstream='perfect_competition')

    def test_bertrand_byte_identical(self, dgp):
        """Bertrand + PerfectCompetition baseline (M=2 needed for MCS)."""
        df, alpha = dgp
        common = dict(
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
        pyRVtest.options.verbose = False
        r_old = pyRVtest.Problem(
            **common,
            model_formulations=(
                pyRVtest.ModelFormulation(
                    model_downstream='bertrand', ownership_downstream='firm_ids',
                ),
                self._pc_formulation(),
            ),
        ).solve(demand_adjustment=False, clustering_adjustment=False)
        r_new = pyRVtest.Problem(
            **common,
            models=[Bertrand(ownership='firm_ids'), PerfectCompetition()],
        ).solve(demand_adjustment=False, clustering_adjustment=False)
        _assert_same_markups(r_old, r_new, 'Bertrand+PC')

    def test_bertrand_vs_perfect_competition_two_model_problem(self, dgp):
        df, alpha = dgp
        common = dict(
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
        pyRVtest.options.verbose = False
        r_old = pyRVtest.Problem(
            **common,
            model_formulations=(
                pyRVtest.ModelFormulation(
                    model_downstream='bertrand', ownership_downstream='firm_ids',
                ),
                pyRVtest.ModelFormulation(model_downstream='perfect_competition'),
            ),
        ).solve(demand_adjustment=True, clustering_adjustment=False)
        r_new = pyRVtest.Problem(
            **common,
            models=[
                Bertrand(ownership='firm_ids'),
                PerfectCompetition(),
            ],
        ).solve(demand_adjustment=True, clustering_adjustment=False)
        _assert_same_markups(r_old, r_new, 'Bertrand+PerfectCompetition')
        _assert_same_trv_f_mcs(r_old, r_new, 'Bertrand+PerfectCompetition')

    def _common(self, dgp):
        df, alpha = dgp
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

    def test_cournot_byte_identical(self, dgp):
        common = self._common(dgp)
        pyRVtest.options.verbose = False
        r_old = pyRVtest.Problem(
            **common,
            model_formulations=(
                pyRVtest.ModelFormulation(
                    model_downstream='cournot', ownership_downstream='firm_ids',
                ),
                self._pc_formulation(),
            ),
        ).solve(demand_adjustment=False)
        r_new = pyRVtest.Problem(
            **common,
            models=[Cournot(ownership='firm_ids'), PerfectCompetition()],
        ).solve(demand_adjustment=False)
        _assert_same_markups(r_old, r_new, 'Cournot+PC')

    def test_monopoly_byte_identical(self, dgp):
        common = self._common(dgp)
        pyRVtest.options.verbose = False
        r_old = pyRVtest.Problem(
            **common,
            model_formulations=(
                pyRVtest.ModelFormulation(
                    model_downstream='monopoly', ownership_downstream='firm_ids',
                ),
                self._pc_formulation(),
            ),
        ).solve(demand_adjustment=False)
        r_new = pyRVtest.Problem(
            **common,
            models=[Monopoly(ownership='firm_ids'), PerfectCompetition()],
        ).solve(demand_adjustment=False)
        _assert_same_markups(r_old, r_new, 'Monopoly+PC')

    def test_partial_collusion_byte_identical(self, dgp):
        common = self._common(dgp)
        pyRVtest.options.verbose = False
        r_old = pyRVtest.Problem(
            **common,
            model_formulations=(
                pyRVtest.ModelFormulation(
                    model_downstream='bertrand', ownership_downstream='firm_ids',
                    kappa_specification_downstream=lambda i, j: 0.5,
                ),
                self._pc_formulation(),
            ),
        ).solve(demand_adjustment=False)
        r_new = pyRVtest.Problem(
            **common,
            models=[
                PartialCollusion(
                    ownership='firm_ids', kappa_specification=lambda i, j: 0.5,
                ),
                PerfectCompetition(),
            ],
        ).solve(demand_adjustment=False)
        _assert_same_markups(r_old, r_new, 'PartialCollusion+PC')

    def test_mix_cournot_bertrand_byte_identical(self, dgp):
        common = self._common(dgp)
        pyRVtest.options.verbose = False
        r_old = pyRVtest.Problem(
            **common,
            model_formulations=(
                pyRVtest.ModelFormulation(
                    model_downstream='mix_cournot_bertrand',
                    ownership_downstream='firm_ids',
                    mix_flag='mix_flag_col',
                ),
                self._pc_formulation(),
            ),
        ).solve(demand_adjustment=False)
        r_new = pyRVtest.Problem(
            **common,
            models=[
                MixCournotBertrand(ownership='firm_ids', mix_flag='mix_flag_col'),
                PerfectCompetition(),
            ],
        ).solve(demand_adjustment=False)
        _assert_same_markups(r_old, r_new, 'MixCournotBertrand+PC')


class TestVerticalByteIdentical:
    """Vertical(downstream=..., upstream=..., ...) vs ModelFormulation with
    model_upstream + ownership_upstream.
    """

    def test_bertrand_downstream_monopoly_upstream(self):
        df, alpha = _make_dgp()
        common = dict(
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
        pyRVtest.options.verbose = False
        r_old = pyRVtest.Problem(
            **common,
            model_formulations=(
                pyRVtest.ModelFormulation(
                    model_downstream='bertrand', ownership_downstream='firm_ids',
                    model_upstream='monopoly', ownership_upstream='upstream_firm_ids',
                    vertical_integration='vi_col',
                ),
                pyRVtest.ModelFormulation(model_downstream='perfect_competition'),
            ),
        ).solve(demand_adjustment=False)
        r_new = pyRVtest.Problem(
            **common,
            models=[
                Vertical(
                    downstream=Bertrand(ownership='firm_ids'),
                    upstream=Monopoly(ownership='upstream_firm_ids'),
                    vertical_integration='vi_col',
                ),
                PerfectCompetition(),
            ],
        ).solve(demand_adjustment=False)
        _assert_same_markups(r_old, r_new, 'Vertical Bertrand+Monopoly')


class TestProblemKwargExclusivity:
    """Validate mutual exclusivity of models= and model_formulations=."""

    def test_both_raises(self):
        df, alpha = _make_dgp()
        pyRVtest.options.verbose = False
        with pytest.raises(TypeError, match="not both"):
            pyRVtest.Problem(
                cost_formulation=pyRVtest.Formulation('1 + z1'),
                instrument_formulation=pyRVtest.Formulation('0 + rival_x1'),
                product_data=df,
                demand_params={
                    'alpha': alpha, 'sigma': [],
                    'beta': np.array([0.0, 0.4]),
                    'x_columns': ['intercept', 'x1'],
                    'demand_instrument_columns': ['rival_x1', 'intercept', 'x1'],
                },
                model_formulations=(pyRVtest.ModelFormulation(
                    model_downstream='bertrand', ownership_downstream='firm_ids',
                ),),
                models=[Bertrand(ownership='firm_ids')],
            )

    def test_invalid_entry_type_raises(self):
        df, alpha = _make_dgp()
        pyRVtest.options.verbose = False
        with pytest.raises(TypeError, match="ConductModel, Vertical"):
            pyRVtest.Problem(
                cost_formulation=pyRVtest.Formulation('1 + z1'),
                instrument_formulation=pyRVtest.Formulation('0 + rival_x1'),
                product_data=df,
                demand_params={
                    'alpha': alpha, 'sigma': [],
                    'beta': np.array([0.0, 0.4]),
                    'x_columns': ['intercept', 'x1'],
                    'demand_instrument_columns': ['rival_x1', 'intercept', 'x1'],
                },
                models=['bertrand'],  # wrong type
            )


class TestCustomConductModelIntegration:
    """CustomConductModel (callable) round-trips through the bridge."""

    def test_custom_bertrand_equivalent_matches(self):
        """A custom callable implementing Bertrand should produce the same
        markups as the built-in Bertrand class on the same DGP.
        """
        df, alpha = _make_dgp()

        def bertrand_callable(ownership, response_matrix, shares):
            # Mirror evaluate_first_order_conditions' Bertrand branch.
            s = shares if shares.ndim > 1 else shares.reshape(-1, 1)
            return -np.linalg.solve(ownership * response_matrix.T, s)

        common = dict(
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
        pyRVtest.options.verbose = False
        r_builtin = pyRVtest.Problem(
            **common,
            models=[Bertrand(ownership='firm_ids'), PerfectCompetition()],
        ).solve(demand_adjustment=False)
        r_custom = pyRVtest.Problem(
            **common,
            models=[
                CustomConductModel(
                    markup_fn=bertrand_callable,
                    ownership='firm_ids',
                    name='bertrand_custom',
                ),
                PerfectCompetition(),
            ],
        ).solve(demand_adjustment=False)
        _assert_same_markups(r_builtin, r_custom, 'CustomConductModel == Bertrand')
