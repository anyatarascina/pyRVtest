"""``Problem.solve(n_endogenous_cost=...)``: effective instrument count on the
``mc_correction`` path.

When a cost parameter is estimated in a first step outside pyRVtest and the
instruments are residualized on the fitted endogenous cost component before
being passed in, one instrument dimension is absorbed (Duarte et al. (2026),
Appendix B: ``r = d_z - d_q``). ``n_endogenous_cost`` tells the engine how many
dimensions were absorbed so the F-statistic critical-value row and the
size/power symbols use ``K - n_endogenous_cost``. ``F`` and ``TRV`` are
algebraically invariant to the count and must not move.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import pyRVtest
from tests.fixtures.tiny_synthetic import make_tiny_data


@pytest.fixture(autouse=True)
def _allow_rank_deficient_instruments():
    """The residualized instruments below are exactly collinear by design."""
    saved = (pyRVtest.options.collinear_atol, pyRVtest.options.collinear_rtol)
    pyRVtest.options.collinear_atol = pyRVtest.options.collinear_rtol = 0
    try:
        yield
    finally:
        pyRVtest.options.collinear_atol, pyRVtest.options.collinear_rtol = saved


def _problem_with_user_markups(seed: int = 0):
    df = make_tiny_data(T=30, J=6, seed=seed)
    rng = np.random.default_rng(seed)
    df['mk1'] = 0.3 * df['prices'].values + rng.normal(scale=0.02, size=len(df))
    df['mk2'] = 0.4 * df['prices'].values + rng.normal(scale=0.02, size=len(df))
    # Residualize the instruments on a fitted "endogenous cost component"
    # (a linear combination of the instruments), as the manual EOS path does:
    # the three residualized columns then have rank two.
    Z = df[['z1', 'z2', 'z3']].values
    fitted = Z @ np.array([0.5, -0.2, 0.7])
    proj = fitted[:, None] * (fitted @ Z) / (fitted @ fitted)
    Ze = Z - proj
    for j, c in enumerate(['ze1', 'ze2', 'ze3']):
        df[c] = Ze[:, j]
    pyRVtest.options.verbose = False
    return pyRVtest.Problem(
        cost_formulation=pyRVtest.Formulation('1 + x1'),
        instrument_formulation=pyRVtest.Formulation('0 + ze1 + ze2 + ze3'),
        models=(
            pyRVtest.UserSuppliedMarkups(markups='mk1', ownership='firm_ids'),
            pyRVtest.UserSuppliedMarkups(markups='mk2', ownership='firm_ids'),
        ),
        product_data=df,
        demand_results=None,
    )


def _cv_rows(K: int):
    import importlib.resources as ir
    import pandas as pd
    with ir.as_file(ir.files('pyRVtest') / 'data' / 'f_critical_values_power_rho.csv') as fp:
        power = pd.read_csv(fp)
    return power[power['K'] == K].set_index('rho')


class TestNEndogenousCost:
    def test_statistics_invariant_and_cv_row_shifts(self):
        problem = _problem_with_user_markups()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r0 = problem.solve(demand_adjustment=False, clustering_adjustment=False)
            r1 = problem.solve(demand_adjustment=False, clustering_adjustment=False, n_endogenous_cost=1)
        # F and rho are computed from sigma_m^2 = trace(.) / K_effective, which
        # cancels algebraically; only float rounding differs.
        np.testing.assert_array_equal(np.asarray(r0.TRV[0]), np.asarray(r1.TRV[0]))
        np.testing.assert_allclose(np.asarray(r0.F[0]), np.asarray(r1.F[0]), rtol=1e-12)
        np.testing.assert_allclose(np.asarray(r0.rho[0]), np.asarray(r1.rho[0]), rtol=1e-12)
        rho = round(float(abs(np.asarray(r0.rho[0])[0, 1])), 2)
        rho = min(rho, 0.99)
        cv0 = np.asarray(r0.F_cv_power_list[0][0, 1], dtype=float)
        cv1 = np.asarray(r1.F_cv_power_list[0][0, 1], dtype=float)
        exp0 = _cv_rows(3).loc[rho, ['r_50', 'r_75', 'r_95']].values.astype(float)
        exp1 = _cv_rows(2).loc[rho, ['r_50', 'r_75', 'r_95']].values.astype(float)
        np.testing.assert_allclose(cv0, exp0)
        np.testing.assert_allclose(cv1, exp1)
        assert not np.array_equal(cv0, cv1)

    def test_effective_rank_audit_is_silenced_by_the_count(self):
        problem = _problem_with_user_markups()
        with pytest.warns(UserWarning, match='residualized instruments have rank'):
            problem.solve(demand_adjustment=False)
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            problem.solve(demand_adjustment=False, n_endogenous_cost=1)

    @pytest.mark.parametrize('bad', [-1, 1.5, True, '1'])
    def test_rejects_bad_values(self, bad):
        problem = _problem_with_user_markups()
        with pytest.raises(ValueError, match='n_endogenous_cost'):
            problem.solve(demand_adjustment=False, n_endogenous_cost=bad)
