"""Regression test for memo 1 §4.1: F-diagnostic must be rank-adjusted when
`endogenous_cost_component` is set.

Bug statement
-------------
`_compute_instrument_results` in `pyRVtest/problem.py` uses `K = Z.shape[1]`
unchanged after `Z_orthogonal` has been residualized on `[w_exog, endog_hat]`.
That joint residualization reduces the rank of `Z_orthogonal` by 1 relative
to residualizing on `w_exog` alone, so the effective number of identifying
restrictions is `K - 1`. Both the `N / (2 K)` scaling of `unscaled_F` and the
`K_lookup` used for the critical-values lookup should use `K - 1` on the
non-constant-cost path.

Test strategy
-------------
Run the same two-model problem twice with identical corrected marginal costs:

    Path A: `endogenous_cost_component='shares'` (the package runs the IV
            correction internally and residualizes Z on [w_exog, endog_hat]).
    Path B: `mc_correction` passed in as a hand-computed array matching what
            Path A produces internally. Path B does NOT residualize Z on
            `endog_hat`, so its effective K equals the raw K.

If Path A and Path B produce the same `omega` (they should, modulo a rank-1
projection), the F-statistic from Path A should exceed Path B's by a factor
of `K / (K - 1)` once the fix is in, and the `K_lookup` used for CV retrieval
should differ by one between the two paths.

Current code produces identical F-stats in the two paths (both use raw K).
After the fix, `F_A / F_B` equals `K / (K - 1)` (approximately) and the
size/power annotations in the two paths read different rows of the CV table.

Expected status
---------------
Currently marked `xfail(strict=True)`. When the fix lands, the test will
unexpectedly pass and pytest flags the marker for removal.
"""

import numpy as np
import pandas as pd
import pytest

import pyRVtest
from tests.fixtures.tiny_synthetic import attach_user_supplied_markups


def _build_problem(data, *, endogenous_cost_component=None):
    """Construct a two-model user-supplied-markup problem.

    Using user-supplied markups avoids running PyBLP and focuses the test
    on the testing shell's handling of `endogenous_cost_component`.
    """
    model_formulations = (
        pyRVtest.ModelFormulation(user_supplied_markups='markups_a'),
        pyRVtest.ModelFormulation(user_supplied_markups='markups_b'),
    )
    return pyRVtest.Problem(
        cost_formulation=pyRVtest.Formulation('1 + x1 + shares'),
        instrument_formulation=pyRVtest.Formulation('0 + z1 + z2 + z3'),
        product_data=data,
        demand_results=None,
        model_formulations=model_formulations,
        endogenous_cost_component=endogenous_cost_component,
    )


@pytest.mark.xfail(
    strict=True,
    reason="memo 1 §4.1: F-diagnostic not yet rank-adjusted for endogenous_cost_component"
)
def test_F_statistic_uses_K_minus_one_when_endogenous_cost_component_is_set(
        tiny_product_data):
    """F-statistic scaling must use `K - 1` effective instruments under the
    non-constant-cost path, not raw `K`.

    After the fix, Path A (endogenous_cost_component) produces an F-statistic
    larger than Path B (manual mc_correction) by a factor of K / (K - 1).
    """
    data = attach_user_supplied_markups(tiny_product_data, 'markups_a', 0.3)
    data = attach_user_supplied_markups(data, 'markups_b', 0.5)

    # Path A: use the package's IV correction
    problem_A = _build_problem(data, endogenous_cost_component='shares')
    results_A = problem_A.solve()

    # Path B: run the same IV correction by hand and feed it via mc_correction
    # Exogenous cost shifters are [1, x1]; endog is shares; Z = [z1, z2, z3]
    N = len(data)
    ones = np.ones(N)
    exog_w = np.column_stack([ones, data['x1'].to_numpy()])
    endog_col = data['shares'].to_numpy().reshape(-1, 1)
    Z_inst = data[['z1', 'z2', 'z3']].to_numpy()

    # First stage: project shares on [exog_w, Z_inst]
    first_stage_X = np.hstack([exog_w, Z_inst])
    Q_fs, _ = np.linalg.qr(first_stage_X, mode='reduced')
    endog_hat = Q_fs @ (Q_fs.T @ endog_col)

    # Second stage: replace shares with fitted values, 2SLS per model
    X_2sls = np.hstack([exog_w, endog_hat])
    Q_2sls, R_2sls = np.linalg.qr(X_2sls, mode='reduced')

    mc_correction = []
    for col in ('markups_a', 'markups_b'):
        markups_m = -data[col].to_numpy().reshape(-1, 1)  # price - markup sign
        y_m = data['prices'].to_numpy().reshape(-1, 1) + markups_m.reshape(-1, 1) * 0 - \
              (data['prices'].to_numpy().reshape(-1, 1) - (-markups_m))
        # Simpler: marginal_cost = prices - user_markup = prices - (-mult * price)
        mc_m = data['prices'].to_numpy().reshape(-1, 1) - data[col].to_numpy().reshape(-1, 1) * (-1)
        params = np.linalg.solve(R_2sls, Q_2sls.T @ mc_m)
        gamma_m = params[-1, 0]
        mc_correction.append(-gamma_m * endog_col)
    mc_correction = np.array(mc_correction).reshape(2, N, 1)

    problem_B = _build_problem(data, endogenous_cost_component=None)
    results_B = problem_B.solve(mc_correction=mc_correction)

    # K = 3 (z1, z2, z3)
    K = 3
    expected_ratio = K / (K - 1)

    F_A = results_A.F[0][0, 1]
    F_B = results_B.F[0][0, 1]

    assert np.isfinite(F_A) and np.isfinite(F_B), (
        f"F-stats are not finite. F_A={F_A}, F_B={F_B}. Something upstream broke."
    )
    actual_ratio = F_A / F_B
    assert actual_ratio == pytest.approx(expected_ratio, rel=1e-3), (
        f"Expected F_A / F_B = K / (K-1) = {expected_ratio:.4f}, "
        f"got {actual_ratio:.4f}. F_A={F_A:.6f}, F_B={F_B:.6f}. "
        f"This suggests the F-diagnostic is still using raw K when "
        f"endogenous_cost_component is set."
    )


@pytest.mark.xfail(
    strict=True,
    reason="memo 1 §4.1: K_lookup not yet rank-adjusted for endogenous_cost_component"
)
def test_F_cv_lookup_uses_K_minus_one_when_endogenous_cost_component_is_set(
        tiny_product_data):
    """Critical-value row lookup must use `K - 1` when endogenous_cost_component is set.

    With K = 3 instruments and no endogenous_cost_component, the CV table is
    read at K=3. With endogenous_cost_component, it should be read at K=2.
    Since the two reads produce different numerical critical values (the CV
    table is monotonic in K), the F_cv_size_list arrays should differ.
    """
    data = attach_user_supplied_markups(tiny_product_data, 'markups_a', 0.3)
    data = attach_user_supplied_markups(data, 'markups_b', 0.5)

    problem_with = _build_problem(data, endogenous_cost_component='shares')
    problem_without = _build_problem(data, endogenous_cost_component=None)

    results_with = problem_with.solve()
    results_without = problem_without.solve()

    cv_with = results_with.F_cv_size_list[0][0, 1]
    cv_without = results_without.F_cv_size_list[0][0, 1]

    # At least one of the three critical values (r_125, r_10, r_075) must differ
    differs = any(
        not np.isclose(float(a), float(b))
        for a, b in zip(cv_with, cv_without)
    )
    assert differs, (
        f"F_cv_size_list is identical between endogenous_cost_component=None "
        f"and ='shares' paths. cv_with={cv_with}, cv_without={cv_without}. "
        f"Expected the two to read different rows of the CV table "
        f"(K=2 vs K=3)."
    )
