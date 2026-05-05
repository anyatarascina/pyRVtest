"""Regression test for memo 1 §4.3: `_compute_first_difference_markups`
silently zeroes out the demand-adjustment gradient when there are no
cost-side fixed effects.

Bug statement
-------------
In `problem.py::_compute_first_difference_markups`, the entire body is
gated on `if self._absorb_cost_ids is not None:`. When there are no cost
fixed effects (the default), `_absorb_cost_ids is None`, the `if` block
is skipped, and `gradient_markups[m][:, theta_index]` is never assigned.
`gradient_markups` remains at its initial value of zero for every theta
index and every model.

The consequence is that `G_k = -1/N * Z_orthogonal.T @ gradient_markups[m]`
is zero, `adjustment_value[m]` is zero, and the `(h_i - h') @
adjustment_value[m].T` correction to `psi[m]` and `phi[m]` is a no-op.

So the entire demand adjustment silently collapses to zero whenever the
user has no cost fixed effects. Both the `TRV` denominator and the F
statistic are reported as if `demand_adjustment=False` had been requested.

Pre-existing
------------
This bug is present at baseline commit `37569e3` (pre-CClean). The CClean
refactor preserved it. The monte carlo example at
`docs/notebooks/monte_carlo_example.py` uses `Formulation('1+z+shares')`
with no `absorb=`, so its `demand_adjustment=True` run is silently a no-op.

Test strategy
-------------
Run the same two-model problem twice with identical data (no cost FEs):

    A) solve(demand_adjustment=False)
    B) solve(demand_adjustment=True)

If the demand adjustment were working, B should produce different F
statistics from A (the variance correction changes the denominator). On
current code, the two are identical because `gradient_markups` stays at
zero. After the fix, they must differ.

Expected status
---------------
Currently marked `xfail(strict=True)`. When the fix lands, the test will
unexpectedly pass and pytest flags the marker for removal.
"""

import numpy as np
import pytest

import pyRVtest
from tests.fixtures.tiny_synthetic import attach_user_supplied_markups


def test_demand_adjustment_has_observable_effect_without_cost_fes(
        tiny_product_data, tiny_pyblp_results):
    """With no cost-side fixed effects, `demand_adjustment=True` must
    produce an F-statistic that differs from `demand_adjustment=False`.

    On current code they are identical because `gradient_markups` stays
    at zero. After the fix, the demand-adjusted variance differs from
    the unadjusted variance and `F` changes accordingly.
    """
    data = attach_user_supplied_markups(tiny_product_data, 'markups_a', 0.3)
    data = attach_user_supplied_markups(data, 'markups_b', 0.5)

    model_formulations = (
        pyRVtest.ModelFormulation(
            model_downstream='bertrand', ownership_downstream='firm_ids'),
        pyRVtest.ModelFormulation(
            model_downstream='cournot', ownership_downstream='firm_ids'),
    )
    # NOTE: no `absorb=` in the cost_formulation, so `_absorb_cost_ids is None`
    problem = pyRVtest.Problem(
        cost_formulation=pyRVtest.Formulation('1 + x1'),
        instrument_formulation=pyRVtest.Formulation('0 + z1 + z2 + z3'),
        product_data=data,
        demand_results=tiny_pyblp_results,
        model_formulations=model_formulations,
    )

    results_unadjusted = problem.solve(demand_adjustment=False)
    results_adjusted = problem.solve(demand_adjustment=True)

    F_unadjusted = results_unadjusted.F[0][0, 1]
    F_adjusted = results_adjusted.F[0][0, 1]

    assert np.isfinite(F_unadjusted) and np.isfinite(F_adjusted), (
        f"F-stats not finite. F_unadjusted={F_unadjusted}, "
        f"F_adjusted={F_adjusted}."
    )

    # Absolute difference should be non-trivial; allow a small numeric tolerance
    # but not zero. On current code, F_unadjusted == F_adjusted exactly.
    assert not np.isclose(F_unadjusted, F_adjusted, rtol=1e-10, atol=1e-10), (
        f"`demand_adjustment=True` produced the same F-statistic as "
        f"`demand_adjustment=False` (F={F_unadjusted:.8f}). This is the "
        f"zero-gradient bug from memo 1 §4.3: with no cost fixed effects, "
        f"`_compute_first_difference_markups` leaves `gradient_markups` at "
        f"zero and the variance correction collapses to a no-op."
    )


def test_TRV_denominator_changes_with_demand_adjustment_without_cost_fes(
        tiny_product_data, tiny_pyblp_results):
    """The RV test-statistic denominator must respond to `demand_adjustment=True`.

    The denominator is built from the psi variance; with a nonzero gradient
    the psi is shifted, so the denominator changes. On current code without
    cost FEs, psi is unchanged and the denominator is the same.
    """
    data = attach_user_supplied_markups(tiny_product_data, 'markups_a', 0.3)
    data = attach_user_supplied_markups(data, 'markups_b', 0.5)

    model_formulations = (
        pyRVtest.ModelFormulation(
            model_downstream='bertrand', ownership_downstream='firm_ids'),
        pyRVtest.ModelFormulation(
            model_downstream='cournot', ownership_downstream='firm_ids'),
    )
    problem = pyRVtest.Problem(
        cost_formulation=pyRVtest.Formulation('1 + x1'),
        instrument_formulation=pyRVtest.Formulation('0 + z1 + z2 + z3'),
        product_data=data,
        demand_results=tiny_pyblp_results,
        model_formulations=model_formulations,
    )

    results_unadjusted = problem.solve(demand_adjustment=False)
    results_adjusted = problem.solve(demand_adjustment=True)

    denom_unadjusted = results_unadjusted.RV_denominator[0][0, 1]
    denom_adjusted = results_adjusted.RV_denominator[0][0, 1]

    assert not np.isclose(
        denom_unadjusted, denom_adjusted, rtol=1e-10, atol=1e-10
    ), (
        f"RV denominator unchanged between demand_adjustment=False and "
        f"demand_adjustment=True (denom={denom_unadjusted:.8f}). This is "
        f"the same §4.3 zero-gradient bug observed through the TRV pathway."
    )
