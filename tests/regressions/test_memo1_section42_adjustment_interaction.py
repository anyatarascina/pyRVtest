"""Regression test for memo 1 §4.2: `demand_adjustment=True` combined with
`endogenous_cost_component` is algebraically inconsistent.

Bug statement
-------------
When `endogenous_cost_component` is set, the per-model IV coefficient
`gamma_m` depends on theta through the markup regression (y_m = prices -
markups_effective[m](theta)), so the corrected omega depends on theta
through two channels:

    d omega[m] / d theta = P_perp([w_exog, endog_hat])
        (-d markups_effective[m] / d theta + d gamma_m / d theta * endog)

The current `_compute_demand_adjustment_gradient` captures only the first
channel. The second is missed, which biases the demand-adjusted variance
formula whenever both flags are set.

Short-term fix (memo 1 §5.1 item 2, option a)
---------------------------------------------
`_validate_solve_args` should raise `ValueError` when `demand_adjustment=True`
is combined with a non-None `endogenous_cost_component`. This gates the
inconsistent combination until a proper variance derivation is in place.

Proper fix (option b)
---------------------
Extend `_compute_demand_adjustment_gradient` to re-run
`_compute_iv_correction` inside each finite-difference perturbation, so the
derivative captures the dgamma/dtheta term. This is more work and is not
what this test pins down; the test pins the short-term gate.

Expected status
---------------
Currently marked `xfail(strict=True)`. When the gate is added, the test will
unexpectedly pass and pytest flags the marker for removal.
"""

import pytest

import pyRVtest
from tests.fixtures.tiny_synthetic import attach_user_supplied_markups


@pytest.mark.xfail(
    strict=True,
    reason=(
        "memo 1 §4.2: `demand_adjustment=True` + `endogenous_cost_component` "
        "should raise ValueError until the finite-difference loop is extended "
        "to capture dgamma/dtheta"
    )
)
def test_solve_raises_when_demand_adjustment_and_endogenous_cost_combined(
        tiny_product_data, tiny_pyblp_results):
    """Combining `demand_adjustment=True` with `endogenous_cost_component`
    must raise ValueError (short-term gate) or succeed with the correct
    variance formula (longer-term fix). Either way, it must not silently
    produce biased standard errors.
    """
    data = attach_user_supplied_markups(tiny_product_data, 'markups_a', 0.3)
    data = attach_user_supplied_markups(data, 'markups_b', 0.5)

    # user_supplied_markups disallows demand_adjustment, so use the real pyblp
    # results. For this test we want a config that exercises the gate.
    model_formulations = (
        pyRVtest.ModelFormulation(
            model_downstream='bertrand', ownership_downstream='firm_ids'),
        pyRVtest.ModelFormulation(
            model_downstream='cournot', ownership_downstream='firm_ids'),
    )
    problem = pyRVtest.Problem(
        cost_formulation=pyRVtest.Formulation('1 + x1 + shares'),
        instrument_formulation=pyRVtest.Formulation('0 + z1 + z2 + z3'),
        product_data=data,
        demand_results=tiny_pyblp_results,
        model_formulations=model_formulations,
        endogenous_cost_component='shares',
    )

    with pytest.raises(ValueError, match=r"demand_adjustment.*endogenous_cost_component"):
        problem.solve(demand_adjustment=True)
