"""v0.4 step 5e: class-based API vs legacy API on every Step 0 snapshot fixture.

The existing snapshot tests in tests/test_snapshots.py pin expected output
for six scenarios using the legacy ``model_formulations=`` API. If the new
``models=`` path produces byte-identical output on each of these fixtures,
then every Step 0 snapshot is also matched by the class-based API — closing
the v0.4 step 5 loop.

This test runs each scenario twice (legacy, new) and asserts markups, TRV,
F, MCS_pvalues, endogenous_cost_coefficient match at machine precision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyRVtest
from tests.test_analytical import _build_base_dgp, _build_scale_dgp


def _solve_both_ways(common, legacy_formulations, new_models, solve_kwargs):
    """Build two Problems (legacy vs new) and solve with identical kwargs."""
    pyRVtest.options.verbose = False
    r_legacy = pyRVtest.Problem(
        **common, model_formulations=legacy_formulations,
    ).solve(**solve_kwargs)
    r_new = pyRVtest.Problem(
        **common, models=new_models,
    ).solve(**solve_kwargs)
    return r_legacy, r_new


def _assert_full_parity(r_legacy, r_new, scenario: str, atol: float = 1e-14):
    """Assert markups, TRV, F, MCS_pvalues, endogenous_cost_coefficient match byte-identically."""
    for m in range(len(r_legacy.markups)):
        np.testing.assert_allclose(
            r_new.markups[m], r_legacy.markups[m], atol=atol,
            err_msg=f"[{scenario}] markups[{m}] diverge",
        )
    np.testing.assert_allclose(
        r_new.TRV, r_legacy.TRV, atol=atol, equal_nan=True,
        err_msg=f"[{scenario}] TRV diverges",
    )
    np.testing.assert_allclose(
        r_new.F, r_legacy.F, atol=atol, equal_nan=True,
        err_msg=f"[{scenario}] F diverges",
    )
    np.testing.assert_allclose(
        r_new.MCS_pvalues, r_legacy.MCS_pvalues, atol=atol, equal_nan=True,
        err_msg=f"[{scenario}] MCS_pvalues diverges",
    )
    legacy_ecc = r_legacy.endogenous_cost_coefficient
    new_ecc = r_new.endogenous_cost_coefficient
    if legacy_ecc is not None and new_ecc is not None:
        # endogenous_cost_coefficient is shape (L, M) — gamma per instrument set and model.
        np.testing.assert_allclose(
            new_ecc, legacy_ecc, atol=atol,
            err_msg=f"[{scenario}] endogenous_cost_coefficient diverges",
        )


# ---------------------------------------------------------------------------
# Scenario 1 + 2: analytical_base + analytical_clustering
# ---------------------------------------------------------------------------

def _common_base(product_data):
    return dict(
        cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
        instrument_formulation=pyRVtest.Formulation('0 + iv0 + iv1 + iv2'),
        product_data=product_data,
        demand_results=None,
    )


def _base_legacy_formulations():
    return (
        pyRVtest.ModelFormulation(
            model_downstream='bertrand', ownership_downstream='firm_ids',
            user_supplied_markups='markups_m1',
        ),
        pyRVtest.ModelFormulation(
            model_downstream='perfect_competition',
            user_supplied_markups='markups_m2',
        ),
    )


def _base_new_models():
    return [
        pyRVtest.Bertrand(
            ownership='firm_ids', user_supplied_markups='markups_m1',
        ),
        pyRVtest.PerfectCompetition(user_supplied_markups='markups_m2'),
    ]


@pytest.mark.parametrize("clustering", [False, True], ids=["base", "clustering"])
def test_analytical_base_and_clustering(clustering):
    product_data, _ = _build_base_dgp()
    r_legacy, r_new = _solve_both_ways(
        _common_base(product_data),
        _base_legacy_formulations(),
        _base_new_models(),
        dict(demand_adjustment=False, clustering_adjustment=clustering),
    )
    scenario = 'analytical_clustering' if clustering else 'analytical_base'
    _assert_full_parity(r_legacy, r_new, scenario)


# ---------------------------------------------------------------------------
# Scenario 3: analytical_base_fe (cost formulation with absorb='C(firm_ids)')
# ---------------------------------------------------------------------------

def test_analytical_base_fe():
    product_data, _ = _build_base_dgp()
    common = dict(
        cost_formulation=pyRVtest.Formulation('0 + cost_shifter', absorb='C(firm_ids)'),
        instrument_formulation=pyRVtest.Formulation('0 + iv0 + iv1 + iv2'),
        product_data=product_data,
        demand_results=None,
    )
    r_legacy, r_new = _solve_both_ways(
        common,
        _base_legacy_formulations(),
        _base_new_models(),
        dict(demand_adjustment=False, clustering_adjustment=False),
    )
    _assert_full_parity(r_legacy, r_new, 'analytical_base_fe')


# ---------------------------------------------------------------------------
# Scenario 4: analytical_scale (endogenous_cost_component='log_quantity')
# ---------------------------------------------------------------------------

def test_analytical_scale():
    product_data, _ = _build_scale_dgp()
    common = dict(
        cost_formulation=pyRVtest.Formulation('1 + cost_shifter + log_quantity'),
        instrument_formulation=pyRVtest.Formulation('0 + iv1 + iv2'),
        product_data=product_data,
        demand_results=None,
        endogenous_cost_component='log_quantity',
    )
    legacy = (
        pyRVtest.ModelFormulation(
            model_downstream='bertrand', ownership_downstream='firm_ids',
            user_supplied_markups='markups_m1',
        ),
        pyRVtest.ModelFormulation(
            model_downstream='perfect_competition',
            user_supplied_markups='markups_m2',
        ),
    )
    new = [
        pyRVtest.Bertrand(
            ownership='firm_ids', user_supplied_markups='markups_m1',
        ),
        pyRVtest.PerfectCompetition(user_supplied_markups='markups_m2'),
    ]
    r_legacy, r_new = _solve_both_ways(
        common, legacy, new,
        dict(demand_adjustment=False, clustering_adjustment=False),
    )
    _assert_full_parity(r_legacy, r_new, 'analytical_scale')


# ---------------------------------------------------------------------------
# Scenarios 5 + 6: first_stage_pyblp_path and first_stage_demand_params_path
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def first_stage_fixture():
    """Replicate the first_stage snapshot DGP (pyblp-estimated logit + pyRVtest)."""
    import pyblp
    pyblp.options.verbose = False
    T, J = 200, 3
    market_ids = np.repeat(np.arange(T), J)
    firm_ids = np.tile(np.arange(J), T)
    id_data = pd.DataFrame({"market_ids": market_ids, "firm_ids": firm_ids})
    X1 = pyblp.Formulation("1 + prices + x1")
    X3 = pyblp.Formulation("1 + z1")
    simulation = pyblp.Simulation(
        product_formulations=(X1, None, X3),
        beta=[0, -2, 1], gamma=[1, 0.5],
        xi_variance=0.3, omega_variance=0.2, correlation=0.0,
        product_data=id_data, seed=99,
    )
    sim_results = simulation.replace_endogenous()
    data = pd.DataFrame(pyblp.data_to_dict(sim_results.product_data))
    for t in range(T):
        idx = np.where(data["market_ids"] == t)[0]
        for j in idx:
            rival = [i for i in idx if i != j]
            data.loc[j, "rival_x1"] = data.loc[rival, "x1"].mean()
            data.loc[j, "rival_z1"] = data.loc[rival, "z1"].mean()
    data["rival_z1_sq"] = data["rival_z1"] ** 2
    data["intercept"] = 1.0
    data["demand_instruments0"] = data["rival_x1"]
    data["demand_instruments1"] = data["rival_z1"] ** 2
    problem = pyblp.Problem((X1,), product_data=data)
    pyblp_results = problem.solve(method="1s")
    alpha_hat = float(pyblp_results.beta[pyblp_results.beta_labels.index("prices")].item())
    beta_x_hat = float(pyblp_results.beta[pyblp_results.beta_labels.index("x1")].item())
    beta_0_hat = float(pyblp_results.beta[pyblp_results.beta_labels.index("1")].item())
    return data, pyblp_results, alpha_hat, beta_x_hat, beta_0_hat


def _first_stage_legacy():
    return (
        pyRVtest.ModelFormulation(
            model_downstream="bertrand", ownership_downstream="firm_ids",
        ),
        pyRVtest.ModelFormulation(model_downstream="perfect_competition"),
    )


def _first_stage_new():
    return [
        pyRVtest.Bertrand(ownership="firm_ids"),
        pyRVtest.PerfectCompetition(),
    ]


def test_first_stage_pyblp_path(first_stage_fixture):
    data, pyblp_results, _, _, _ = first_stage_fixture
    common = dict(
        cost_formulation=pyRVtest.Formulation("1 + z1"),
        instrument_formulation=pyRVtest.Formulation("0 + rival_x1 + rival_z1"),
        product_data=data,
        demand_results=pyblp_results,
    )
    r_legacy, r_new = _solve_both_ways(
        common, _first_stage_legacy(), _first_stage_new(),
        dict(demand_adjustment=True, clustering_adjustment=False),
    )
    _assert_full_parity(r_legacy, r_new, 'first_stage_pyblp_path')


def test_first_stage_demand_params_path(first_stage_fixture):
    data, _, alpha_hat, beta_x_hat, beta_0_hat = first_stage_fixture
    common = dict(
        cost_formulation=pyRVtest.Formulation("1 + z1"),
        instrument_formulation=pyRVtest.Formulation("0 + rival_x1 + rival_z1"),
        product_data=data,
        demand_params={
            "alpha": alpha_hat, "sigma": [],
            "beta": np.array([beta_0_hat, beta_x_hat]),
            "x_columns": ["intercept", "x1"],
            "demand_instrument_columns": [
                "rival_x1", "rival_z1_sq", "intercept", "x1",
            ],
        },
    )
    r_legacy, r_new = _solve_both_ways(
        common, _first_stage_legacy(), _first_stage_new(),
        dict(demand_adjustment=True, clustering_adjustment=False),
    )
    _assert_full_parity(r_legacy, r_new, 'first_stage_demand_params_path')
