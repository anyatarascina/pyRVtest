"""v0.4 Step 0b: snapshot regression suite.

Snapshots the numerical output of several pinned DGPs' `.solve()` calls
to `tests/snapshots/*.json`. Every migration step in the v0.4 refactor
must either reproduce these to atol=1e-10 or document a deliberate
change per the decision rule in .claude/plans/v0.4-refactor.md §5:

  - delta <= 1e-12          auto-update (numerical noise)
  - 1e-12 < delta <= 1e-7   requires deliberate-source commit message
  - delta > 1e-7            BLOCKS MERGE; investigate root cause

Regenerate snapshots by running:

    REGENERATE_SNAPSHOTS=1 python3 -m pytest tests/test_snapshots.py

and committing the new JSON files with a message explaining the source.

Coverage (as of initial landing):

  - snap_analytical_base:        user_supplied_markups, Bertrand vs
                                 perfect-comp, no demand adjustment,
                                 no clustering. Base DGP seed=12345.
  - snap_analytical_clustering:  same DGP, clustering_adjustment=True.
  - snap_analytical_scale:       endogenous_cost_component path with
                                 scale economies. Seed=54321.
  - snap_analytical_scale_fe:    base DGP with cost fixed effects
                                 (absorb='C(firm_ids)').

The demand_params + demand-adjustment snapshots (which exercise the
b3b08a3 bug-fix paths) will land in a follow-up commit once Step 0d's
golden file is pinned. For now, the first-stage-correction equivalence
test (Step 0c, already DONE) covers those paths on its own DGP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyRVtest
import pytest

from ._snapshot_helpers import assert_snapshot
from .test_analytical import (
    _build_base_dgp,
    _build_scale_dgp,
    _run_pyrvtest_base,
)


# ---------------------------------------------------------------------------
# Base analytical DGP (seed=12345): the classic Bertrand-logit with
# user_supplied_markups. Covers the markups-assembly -> orthogonalize ->
# GMM -> RV test pipeline without demand-parameter integration.
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def base_results():
    product_data, _ = _build_base_dgp()
    return _run_pyrvtest_base(product_data, clustering=False)


@pytest.fixture(scope='module')
def clustering_results():
    product_data, _ = _build_base_dgp()
    return _run_pyrvtest_base(product_data, clustering=True)


def test_snapshot_analytical_base(base_results):
    """Snapshot: base Bertrand DGP, no clustering, no demand adjustment."""
    assert_snapshot('analytical_base', base_results)


def test_snapshot_analytical_clustering(clustering_results):
    """Snapshot: base DGP with clustering_adjustment=True."""
    assert_snapshot('analytical_clustering', clustering_results)


# ---------------------------------------------------------------------------
# Base DGP with cost fixed effects (absorb='C(firm_ids)'). This exercises
# the QR residualization path with a categorical absorb.
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def base_fe_results():
    product_data, _ = _build_base_dgp()
    model_formulations = (
        pyRVtest.ModelFormulation(
            model_downstream='bertrand', ownership_downstream='firm_ids',
            user_supplied_markups='markups_m1',
        ),
        pyRVtest.ModelFormulation(
            model_downstream='perfect_competition',
            user_supplied_markups='markups_m2',
        ),
    )
    testing_problem = pyRVtest.Problem(
        cost_formulation=pyRVtest.Formulation('0 + cost_shifter', absorb='C(firm_ids)'),
        instrument_formulation=pyRVtest.Formulation('0 + iv0 + iv1 + iv2'),
        model_formulations=model_formulations,
        product_data=product_data,
        demand_results=None,
    )
    return testing_problem.solve(demand_adjustment=False, clustering_adjustment=False)


def test_snapshot_analytical_base_with_fe(base_fe_results):
    """Snapshot: base DGP with cost fixed effects (absorb C(firm_ids))."""
    assert_snapshot('analytical_base_fe', base_fe_results)


# ---------------------------------------------------------------------------
# Scale economies / endogenous cost component DGP (seed=54321).
# Exercises the IV-correction code path that scales marginal cost with
# log quantity.
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def scale_results():
    product_data, _ = _build_scale_dgp()
    model_formulations = (
        pyRVtest.ModelFormulation(
            model_downstream='bertrand', ownership_downstream='firm_ids',
            user_supplied_markups='markups_m1',
        ),
        pyRVtest.ModelFormulation(
            model_downstream='perfect_competition',
            user_supplied_markups='markups_m2',
        ),
    )
    testing_problem = pyRVtest.Problem(
        cost_formulation=pyRVtest.Formulation('1 + cost_shifter + log_quantity'),
        instrument_formulation=pyRVtest.Formulation('0 + iv1 + iv2'),
        model_formulations=model_formulations,
        product_data=product_data,
        demand_results=None,
        endogenous_cost_component='log_quantity',
    )
    return testing_problem.solve(demand_adjustment=False, clustering_adjustment=False)


def test_snapshot_analytical_scale(scale_results):
    """Snapshot: scale-economies DGP with endogenous_cost_component."""
    assert_snapshot('analytical_scale', scale_results)


# ---------------------------------------------------------------------------
# First-stage-correction DGP (seed=99): PyBLP-estimated logit with
# over-identified demand, then pyRVtest with demand_adjustment=True via
# both PyBLP and demand_params paths. This is the code that had three
# bugs in b3b08a3 (sign error, missing concentration adjustment, wrong
# weight matrix). Snapshotting BOTH paths guards against silent drift
# in either.
# ---------------------------------------------------------------------------

def _build_first_stage_dgp_and_solve():
    """Reproduce the logit_dgp_and_estimation + comparison_results fixtures.

    Returns (r1, r2) where r1 is the PyBLP-path result and r2 is the
    demand_params-path result, both with demand_adjustment=True.

    Kept as a plain function (not a fixture) to avoid cross-module pytest
    fixture resolution. The DGP parameters match
    tests/test_first_stage_correction.py logit_dgp_and_estimation +
    TestFirstStageEquivalenceWithDemandAdjustment.comparison_results so
    snapshots here track the same code paths that suite exercises.
    """
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
        beta=[0, -2, 1],
        gamma=[1, 0.5],
        xi_variance=0.3,
        omega_variance=0.2,
        correlation=0.0,
        product_data=id_data,
        seed=99,
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

    alpha = float(pyblp_results.beta[pyblp_results.beta_labels.index("prices")])
    beta_x = float(pyblp_results.beta[pyblp_results.beta_labels.index("x1")])
    beta_0 = float(pyblp_results.beta[pyblp_results.beta_labels.index("1")])

    models = (
        pyRVtest.ModelFormulation(model_downstream="bertrand", ownership_downstream="firm_ids"),
        pyRVtest.ModelFormulation(model_downstream="perfect_competition"),
    )
    common = dict(
        cost_formulation=pyRVtest.Formulation("1 + z1"),
        instrument_formulation=pyRVtest.Formulation("0 + rival_x1 + rival_z1"),
        model_formulations=models,
        product_data=data,
    )

    pyRVtest.options.verbose = False
    r1 = pyRVtest.Problem(**common, demand_results=pyblp_results).solve(demand_adjustment=True)
    r2 = pyRVtest.Problem(
        **common,
        demand_params={
            "alpha": alpha,
            "sigma": [],
            "beta": np.array([beta_0, beta_x]),
            "x_columns": ["intercept", "x1"],
            "demand_instrument_columns": ["rival_x1", "rival_z1_sq", "intercept", "x1"],
        },
    ).solve(demand_adjustment=True)
    pyRVtest.options.verbose = True
    return r1, r2


@pytest.fixture(scope='module')
def first_stage_results():
    return _build_first_stage_dgp_and_solve()


def test_snapshot_first_stage_pyblp_path(first_stage_results):
    """Snapshot: PyBLP path with demand_adjustment=True.

    Exercises the code that had Bug 3 in b3b08a3 (updated_W vs W). Any
    future change to the weight-matrix selection logic would fail this
    snapshot.
    """
    r1, _ = first_stage_results
    assert_snapshot('first_stage_pyblp_path', r1)


def test_snapshot_first_stage_demand_params_path(first_stage_results):
    """Snapshot: demand_params path with demand_adjustment=True.

    Exercises the code that had Bugs 1 and 2 in b3b08a3 (sign error in
    d(markup)/dα; missing concentration adjustment of ∂ξ/∂θ on X_D).
    """
    _, r2 = first_stage_results
    assert_snapshot('first_stage_demand_params_path', r2)
