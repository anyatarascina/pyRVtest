"""Shared fixtures for pyRVtest tests.

Session-scoped fixtures avoid re-constructing expensive objects (notably the
PyBLP demand solve) more than once per run.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session")
def tiny_product_data():
    """Small synthetic product-level dataset (T=20 markets x J=6 products).

    Deterministic under seed=42. Columns: market_ids, firm_ids, shares,
    prices, x1, x2, z1, z2, z3, clustering_ids, log_shares.
    """
    from tests.fixtures.tiny_synthetic import make_tiny_data
    return make_tiny_data(T=20, J=6, seed=42)


@pytest.fixture(scope="session")
def tiny_pyblp_results(tiny_product_data):
    """Minimal PyBLP demand estimation on the tiny dataset.

    Used by tests that exercise the demand_adjustment=True path.
    """
    try:
        import pyblp
    except ImportError:
        pytest.skip("pyblp not installed")

    data = tiny_product_data.copy()
    for i, col in enumerate(['z1', 'z2', 'z3']):
        data[f'demand_instruments{i}'] = data[col]

    formulation = (
        pyblp.Formulation('1 + prices + x1 + x2'),
        pyblp.Formulation('0 + x1'),
    )
    integration = pyblp.Integration('product', 3)
    problem = pyblp.Problem(
        product_formulations=formulation,
        product_data=data,
        integration=integration,
    )
    return problem.solve(sigma=0.5, method='1s')


@pytest.fixture(scope="module")
def logit_dgp_and_estimation():
    """Logit DGP with an explicit excluded demand instrument.

    The demand-adjustment first-stage correction (DMSS Appendix C eq. 77) is only
    non-trivial when there is variation in the instruments beyond the exogenous
    regressors in X1. Under pure logit with X1 = '1 + prices + x1' and no
    demand_instruments, PyBLP's ZD reduces to (intercept, x1) = X1_exogenous; the
    2SLS-style projection in the first-stage correction zeros out H by
    construction, and the correction vanishes. We add an explicit
    demand_instruments0 column (rival_x1) to avoid this degenerate case.

    ZD columns after this fixture: (intercept, x1, rival_x1). Both paths reference
    these same columns by name.

    Returns (data, pyblp_results, alpha_hat, beta_x_hat).
    """
    import numpy as np
    import pandas as pd
    import pyblp
    pyblp.options.verbose = False

    # Larger sample for stable GMM estimation. Small T leads to wildly wrong alpha
    # estimates (e.g., +21 instead of the true -2) which crashes downstream
    # demand_params validation (requires alpha < 0).
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

    # Rival characteristics for conduct-testing instruments AND as an excluded
    # demand instrument (so the demand estimation is over-identified).
    # Vectorize to avoid per-row DataFrame ops on larger samples.
    for t in range(T):
        idx = np.where(data["market_ids"] == t)[0]
        for j in idx:
            rival = [i for i in idx if i != j]
            data.loc[j, "rival_x1"] = data.loc[rival, "x1"].mean()
            data.loc[j, "rival_z1"] = data.loc[rival, "z1"].mean()
    # Additional cost-based instrument: rival z1 at a different moment
    data["rival_z1_sq"] = data["rival_z1"] ** 2

    # Explicit intercept for demand_params to reference by name.
    data["intercept"] = 1.0
    # Add two excluded demand instruments to over-identify demand (4 moments,
    # 3 parameters). Over-identification makes the first-stage-correction Λ
    # depend on the weight matrix, so the demand_adjustment_weight option toggle
    # produces a meaningful TRV difference.
    data["demand_instruments0"] = data["rival_x1"]
    data["demand_instruments1"] = data["rival_z1"] ** 2

    problem = pyblp.Problem((X1,), product_data=data)
    pyblp_results = problem.solve(method="1s")

    alpha_hat = float(pyblp_results.beta[pyblp_results.beta_labels.index("prices")].item())
    beta_x_hat = float(pyblp_results.beta[pyblp_results.beta_labels.index("x1")].item())

    return data, pyblp_results, alpha_hat, beta_x_hat
