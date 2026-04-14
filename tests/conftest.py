"""Shared pytest fixtures for the pyRVtest regression suite.

These fixtures are session-scoped where possible so that expensive objects
(the PyBLP demand solve in particular) are constructed at most once per run.
"""

import pytest


@pytest.fixture(scope="session")
def tiny_product_data():
    """A small synthetic product-level dataset for fast pipeline tests.

    20 markets x 6 products = 120 observations. No fixed effects. Includes
    three instrument columns (z1, z2, z3), two characteristics (x1, x2),
    and a clustering id. Deterministic under a fixed seed.
    """
    from tests.fixtures.tiny_synthetic import make_tiny_data
    return make_tiny_data(T=20, J=6, seed=42)


@pytest.fixture(scope="session")
def tiny_pyblp_results(tiny_product_data):
    """Run a minimal PyBLP demand estimation on the tiny dataset.

    Used only by tests that exercise the `demand_adjustment=True` path.
    Cached per session; PyBLP's solve is the slowest step in the suite.
    """
    try:
        import pyblp
    except ImportError:
        pytest.skip("pyblp not installed")

    import numpy as np
    data = tiny_product_data.copy()
    # PyBLP requires a demand-instrument naming convention
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
    results = problem.solve(sigma=0.5, method='1s')
    return results
