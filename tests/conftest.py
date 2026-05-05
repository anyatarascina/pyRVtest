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
