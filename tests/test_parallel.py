"""Opt-in threaded per-market loops: equivalence, determinism, validation.

`Problem.solve(n_jobs=...)` parallelizes the per-market loops on the
(nested-)logit path (the markups-stage Jacobian build, the gradient pass, and
the demand-adjustment markup-gradient loop). Each market computes independently
and writes a disjoint output block, so the result must be **bit-identical**
regardless of `n_jobs` or worker scheduling. These tests lock that invariant in
and check the `n_jobs` validation surface.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import pyRVtest
from pyRVtest._parallel import resolve_n_jobs, for_each
from pyRVtest.backends import LogitBackend, NestedLogitBackend
from pyRVtest.backends.logit import compute_analytical_jacobian
from tests.test_demand_adjustment import _make_logit_fixture


# ---------------------------------------------------------------------------
# Irregular-market data (sizes 2..6, two nests) for the backend-level checks.
# ---------------------------------------------------------------------------

def _irregular_nested_data(seed: int = 99, n_markets: int = 60):
    rng = np.random.default_rng(seed)
    market_ids, shares, nesting_ids = [], [], []
    labels = rng.permutation(np.arange(500, 500 + n_markets) * 3)
    for t in labels:
        J_t = int(rng.integers(2, 7))
        e = np.exp(rng.normal(size=J_t))
        s = e / (1.0 + e.sum())
        market_ids.extend([t] * J_t)
        shares.extend(s.tolist())
        nesting_ids.extend(rng.choice(['A', 'B'], size=J_t).tolist())
    return pd.DataFrame({
        'market_ids': np.asarray(market_ids),
        'shares': np.asarray(shares),
        'nesting_ids': np.asarray(nesting_ids),
    })


# ---------------------------------------------------------------------------
# resolve_n_jobs / for_each unit behavior
# ---------------------------------------------------------------------------

class TestResolveNJobs:
    def test_default_and_one_are_serial(self):
        assert resolve_n_jobs(None) == 1
        assert resolve_n_jobs(1) == 1

    def test_minus_one_is_all_cores(self):
        assert resolve_n_jobs(-1) == max(1, os.cpu_count() or 1)

    def test_positive_passthrough(self):
        assert resolve_n_jobs(4) == 4

    def test_zero_and_negative_raise(self):
        with pytest.raises(ValueError):
            resolve_n_jobs(0)
        with pytest.raises(ValueError):
            resolve_n_jobs(-2)

    def test_non_integer_raises(self):
        with pytest.raises(ValueError):
            resolve_n_jobs('lots')


def test_for_each_visits_every_item_threaded():
    # Disjoint writes (each index set once) -> safe to scatter from threads.
    out = np.zeros(50, dtype=int)
    for_each(range(50), lambda k: out.__setitem__(k, k * k), n_jobs=4)
    assert np.array_equal(out, np.arange(50) ** 2)


# ---------------------------------------------------------------------------
# Builder + gradient equivalence across n_jobs (bit-identical)
# ---------------------------------------------------------------------------

def test_builder_bit_identical_plain_logit_across_n_jobs():
    data = _irregular_nested_data(seed=1)
    serial = compute_analytical_jacobian(-1.5, [], data, n_jobs=1)
    threaded = compute_analytical_jacobian(-1.5, [], data, n_jobs=4)
    assert np.array_equal(serial, threaded, equal_nan=True)


def test_builder_bit_identical_nested_logit_across_n_jobs():
    data = _irregular_nested_data(seed=2)
    kw = dict(nesting_ids_columns=['nesting_ids'])
    serial = compute_analytical_jacobian(-1.5, [0.4], data, n_jobs=1, **kw)
    threaded = compute_analytical_jacobian(-1.5, [0.4], data, n_jobs=4, **kw)
    assert np.array_equal(serial, threaded, equal_nan=True)


def test_jacobian_gradient_all_markets_identical_across_n_jobs():
    data = _irregular_nested_data(seed=3)
    backend = NestedLogitBackend(
        alpha=-2.0, rho=[0.3], product_data=data,
        nesting_ids_columns=['nesting_ids'],
    )
    backend._n_jobs = 1
    serial = backend.jacobian_gradient_all_markets()
    # Fresh backend so the threaded run rebuilds caches under threading too.
    backend2 = NestedLogitBackend(
        alpha=-2.0, rho=[0.3], product_data=data,
        nesting_ids_columns=['nesting_ids'],
    )
    backend2._n_jobs = 4
    threaded = backend2.jacobian_gradient_all_markets()
    assert serial.keys() == threaded.keys()
    for t in serial:
        assert np.array_equal(serial[t], threaded[t], equal_nan=True)


def test_logit_builder_via_backend_n_jobs_attr():
    """LogitBackend.compute_jacobian honors _n_jobs and stays bit-identical."""
    data = _irregular_nested_data(seed=4)[['market_ids', 'shares']]
    b1 = LogitBackend(alpha=-1.5, product_data=data)
    b1._n_jobs = 1
    b4 = LogitBackend(alpha=-1.5, product_data=data)
    b4._n_jobs = 4
    assert np.array_equal(b1.compute_jacobian(), b4.compute_jacobian(), equal_nan=True)


# ---------------------------------------------------------------------------
# End-to-end determinism through Problem.solve (nested logit, two models,
# demand adjustment + clustering). n_jobs=1 vs n_jobs=4 must match exactly.
# ---------------------------------------------------------------------------

def _nested_problem():
    df = _make_logit_fixture(seed=7, T=60, J=4, with_nesting=True)
    df['clustering_ids'] = df['market_ids']
    pyRVtest.options.verbose = False
    return pyRVtest.Problem(
        cost_formulation=pyRVtest.Formulation('1 + z1'),
        instrument_formulation=pyRVtest.Formulation('0 + rival_z1'),
        product_data=df,
        models=[
            pyRVtest.Bertrand(ownership='firm_ids'),
            pyRVtest.Monopoly(),
        ],
        demand_params={
            'alpha': -1.5, 'rho': [0.3],
            'beta': np.array([0.0, 0.4]),
            'x_columns': ['intercept', 'x1'],
            'demand_instrument_columns': ['rival_x1', 'intercept', 'x1'],
        },
    )


def test_solve_results_identical_across_n_jobs():
    r1 = _nested_problem().solve(demand_adjustment=True, clustering_adjustment=True, n_jobs=1)
    r4 = _nested_problem().solve(demand_adjustment=True, clustering_adjustment=True, n_jobs=4)
    for m in range(len(r1.markups)):
        np.testing.assert_allclose(r4.markups[m], r1.markups[m], atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(r4.TRV, r1.TRV, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(r4.F, r1.F, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(r4.MCS_pvalues, r1.MCS_pvalues, atol=1e-12, equal_nan=True)


def test_solve_rejects_invalid_n_jobs():
    problem = _nested_problem()
    with pytest.raises(ValueError):
        problem.solve(demand_adjustment=False, n_jobs=0)
