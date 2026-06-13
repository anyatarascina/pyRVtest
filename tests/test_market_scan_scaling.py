"""Regression guards for the per-market groupby on the (nested-)logit solve path.

Three routines used to recompute per-market row indices with a full-length
``np.where`` / ``np.sum`` *inside a loop over markets*, giving O(N * n_markets)
behavior that made a fine-market problem (store x week) never finish:

* ``compute_analytical_jacobian``            (Hotspot 1, backends/logit.py)
* ``NestedLogitBackend.compute_jacobian``    (Hotspot 2, backends/nested_logit.py)
* ``compute_demand_adjustment``              (Hotspot 3, solve/demand_adjustment.py)

All three were switched to the O(N log N) stable-sort groupby already used by
``LogitBackend._ensure_market_indices``. Because the sort is *stable*, each
market's contiguous sorted segment equals ``np.where(market_ids == t)[0]`` in
its original row order, so the fixes are bit-identical.

This file asserts (a) the groupby is bit-identical to the old per-market
``np.where`` build, and (b) per-market cost no longer scales with n_markets:
two problems with the *same* N but ~10x different n_markets must build the
Jacobian in comparable wall time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyRVtest.backends import LogitBackend, NestedLogitBackend
from pyRVtest.backends.logit import (
    compute_analytical_jacobian, _logit_jacobian, _nested_logit_jacobian,
)


# ---------------------------------------------------------------------------
# Synthetic data with IRREGULAR market sizes (the case the old code padded).
# ---------------------------------------------------------------------------

def _irregular_data(seed: int = 12345, n_markets: int = 50,
                    min_J: int = 1, max_J: int = 6, nested: bool = False):
    """Build (nested-)logit data with per-market product counts in [min_J, max_J].

    Market ids are intentionally non-contiguous and shuffled-in-value so the
    stable-sort groupby is genuinely exercised (not an identity ordering).
    """
    rng = np.random.default_rng(seed)
    market_ids, shares, nesting_ids = [], [], []
    # Non-contiguous, non-monotone market labels.
    labels = rng.permutation(np.arange(1000, 1000 + n_markets) * 7)
    for t in labels:
        J_t = int(rng.integers(min_J, max_J + 1))
        u = rng.normal(size=J_t)
        exp_u = np.exp(u)
        s = exp_u / (1.0 + exp_u.sum())
        market_ids.extend([t] * J_t)
        shares.extend(s.tolist())
        if nested:
            nesting_ids.extend(rng.choice(['A', 'B'], size=J_t).tolist())
    data = {
        'market_ids': np.asarray(market_ids),
        'shares': np.asarray(shares),
    }
    if nested:
        data['nesting_ids'] = np.asarray(nesting_ids)
    return pd.DataFrame(data)


def _reference_jacobian(alpha, rho, product_data, nesting_ids_columns=None):
    """The pre-fix algorithm: J_max / index lookup via per-market ``np.where``.

    Kept here as an independent oracle so the groupby refactor is checked to be
    bit-identical rather than self-consistent.
    """
    market_ids = np.asarray(product_data['market_ids']).flatten()
    shares = np.asarray(product_data['shares']).flatten()
    markets = np.unique(market_ids)
    N = len(market_ids)
    J_max = max(np.sum(market_ids == t) for t in markets)
    rho = [s for s in rho if s > 0]
    L = len(rho)
    nesting_arrays = []
    if L > 0:
        cols = nesting_ids_columns
        for col in cols:
            nesting_arrays.append(np.asarray(product_data[col]).flatten())
    jacobian = np.full((N, J_max), np.nan)
    for t in markets:
        idx = np.where(market_ids == t)[0]
        J_t = len(idx)
        s_t = shares[idx]
        if L == 0:
            D_t = _logit_jacobian(alpha, s_t)
        else:
            nesting_t = [arr[idx] for arr in nesting_arrays]
            D_t = _nested_logit_jacobian(alpha, rho, s_t, nesting_t)
        jacobian[idx[:, None], np.arange(J_t)[None, :]] = D_t
    return jacobian


# ---------------------------------------------------------------------------
# Bit-identical correctness (Hotspots 1 & 2)
# ---------------------------------------------------------------------------

def test_full_builder_bit_identical_plain_logit():
    data = _irregular_data(seed=1, nested=False)
    fast = compute_analytical_jacobian(-1.5, [], data, nesting_ids_columns=None)
    ref = _reference_jacobian(-1.5, [], data)
    assert np.array_equal(fast, ref, equal_nan=True)


def test_full_builder_bit_identical_nested_logit():
    data = _irregular_data(seed=2, nested=True)
    fast = compute_analytical_jacobian(
        -1.5, [0.4], data, nesting_ids_columns=['nesting_ids'])
    ref = _reference_jacobian(
        -1.5, [0.4], data, nesting_ids_columns=['nesting_ids'])
    assert np.array_equal(fast, ref, equal_nan=True)


def test_nested_per_market_block_matches_where_slice():
    """Hotspot 2: per-market slice via the cached groupby == np.where slice."""
    data = _irregular_data(seed=3, nested=True)
    backend = NestedLogitBackend(alpha=-2.0, rho=[0.3], product_data=data)
    full = backend.compute_jacobian()
    mids = np.asarray(data['market_ids']).flatten()
    for t in np.unique(mids):
        idx = np.where(mids == t)[0]
        expected = full[idx]
        expected = expected[:, ~np.isnan(expected).all(axis=0)]
        actual = backend.compute_jacobian(market_id=t)
        assert np.array_equal(actual, expected, equal_nan=True)


def test_logit_per_market_block_matches_where_slice():
    data = _irregular_data(seed=4, nested=False)
    backend = LogitBackend(alpha=-2.0, product_data=data)
    full = backend.compute_jacobian()
    mids = np.asarray(data['market_ids']).flatten()
    for t in np.unique(mids):
        idx = np.where(mids == t)[0]
        expected = full[idx]
        expected = expected[:, ~np.isnan(expected).all(axis=0)]
        actual = backend.compute_jacobian(market_id=t)
        assert np.array_equal(actual, expected, equal_nan=True)


# ---------------------------------------------------------------------------
# Scaling guard: same N, ~10x different n_markets -> comparable build time.
# A reintroduced per-market full scan would make the many-markets build scale
# with n_markets and blow past the (generous) bound.
# ---------------------------------------------------------------------------

def _fixed_N_data(seed: int, n_markets: int, N: int = 40000):
    """Plain-logit data with exactly N rows split across n_markets markets."""
    rng = np.random.default_rng(seed)
    base = N // n_markets
    counts = np.full(n_markets, base)
    counts[: N - base * n_markets] += 1  # distribute the remainder
    market_ids = np.repeat(np.arange(n_markets), counts)
    u = rng.normal(size=N)
    shares = np.empty(N)
    start = 0
    for c in counts:
        seg = u[start:start + c]
        e = np.exp(seg)
        shares[start:start + c] = e / (1.0 + e.sum())
        start += c
    return pd.DataFrame({'market_ids': market_ids, 'shares': shares})


def _best_build_time(data):
    import time
    best = float('inf')
    for _ in range(3):
        t0 = time.perf_counter()
        compute_analytical_jacobian(-1.5, [], data, nesting_ids_columns=None)
        best = min(best, time.perf_counter() - t0)
    return best


def test_build_time_does_not_scale_with_n_markets():
    few = _fixed_N_data(seed=10, n_markets=200)
    many = _fixed_N_data(seed=11, n_markets=2000)  # 10x more markets, same N
    t_few = _best_build_time(few)
    t_many = _best_build_time(many)
    # With the O(N log N) groupby, the many-markets build is comparable to (in
    # fact usually faster than) the few-markets build. The old O(N*n_markets)
    # scan would make it ~10x slower. Generous bound to stay non-flaky on CI.
    assert t_many < 8.0 * t_few + 0.05, (
        f"many-markets build ({t_many:.4f}s) scales with n_markets vs "
        f"few-markets ({t_few:.4f}s); a per-market full scan was likely "
        f"reintroduced."
    )
