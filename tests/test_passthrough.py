"""v0.4 step 11: public ``build_passthrough`` helper.

Exercises :func:`pyRVtest.build_passthrough` against a minimal vertical
DGP (Bertrand downstream + Monopoly upstream). Asserts:

* Single-market call returns a ``(J_t, J_t)`` matrix.
* ``market_id=None`` returns a dict keyed by every market id.
* The returned matrix matches a local replication using the same backend
  calls (byte-identical at atol=1e-14), confirming the helper wraps the
  same math the solve pipeline uses internally.
* Non-vertical model, invalid market id, and out-of-range model index
  all raise clear errors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyRVtest
from pyRVtest import (
    Bertrand,
    Monopoly,
    PerfectCompetition,
    Vertical,
)
from pyRVtest.markups import _construct_passthrough_from_hessian


def _make_vertical_dgp(seed: int = 1234, T: int = 10, J: int = 4, alpha: float = -1.5):
    """Minimal plain-logit DGP with upstream firm ids for vertical models."""
    rng = np.random.default_rng(seed=seed)
    N = T * J
    market_ids = np.repeat(np.arange(T), J)
    firm_ids = np.tile(np.arange(J), T)
    upstream_firm_ids = np.tile(['m1', 'm1', 'm2', 'm2'], T)
    vi_col = np.zeros(N, dtype=int)
    x1 = rng.normal(size=N)
    prices = rng.uniform(0.8, 2.0, size=N)
    u = 0.4 * x1 + rng.normal(scale=0.2, size=N)
    delta = u + alpha * prices
    shares = np.zeros(N)
    for t in range(T):
        idx = np.where(market_ids == t)[0]
        e = np.exp(delta[idx])
        shares[idx] = e / (1.0 + e.sum())
    rival_x1 = np.zeros(N)
    for t in range(T):
        idx = np.where(market_ids == t)[0]
        for j in idx:
            others = [i for i in idx if i != j]
            rival_x1[j] = x1[others].mean()
    z1 = rng.normal(size=N) + 1.5
    df = pd.DataFrame({
        'market_ids': market_ids,
        'firm_ids': firm_ids,
        'upstream_firm_ids': upstream_firm_ids,
        'vi_col': vi_col,
        'x1': x1,
        'prices': prices,
        'shares': shares,
        'rival_x1': rival_x1,
        'z1': z1,
        'intercept': np.ones(N),
    })
    return df, alpha


def _make_vertical_problem():
    df, alpha = _make_vertical_dgp()
    pyRVtest.options.verbose = False
    return pyRVtest.Problem(
        cost_formulation=pyRVtest.Formulation('1 + z1'),
        instrument_formulation=pyRVtest.Formulation('0 + rival_x1'),
        product_data=df,
        demand_params={
            'alpha': alpha, 'rho': [],
            'beta': np.array([0.0, 0.4]),
            'x_columns': ['intercept', 'x1'],
            'demand_instrument_columns': ['rival_x1', 'intercept', 'x1'],
        },
        models=[
            Vertical(
                downstream=Bertrand(ownership='firm_ids'),
                upstream=Monopoly(ownership='upstream_firm_ids'),
                vertical_integration='vi_col',
            ),
            PerfectCompetition(),
        ],
    )


@pytest.fixture(scope='module')
def vertical_problem():
    return _make_vertical_problem()


class TestBuildPassthrough:
    """v0.4 step 11: public build_passthrough helper."""

    def test_single_market_returns_matrix(self, vertical_problem):
        """build_passthrough with a specific market_id returns (J_t, J_t)."""
        t = vertical_problem.unique_market_ids[0]
        M = pyRVtest.build_passthrough(vertical_problem, 0, market_id=t)
        assert isinstance(M, np.ndarray)
        assert M.ndim == 2
        # All markets in the fixture have J=4 products.
        j_t = int(np.sum(vertical_problem.products.market_ids.flatten() == t))
        assert M.shape == (j_t, j_t)

    def test_all_markets_returns_dict(self, vertical_problem):
        """build_passthrough with market_id=None returns {t: matrix_t}."""
        result = pyRVtest.build_passthrough(vertical_problem, 0)
        assert isinstance(result, dict)
        expected_keys = set(vertical_problem.unique_market_ids.tolist())
        assert set(result.keys()) == expected_keys
        for t, M in result.items():
            j_t = int(np.sum(vertical_problem.products.market_ids.flatten() == t))
            assert M.shape == (j_t, j_t)

    def test_matches_internal_solve_passthrough(self, vertical_problem):
        """The helper result matches a local replication using the same backend
        calls — the same math the solve pipeline invokes internally."""
        # Replicate exactly what build_passthrough does, independent of the
        # helper, using the same backend primitives the solve pipeline uses.
        _, markups_downstream, _ = vertical_problem._perturb_and_build_markups()
        markups_down = markups_downstream[0]
        ownership_downstream = vertical_problem.models['ownership_downstream'][0]
        product_market_ids = vertical_problem.products.market_ids.flatten()
        backend = vertical_problem._demand_backend

        produced = pyRVtest.build_passthrough(vertical_problem, 0)
        for t in vertical_problem.unique_market_ids:
            index_t = np.where(product_market_ids == t)[0]
            D_t = backend.compute_jacobian(market_id=t)
            d2s_dp2_t = backend.compute_hessian(market_id=t)
            ownership_t = ownership_downstream[index_t]
            ownership_t = ownership_t[:, ~np.isnan(ownership_t).all(axis=0)]
            markups_t = markups_down[index_t]
            expected = _construct_passthrough_from_hessian(
                d2s_dp2_t, D_t, ownership_t, markups_t
            )
            np.testing.assert_allclose(
                produced[t], expected, atol=1e-14,
                err_msg=f"build_passthrough differs from local replication in market {t!r}",
            )

    def test_non_vertical_model_returns_identity_for_pc(self, vertical_problem):
        """Phase 1 of the DMQSW diagnostic generalises ``build_passthrough``
        to every conduct class. ``PerfectCompetition`` is the trivial-closed-
        form case: ``Delta = 0`` for every product so ``P = I`` exactly."""
        # Model index 1 is PerfectCompetition.
        result = pyRVtest.build_passthrough(vertical_problem, 1)
        assert isinstance(result, dict)
        for t, M in result.items():
            j_t = int(np.sum(vertical_problem.products.market_ids.flatten() == t))
            assert M.shape == (j_t, j_t)
            np.testing.assert_allclose(M, np.eye(j_t), atol=1e-14)

    def test_invalid_market_id_raises(self, vertical_problem):
        """Non-existent market_id raises ValueError."""
        with pytest.raises(ValueError, match="not in problem.unique_market_ids"):
            pyRVtest.build_passthrough(vertical_problem, 0, market_id=999)

    def test_invalid_model_index_raises(self, vertical_problem):
        """model_index out of range raises ValueError."""
        n = len(vertical_problem._models)
        with pytest.raises(ValueError, match="out of range"):
            pyRVtest.build_passthrough(vertical_problem, n)
        with pytest.raises(ValueError, match="out of range"):
            pyRVtest.build_passthrough(vertical_problem, -1)
