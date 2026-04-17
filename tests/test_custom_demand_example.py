"""End-to-end test of the ``UserSuppliedBackend`` worked example.

v0.4 step 15. This test mirrors the code block in
``docs/custom_demand.rst`` 1:1 so a reader who copy-pastes the doc code
sees the same numerical output. If the doc and the test drift, fix both
together.

The example builds a stylized linear demand system (not a logit), wraps
the analytical Jacobian in :class:`UserSuppliedBackend`, and runs the
markup computation for a Bertrand vs perfect-competition comparison. We
verify that:

* Bertrand markups are finite, correctly shaped, and strictly positive.
* Perfect-competition markups are identically zero.
* Because the data-generating process is Bertrand-Nash pricing with the
  same demand Jacobian, the implied marginal cost
  :math:`p - \\text{markup}^{\\text{Bertrand}}` recovers the true ``mc``
  to machine precision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import pyRVtest
from pyRVtest.backends import UserSuppliedBackend
from pyRVtest.markups import _compute_markups
from pyRVtest.problem import Models


def test_custom_demand_worked_example():
    rng = np.random.default_rng(seed=42)
    T, J = 30, 3
    N = T * J

    market_ids = np.repeat(np.arange(T), J)
    firm_ids = np.tile(np.arange(J), T)

    # Stylized linear demand:
    #   s_{jt} = a_j - b p_{jt} + (c/(J-1)) sum_{k!=j} p_{kt}
    # Jacobian entries: ds_j/dp_j = -b, ds_j/dp_k = c/(J-1) for k != j.
    b, c = 0.8, 0.15
    a_j = np.array([1.8, 1.6, 1.4])

    def demand_jacobian_block(J_t):
        D = np.full((J_t, J_t), c / (J_t - 1))
        np.fill_diagonal(D, -b)
        return D

    # Simulate Bertrand-Nash prices and shares (true DGP is Bertrand with
    # single-product firms, so the FOC per product is p - mc = s / b).
    mc = 0.5 + 0.1 * rng.normal(size=N)
    prices = np.empty(N)
    shares = np.empty(N)
    for t in range(T):
        idx = np.where(market_ids == t)[0]
        p_t = mc[idx] + 0.5
        for _ in range(5000):
            s_t = a_j - b * p_t + (c / (J - 1)) * (p_t.sum() - p_t)
            p_new = mc[idx] + s_t / b
            if np.max(np.abs(p_new - p_t)) < 1e-14:
                break
            p_t = 0.5 * p_t + 0.5 * p_new
        s_t = a_j - b * p_t + (c / (J - 1)) * (p_t.sum() - p_t)
        prices[idx] = p_t
        shares[idx] = s_t

    # Rival-level cost shifter for use as a testing instrument downstream.
    rival_mean_mc = np.zeros(N)
    for t in range(T):
        idx = np.where(market_ids == t)[0]
        for i in idx:
            rival_mean_mc[i] = np.mean([mc[k] for k in idx if k != i])

    product_data = pd.DataFrame({
        'market_ids': market_ids,
        'firm_ids': firm_ids,
        'prices': prices,
        'shares': shares,
        'w_cost': mc,
        'z_rival_mc': rival_mean_mc,
    }).to_records(index=False)

    # Build the stacked NaN-padded user-supplied Jacobian.
    J_max = J
    jacobian = np.full((N, J_max), np.nan)
    for t in range(T):
        idx = np.where(market_ids == t)[0]
        J_t = len(idx)
        jacobian[idx[:, None], np.arange(J_t)[None, :]] = demand_jacobian_block(J_t)

    backend = UserSuppliedBackend(jacobian=jacobian, market_ids=market_ids)

    # Compute markups for two candidate conduct models.
    models_list = [
        pyRVtest.Bertrand(ownership='firm_ids'),
        pyRVtest.PerfectCompetition(),
    ]
    models = Models(models_list, product_data)

    markups, markups_down, markups_up = _compute_markups(
        product_data=product_data,
        pyblp_results=None,
        model_downstream=models['models_downstream'],
        ownership_downstream=models['ownership_downstream'],
        model_upstream=models['models_upstream'],
        ownership_upstream=models['ownership_upstream'],
        vertical_integration=models['vertical_integration'],
        custom_model_specification=models['custom_model_specification'],
        user_supplied_markups=models['user_supplied_markups'],
        mix_flag=models['mix_flag'],
        demand_backend=backend,
    )

    bertrand_markups = markups[0].flatten()
    pc_markups = markups[1].flatten()
    implied_mc = prices - bertrand_markups

    # Shape + finiteness.
    assert markups[0].shape == (N, 1)
    assert markups[1].shape == (N, 1)
    assert np.isfinite(markups[0]).all()
    assert np.isfinite(markups[1]).all()

    # Perfect-competition markups are identically zero.
    assert np.abs(pc_markups).max() < 1e-12

    # Bertrand markups are strictly positive (every product-market).
    assert (bertrand_markups > 0).all()

    # Because the DGP is Bertrand-Nash with the same Jacobian the backend
    # reports, the implied marginal cost recovers the true mc to machine
    # precision.
    assert np.abs(implied_mc - mc).max() < 1e-8
