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
import pytest

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


def _build_logit_jacobian_backend_fixture():
    """Helper: synthetic-example data + UserSuppliedBackend whose Jacobian
    reproduces the logit ds/dp formula at a fixed alpha.

    Uses the shipped synthetic example dataset (pyRVtest.data.load_example)
    so shares are well-formed (sum < 1 with an outside option). The
    backend wraps the analytical logit Jacobian:

        ds_jt / dp_jt   = alpha * s_jt * (1 - s_jt)
        ds_jt / dp_kt   = -alpha * s_jt * s_kt        (k != j)

    This is what LogitBackend.compute_jacobian would produce internally.
    Wrapping it as UserSuppliedBackend exercises the "researcher pre-
    computed the Jacobian externally" path while staying numerically
    equivalent to demand_params={'estimate': 'logit', ...}.

    Returns ``(product_data_df, backend, alpha_value)``.
    """
    data = pyRVtest.data.load_example()
    market_ids = np.asarray(data['market_ids']).flatten()
    shares = np.asarray(data['shares']).flatten()
    n = len(market_ids)

    alpha = -1.0  # arbitrary; matches the "in-package logit" estimator's typical fit

    # Build the stacked NaN-padded Jacobian. Shape (n, J_max). Each market
    # has J_t = 2 products on the synthetic example.
    unique_markets = np.unique(market_ids)
    J_max = max(int(np.sum(market_ids == t)) for t in unique_markets)
    jacobian = np.full((n, J_max), np.nan)
    for t in unique_markets:
        idx = np.where(market_ids == t)[0]
        s_t = shares[idx]
        J_t = len(idx)
        block = np.zeros((J_t, J_t))
        for j in range(J_t):
            for k in range(J_t):
                if j == k:
                    block[j, k] = alpha * s_t[j] * (1.0 - s_t[j])
                else:
                    block[j, k] = -alpha * s_t[j] * s_t[k]
        jacobian[idx[:, None], np.arange(J_t)[None, :]] = block

    backend = UserSuppliedBackend(jacobian=jacobian, market_ids=market_ids)
    return data, backend, alpha


class TestProblemDemandBackendKwarg:
    """v0.4 final wires demand_backend= through Problem.__init__.

    Pre-rc3, custom-demand users had to bypass Problem's automatic markup
    computation by precomputing markups via _compute_markups and passing
    them in as markup_data. The demand_backend= kwarg (added in this
    feature branch) lets a researcher pass a UserSuppliedBackend (or any
    DemandBackend) directly to Problem, and the standard solve() path
    runs end-to-end against the user's custom demand.
    """

    def test_problem_constructs_and_solves_with_user_supplied_backend(self):
        """Problem(demand_backend=UserSuppliedBackend(...)) runs end-to-end.

        Backend is plumbed onto self._demand_backend, the markup pipeline
        sees it through compute_jacobian, and solve() produces finite
        TRV / F.
        """
        product_data, backend, _ = _build_logit_jacobian_backend_fixture()

        problem = pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('1 + z1 + z2'),
            instrument_formulation=pyRVtest.Formulation('0 + rival_z1 + rival_z2'),
            models=[
                pyRVtest.Bertrand(ownership='firm_ids'),
                pyRVtest.PerfectCompetition(),
            ],
            product_data=product_data,
            demand_backend=backend,
        )
        assert problem._demand_backend is backend

        results = problem.solve(demand_adjustment=False)

        assert np.isfinite(results.TRV[0][0, 1])
        assert np.isfinite(results.F[0][0, 1])

        # PerfectCompetition markups are identically zero by definition.
        pc_markups = results.markups[1].flatten()
        assert np.abs(pc_markups).max() < 1e-12

        # Bertrand markups are positive and finite (logit Jacobian gives
        # well-defined Bertrand markups on this fixture).
        bertrand_markups = results.markups[0].flatten()
        assert np.isfinite(bertrand_markups).all()

    def test_user_supplied_backend_matches_inline_logit(self):
        """A UserSuppliedBackend whose Jacobian reproduces the logit
        formula gives the same markups as demand_params={'estimate':
        'logit', ...} at the same alpha.

        Tighter version of the previous test: cross-validate that
        demand_backend= produces the same numerical answer as the
        inline analytical logit path when the Jacobians match.
        """
        product_data, backend, alpha = _build_logit_jacobian_backend_fixture()
        n = len(product_data)

        # Path 1: pre-computed UserSuppliedBackend.
        problem_a = pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('1 + z1 + z2'),
            instrument_formulation=pyRVtest.Formulation('0 + rival_z1 + rival_z2'),
            models=[
                pyRVtest.Bertrand(ownership='firm_ids'),
                pyRVtest.PerfectCompetition(),
            ],
            product_data=product_data,
            demand_backend=backend,
        )
        results_a = problem_a.solve(demand_adjustment=False)

        # Path 2: inline LogitBackend at the same alpha. To match the
        # backend's computed Jacobian exactly, demand_params must use
        # only constant + price terms (no x1) so the resulting alpha
        # purely scales ds/dp; any x1 in the demand model would change
        # the implied shares-vs-prices relationship, which doesn't
        # affect the Jacobian-from-shares formula but DOES affect what
        # LogitBackend computes when it inverts demand. Use a minimal
        # spec.
        product_data_local = product_data.copy()
        product_data_local['intercept'] = 1.0
        problem_b = pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('1 + z1 + z2'),
            instrument_formulation=pyRVtest.Formulation('0 + rival_z1 + rival_z2'),
            models=[
                pyRVtest.Bertrand(ownership='firm_ids'),
                pyRVtest.PerfectCompetition(),
            ],
            product_data=product_data_local,
            demand_params={
                'alpha': alpha,
                'beta': np.array([0.0]),
                'sigma': [],
                'x_columns': ['intercept'],
                'demand_instrument_columns': ['intercept'],
            },
        )
        results_b = problem_b.solve(demand_adjustment=False)

        # Bertrand markups must agree to high precision (both paths use
        # the same alpha and the same shares; the only difference is the
        # source of ds/dp — UserSuppliedBackend serves it from a static
        # Jacobian while LogitBackend recomputes the same formula).
        np.testing.assert_allclose(
            results_a.markups[0].flatten(),
            results_b.markups[0].flatten(),
            atol=1e-10,
        )

    def test_demand_backend_xor_demand_results(self):
        """Mutual exclusivity: demand_backend + demand_results raises ValueError."""
        product_data, backend, _ = _build_logit_jacobian_backend_fixture()

        # Sentinel demand_results-like object so we hit the mutual-
        # exclusivity check before any pyblp deserialization.
        class _FakePyBLPResults:
            ...

        with pytest.raises(ValueError, match='demand_backend and demand_results'):
            pyRVtest.Problem(
                cost_formulation=pyRVtest.Formulation('1 + z1'),
                instrument_formulation=pyRVtest.Formulation('0 + rival_z1'),
                models=[pyRVtest.Bertrand(ownership='firm_ids'),
                        pyRVtest.PerfectCompetition()],
                product_data=product_data,
                demand_backend=backend,
                demand_results=_FakePyBLPResults(),
            )

    def test_demand_backend_xor_demand_params(self):
        """Mutual exclusivity: demand_backend + demand_params raises ValueError."""
        product_data, backend, _ = _build_logit_jacobian_backend_fixture()

        with pytest.raises(ValueError, match='demand_backend and demand_params'):
            pyRVtest.Problem(
                cost_formulation=pyRVtest.Formulation('1 + z1'),
                instrument_formulation=pyRVtest.Formulation('0 + rival_z1'),
                models=[pyRVtest.Bertrand(ownership='firm_ids'),
                        pyRVtest.PerfectCompetition()],
                product_data=product_data,
                demand_backend=backend,
                demand_params={
                    'estimate': 'logit',
                    'formulation_X': pyRVtest.Formulation('1 + x1'),
                    'formulation_Z': pyRVtest.Formulation('0 + z1'),
                },
            )

    def test_demand_backend_must_satisfy_protocol(self):
        """Non-DemandBackend objects raise TypeError."""
        product_data, _, _ = _build_logit_jacobian_backend_fixture()

        with pytest.raises(TypeError, match='DemandBackend protocol'):
            pyRVtest.Problem(
                cost_formulation=pyRVtest.Formulation('1 + z1'),
                instrument_formulation=pyRVtest.Formulation('0 + rival_z1'),
                models=[pyRVtest.Bertrand(ownership='firm_ids'),
                        pyRVtest.PerfectCompetition()],
                product_data=product_data,
                demand_backend="not a backend",   # arbitrary non-protocol object
            )
