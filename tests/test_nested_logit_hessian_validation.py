"""Validation tests for the analytical nested-logit Hessian landed in step 7.

Step 7 (commit d6007f2) replaced the finite-difference ``dD/ds`` inside
``compute_analytical_hessian`` with a closed form for plain logit and
1-level nested logit. Agent 7 cross-checked the new formula against the
existing finite-diff implementation, but that check is self-consistency
only — if the analytical derivation has a systematic error, finite-diff
would catch it only if the error affects the Jacobian-of-Jacobian
structure (which is what finite-diff approximates too).

This module adds three INDEPENDENT checks that don't rely on the
finite-diff reference:

  1. **Symmetry in price indices.** The Hessian
     ``H[j, k, l] = d^2 s_j / (dp_k dp_l)`` is symmetric in (k, l) by
     Clairaut's theorem. Pure math property; catches coordinate-mixing
     bugs.

  2. **Cross-check against pyblp's ``compute_demand_hessians``.** For a
     scalar-rho nested logit estimated in pyblp, the Hessian computed
     by pyblp's own code and by ``NestedLogitBackend.compute_hessian``
     describe the same mathematical object. They should agree at
     machine precision modulo pyblp's internal numerical slop.

  3. **Snapshot for the nested-logit vertical pipeline.** Downstream
     regression protection: any future change to the nested-logit
     Hessian code, the vertical passthrough path, or the Villas-Boas
     upstream markup will perturb this snapshot and fail the test.
"""

from __future__ import annotations

import contextlib
import os

import numpy as np
import pandas as pd
import pytest

import pyRVtest
from pyRVtest.backends import NestedLogitBackend
from pyRVtest.backends.logit import compute_analytical_hessian

from ._snapshot_helpers import assert_snapshot


# ---------------------------------------------------------------------------
# Property 1: Hessian symmetry in price indices (k, l).
# ---------------------------------------------------------------------------

class TestNestedLogitHessianSymmetry:
    """``H[j, k, l] == H[j, l, k]`` for all j, k, l. Clairaut's theorem."""

    @pytest.mark.parametrize("seed,J,rho", [
        (1, 4, 0.2),
        (2, 6, 0.5),
        (3, 5, 0.7),
        (4, 4, 0.9),
        (5, 8, 0.35),
    ])
    def test_one_level_nested_hessian_is_symmetric(self, seed, J, rho):
        rng = np.random.default_rng(seed)
        s = rng.dirichlet(np.ones(J + 1))[:J]
        alpha = -1.5 - rng.uniform()
        # Alternating nest assignment with at least 2 products per nest.
        nest_ids = np.array([i % 2 for i in range(J)])
        if np.bincount(nest_ids).min() < 2:
            nest_ids[-1] = nest_ids[0]
        H = compute_analytical_hessian(alpha, [rho], s, [nest_ids])
        np.testing.assert_allclose(
            H, H.transpose(0, 2, 1), atol=1e-12,
            err_msg=(
                f"Nested-logit Hessian is not symmetric in price indices "
                f"(Clairaut violation) for seed={seed}, J={J}, rho={rho}."
            ),
        )

    @pytest.mark.parametrize("seed,J", [(10, 3), (11, 5), (12, 6)])
    def test_plain_logit_hessian_is_symmetric(self, seed, J):
        rng = np.random.default_rng(seed)
        s = rng.dirichlet(np.ones(J + 1))[:J]
        alpha = -2.0
        H = compute_analytical_hessian(alpha, [], s, [])
        np.testing.assert_allclose(H, H.transpose(0, 2, 1), atol=1e-12)


# ---------------------------------------------------------------------------
# Property 2: cross-check against pyblp's compute_demand_hessians for
# scalar-rho nested logit. Independent math validation.
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def pyblp_nested_logit_fixture():
    """Small pyblp-estimated nested logit with a scalar rho."""
    import pyblp
    pyblp.options.verbose = False
    T, J = 20, 4
    market_ids = np.repeat(np.arange(T), J)
    firm_ids = np.tile(np.arange(J), T)
    # Two nests per market: products {0,1} -> 'A', products {2,3} -> 'B'
    nesting_ids = np.tile(['A', 'A', 'B', 'B'], T)
    id_data = pd.DataFrame({
        'market_ids': market_ids,
        'firm_ids': firm_ids,
        'nesting_ids': nesting_ids,
    })
    X1 = pyblp.Formulation('1 + prices + x1')
    X3 = pyblp.Formulation('1 + z1')
    simulation = pyblp.Simulation(
        product_formulations=(X1, None, X3),
        beta=[0.0, -2.0, 1.0], gamma=[1.0, 0.5],
        rho=0.4,
        xi_variance=0.1, omega_variance=0.1, correlation=0.0,
        product_data=id_data, seed=2026,
    )
    sim_results = simulation.replace_endogenous()
    data = pd.DataFrame(pyblp.data_to_dict(sim_results.product_data))
    # Rival instrument
    rival_x1 = np.zeros(T * J)
    for t in range(T):
        idx = np.where(data['market_ids'] == t)[0]
        for j in idx:
            others = [i for i in idx if i != j]
            rival_x1[j] = data['x1'].values[others].mean()
    data['rival_x1'] = rival_x1
    data['demand_instruments0'] = data['rival_x1']
    problem = pyblp.Problem((X1,), product_data=data)
    results = problem.solve(rho=0.3, method='1s')
    return data, results


class TestNestedLogitHessianMatchesPyblp:
    """pyblp's ``compute_demand_hessians`` and
    ``NestedLogitBackend.compute_hessian`` compute the same mathematical
    object (d²s_j / dp_k dp_l) for a scalar-rho nested logit. They should
    agree up to pyblp's internal numerical tolerance.
    """

    def test_hessian_matches_pyblp_on_first_market(self, pyblp_nested_logit_fixture):
        data, pyblp_results = pyblp_nested_logit_fixture
        # Grab the first market id
        t0 = np.unique(data['market_ids'].values)[0]

        # pyblp's Hessian for this market
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            H_pyblp = pyblp_results.compute_demand_hessians(market_id=t0)

        # NestedLogitBackend's analytical Hessian with matching parameters
        alpha = float(pyblp_results.beta[pyblp_results.beta_labels.index('prices')].item())
        rho_arr = np.atleast_1d(np.asarray(pyblp_results.rho).flatten())
        rho = float(rho_arr[0])
        backend = NestedLogitBackend(
            alpha=alpha, sigma=[rho], product_data=data,
            nesting_ids_columns=['nesting_ids'],
        )
        H_backend = backend.compute_hessian(market_id=t0)

        # Both are (J_t, J_t, J_t). Cross-validate at a tolerance that
        # accommodates pyblp's finite-diff (compute_demand_hessians uses
        # finite-diff internally) while still catching a real systematic
        # discrepancy.
        assert H_pyblp.shape == H_backend.shape
        np.testing.assert_allclose(
            H_backend, H_pyblp, atol=1e-6,
            err_msg=(
                "NestedLogitBackend analytical Hessian disagrees with "
                "pyblp's compute_demand_hessians output for a scalar-rho "
                "nested logit. The two should agree on d^2 s_j / dp_k dp_l "
                "because they describe the same mathematical object. A "
                "persistent disagreement indicates a bug in the step-7 "
                "analytical derivation (or, less likely, a pyblp regression)."
            ),
        )

    def test_hessian_matches_pyblp_on_several_markets(
            self, pyblp_nested_logit_fixture):
        """Repeat the check on 5 arbitrary markets to rule out market-specific
        coincidence."""
        data, pyblp_results = pyblp_nested_logit_fixture
        market_ids = np.unique(data['market_ids'].values)[:5]
        alpha = float(pyblp_results.beta[pyblp_results.beta_labels.index('prices')].item())
        rho_arr = np.atleast_1d(np.asarray(pyblp_results.rho).flatten())
        rho = float(rho_arr[0])
        backend = NestedLogitBackend(
            alpha=alpha, sigma=[rho], product_data=data,
            nesting_ids_columns=['nesting_ids'],
        )
        for t in market_ids:
            with contextlib.redirect_stdout(open(os.devnull, 'w')):
                H_pyblp = pyblp_results.compute_demand_hessians(market_id=t)
            H_backend = backend.compute_hessian(market_id=t)
            np.testing.assert_allclose(
                H_backend, H_pyblp, atol=1e-6,
                err_msg=f"Hessian mismatch vs pyblp at market_id={t}",
            )


# ---------------------------------------------------------------------------
# Snapshot 3: nested-logit vertical pipeline.
# ---------------------------------------------------------------------------

def _build_nested_vertical_dgp(seed: int = 909, T: int = 12, J: int = 4):
    """Synthetic nested-logit vertical DGP for snapshot regression."""
    rng = np.random.default_rng(seed=seed)
    N = T * J
    market_ids = np.repeat(np.arange(T), J)
    downstream_firm_ids = np.tile([0, 0, 1, 1], T)
    upstream_firm_ids = np.ones(N, dtype=int)  # single upstream manufacturer
    nesting_ids = np.tile(['A', 'A', 'B', 'B'], T)
    vi_col = np.zeros(N, dtype=int)  # no vertical integration
    x1 = rng.normal(size=N)
    prices = rng.uniform(0.8, 2.0, size=N)
    alpha = -1.5
    rho = 0.35
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
    iv0 = rng.normal(size=N)
    return pd.DataFrame({
        'market_ids': market_ids,
        'firm_ids': downstream_firm_ids,
        'upstream_firm_ids': upstream_firm_ids,
        'nesting_ids': nesting_ids,
        'vi_col': vi_col,
        'prices': prices,
        'shares': shares,
        'x1': x1, 'intercept': np.ones(N),
        'rival_x1': rival_x1,
        'z1': z1,
        'iv0': iv0,
    }), alpha, rho


@pytest.fixture(scope='module')
def nested_logit_vertical_results():
    df, alpha, rho = _build_nested_vertical_dgp()
    pyRVtest.options.verbose = False
    problem = pyRVtest.Problem(
        cost_formulation=pyRVtest.Formulation('1 + z1'),
        instrument_formulation=pyRVtest.Formulation('0 + iv0'),
        product_data=df,
        demand_params={
            'alpha': alpha,
            'rho': [rho],
            'beta': np.array([0.0, 0.4]),
            'x_columns': ['intercept', 'x1'],
            'demand_instrument_columns': ['rival_x1', 'intercept', 'x1'],
            'nesting_ids_columns': ['nesting_ids'],
        },
        models=[
            pyRVtest.Vertical(
                downstream=pyRVtest.Bertrand(ownership='firm_ids'),
                upstream=pyRVtest.Monopoly(ownership='upstream_firm_ids'),
                vertical_integration='vi_col',
            ),
            pyRVtest.PerfectCompetition(),
        ],
    )
    return problem.solve(demand_adjustment=False, clustering_adjustment=False)


def test_snapshot_nested_logit_vertical(nested_logit_vertical_results):
    """Snapshot: nested-logit Vertical (Bertrand downstream + Monopoly
    upstream) on a 2-nest DGP. Exercises the nested-logit
    ``compute_hessian`` path (step 7) through the Villas-Boas vertical
    passthrough. Any future change to the nested-logit Hessian code,
    the passthrough helper, or the upstream markup assembly perturbs
    this snapshot.
    """
    assert_snapshot('nested_logit_vertical', nested_logit_vertical_results)
