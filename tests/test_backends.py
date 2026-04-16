"""Unit tests for the DemandBackend implementations (v0.4 step 3).

Each backend is tested independently: this file verifies the backend
produces the correct Jacobian / parameter enumeration / perturbation
behavior, WITHOUT routing through `_compute_markups`. Step 3e wires
the backends into the real pipeline; Step 0 snapshots + first-stage-
correction tests catch regressions there.
"""

from __future__ import annotations

import contextlib
import os

import numpy as np
import pandas as pd
import pyblp
import pytest

import pyRVtest
from pyRVtest.backends import (
    DemandBackend, LogitBackend, NestedLogitBackend, PyBLPBackend,
    SupportsDemandAdjustment,
)


# ---------------------------------------------------------------------------
# Shared PyBLP logit DGP (small, fast — reuse first_stage_correction pattern
# but with fewer markets for unit-test speed).
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def pyblp_logit_results():
    """Estimate a small logit demand for backend testing."""
    pyblp.options.verbose = False

    T, J = 40, 3
    market_ids = np.repeat(np.arange(T), J)
    firm_ids = np.tile(np.arange(J), T)
    id_data = pd.DataFrame({'market_ids': market_ids, 'firm_ids': firm_ids})

    X1 = pyblp.Formulation('1 + prices + x1')
    X3 = pyblp.Formulation('1 + z1')
    simulation = pyblp.Simulation(
        product_formulations=(X1, None, X3),
        beta=[0, -2, 1], gamma=[1, 0.5],
        xi_variance=0.2, omega_variance=0.2, correlation=0.0,
        product_data=id_data, seed=99,
    )
    sim_results = simulation.replace_endogenous()
    data = pd.DataFrame(pyblp.data_to_dict(sim_results.product_data))

    for t in range(T):
        idx = np.where(data['market_ids'] == t)[0]
        for j in idx:
            rival = [i for i in idx if i != j]
            data.loc[j, 'rival_x1'] = data.loc[rival, 'x1'].mean()
    data['demand_instruments0'] = data['rival_x1']

    problem = pyblp.Problem((X1,), product_data=data)
    return problem.solve(method='1s')


# ---------------------------------------------------------------------------
# PyBLPBackend
# ---------------------------------------------------------------------------

class TestPyBLPBackendBasics:
    """Construction, protocol conformance, and parameter enumeration."""

    def test_construction(self, pyblp_logit_results):
        backend = PyBLPBackend(pyblp_logit_results)
        assert backend is not None

    def test_satisfies_core_protocol(self, pyblp_logit_results):
        """Runtime isinstance check against the DemandBackend Protocol."""
        backend = PyBLPBackend(pyblp_logit_results)
        assert isinstance(backend, DemandBackend)

    def test_satisfies_demand_adjustment_mixin(self, pyblp_logit_results):
        """PyBLPBackend must implement SupportsDemandAdjustment."""
        backend = PyBLPBackend(pyblp_logit_results)
        assert isinstance(backend, SupportsDemandAdjustment)

    def test_alpha_is_enumerated(self, pyblp_logit_results):
        """For the logit DGP (alpha non-zero, no sigma/pi/rho), theta == ['alpha']."""
        backend = PyBLPBackend(pyblp_logit_results)
        assert backend.n_parameters >= 1
        assert 'alpha' in backend.theta_names


class TestPyBLPBackendJacobian:
    """compute_jacobian matches the inline PyBLP call used today."""

    def test_stacked_matches_compute_demand_jacobians(self, pyblp_logit_results):
        """backend.compute_jacobian() must match pyblp_results.compute_demand_jacobians()."""
        backend = PyBLPBackend(pyblp_logit_results)
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            expected = pyblp_logit_results.compute_demand_jacobians()
        actual = backend.compute_jacobian()
        np.testing.assert_allclose(actual, expected, atol=1e-14, equal_nan=True)

    def test_per_market_slice_matches_stacked(self, pyblp_logit_results):
        """compute_jacobian(market_id=t) returns the dense block for market t."""
        backend = PyBLPBackend(pyblp_logit_results)
        stacked = backend.compute_jacobian()
        mids = pyblp_logit_results.problem.products.market_ids.flatten()
        # Market id 0 should return a 3x3 block (J=3)
        block = backend.compute_jacobian(market_id=mids[0])
        assert block.shape == (3, 3)
        # No NaN in the dense block
        assert not np.isnan(block).any()
        # The stacked row-indices for market 0 should match the block
        rows_market_0 = stacked[mids == mids[0]]
        rows_nonan = rows_market_0[:, ~np.isnan(rows_market_0).all(axis=0)]
        np.testing.assert_allclose(block, rows_nonan, atol=1e-14)

    def test_cache_invalidated_after_perturb(self, pyblp_logit_results):
        """After perturb + restore, compute_jacobian returns the un-perturbed value."""
        backend = PyBLPBackend(pyblp_logit_results)
        before = backend.compute_jacobian().copy()
        # Perturb some parameter (use index 0 which must exist)
        with backend.perturbed(0, 0.01):
            during = backend.compute_jacobian()
            # During perturbation, the Jacobian should differ from the unperturbed one.
            assert not np.allclose(during, before, atol=1e-14, equal_nan=True)
        after = backend.compute_jacobian()
        np.testing.assert_allclose(after, before, atol=1e-14, equal_nan=True,
                                   err_msg="State not restored after perturbed() context exit")


class TestPyBLPBackendDemandMoments:
    """demand_moments returns the values DMSS eq. (77) expects."""

    def test_xi_matches_pyblp(self, pyblp_logit_results):
        """xi from demand_moments matches pyblp_results.xi."""
        backend = PyBLPBackend(pyblp_logit_results)
        xi, _, _ = backend.demand_moments()
        np.testing.assert_array_equal(xi, pyblp_logit_results.xi)

    def test_weight_matrix_is_W_not_updated_W_by_default(self, pyblp_logit_results):
        """Default option 'W' returns the weight used in estimation, not updated_W.

        This guards against a regression of Bug 3 (b3b08a3): DMSS eq. (77)
        specifies W (the weight actually used), not updated_W.
        """
        pyRVtest.options.demand_adjustment_weight = 'W'
        backend = PyBLPBackend(pyblp_logit_results)
        _, _, W_D = backend.demand_moments()
        np.testing.assert_array_equal(W_D, pyblp_logit_results.W)

    def test_weight_matrix_option_switches_to_updated_W(self, pyblp_logit_results):
        """Setting option to 'updated_W' toggles the weight source."""
        pyRVtest.options.demand_adjustment_weight = 'updated_W'
        try:
            backend = PyBLPBackend(pyblp_logit_results)
            _, _, W_D = backend.demand_moments()
            np.testing.assert_array_equal(W_D, pyblp_logit_results.updated_W)
        finally:
            pyRVtest.options.demand_adjustment_weight = 'W'


# ---------------------------------------------------------------------------
# LogitBackend (analytical, no PyBLP)
# ---------------------------------------------------------------------------

def _synthetic_logit_data(seed: int = 12345, T: int = 10, J: int = 3):
    """Build synthetic logit shares for LogitBackend unit tests."""
    rng = np.random.default_rng(seed=seed)
    market_ids = np.repeat(np.arange(T), J)
    firm_ids = np.tile(np.arange(J), T)
    # Generate logit shares directly from random utilities
    utilities = rng.normal(size=T * J)
    shares = np.zeros(T * J)
    for t in range(T):
        idx = np.where(market_ids == t)[0]
        u_t = utilities[idx]
        exp_u = np.exp(u_t)
        shares[idx] = exp_u / (1.0 + exp_u.sum())
    return pd.DataFrame({'market_ids': market_ids, 'firm_ids': firm_ids, 'shares': shares})


class TestLogitBackend:
    """Plain logit analytical backend."""

    def test_satisfies_core_protocol(self):
        data = _synthetic_logit_data()
        backend = LogitBackend(alpha=-2.0, product_data=data)
        assert isinstance(backend, DemandBackend)

    def test_does_not_satisfy_demand_adjustment(self):
        """LogitBackend does NOT yet implement SupportsDemandAdjustment (step 4 adds it)."""
        data = _synthetic_logit_data()
        backend = LogitBackend(alpha=-2.0, product_data=data)
        assert not isinstance(backend, SupportsDemandAdjustment)

    def test_n_parameters_is_one(self):
        data = _synthetic_logit_data()
        backend = LogitBackend(alpha=-2.0, product_data=data)
        assert backend.n_parameters == 1
        assert backend.theta_names == ['alpha']

    def test_jacobian_matches_module_level_function(self):
        """LogitBackend.compute_jacobian() matches compute_analytical_jacobian directly."""
        from pyRVtest.backends.logit import compute_analytical_jacobian
        data = _synthetic_logit_data()
        alpha = -2.0
        backend = LogitBackend(alpha=alpha, product_data=data)
        expected = compute_analytical_jacobian(alpha, [], data, nesting_ids_columns=None)
        actual = backend.compute_jacobian()
        np.testing.assert_allclose(actual, expected, atol=1e-14, equal_nan=True)

    def test_per_market_jacobian_shape(self):
        data = _synthetic_logit_data(T=10, J=3)
        backend = LogitBackend(alpha=-2.0, product_data=data)
        block = backend.compute_jacobian(market_id=0)
        assert block.shape == (3, 3)
        assert not np.isnan(block).any()

    def test_hessian_shape(self):
        data = _synthetic_logit_data(T=10, J=3)
        backend = LogitBackend(alpha=-2.0, product_data=data)
        H = backend.compute_hessian(market_id=0)
        assert H.shape == (3, 3, 3)

    def test_perturbed_shifts_alpha(self):
        """perturbed(0, delta) temporarily shifts alpha, then restores."""
        data = _synthetic_logit_data()
        backend = LogitBackend(alpha=-2.0, product_data=data)
        before = backend.compute_jacobian().copy()
        with backend.perturbed(0, 0.1):
            during = backend.compute_jacobian()
            assert not np.allclose(during, before, atol=1e-14, equal_nan=True)
        after = backend.compute_jacobian()
        np.testing.assert_allclose(after, before, atol=1e-14, equal_nan=True)

    def test_perturbed_rejects_invalid_index(self):
        data = _synthetic_logit_data()
        backend = LogitBackend(alpha=-2.0, product_data=data)
        with pytest.raises(IndexError):
            with backend.perturbed(1, 0.01):
                pass


# ---------------------------------------------------------------------------
# NestedLogitBackend (analytical, with nesting_ids)
# ---------------------------------------------------------------------------

def _synthetic_nested_logit_data(seed: int = 777, T: int = 10, J: int = 4):
    """Build synthetic nested-logit shares. Each market has J=4 products in 2 nests."""
    rng = np.random.default_rng(seed=seed)
    market_ids = np.repeat(np.arange(T), J)
    firm_ids = np.tile(np.arange(J), T)
    # Products 0, 1 in nest A; products 2, 3 in nest B (per market)
    nesting_ids = np.tile(['A', 'A', 'B', 'B'], T)
    utilities = rng.normal(size=T * J)
    shares = np.zeros(T * J)
    for t in range(T):
        idx = np.where(market_ids == t)[0]
        u_t = utilities[idx]
        exp_u = np.exp(u_t)
        shares[idx] = exp_u / (1.0 + exp_u.sum())
    return pd.DataFrame({
        'market_ids': market_ids, 'firm_ids': firm_ids,
        'shares': shares, 'nesting_ids': nesting_ids,
    })


class TestNestedLogitBackend:
    """L=1 nested logit analytical backend."""

    def test_satisfies_core_protocol(self):
        data = _synthetic_nested_logit_data()
        backend = NestedLogitBackend(alpha=-2.0, sigma=[0.3], product_data=data)
        assert isinstance(backend, DemandBackend)

    def test_n_parameters_is_two_for_single_nest(self):
        data = _synthetic_nested_logit_data()
        backend = NestedLogitBackend(alpha=-2.0, sigma=[0.3], product_data=data)
        assert backend.n_parameters == 2
        assert backend.theta_names == ['alpha', 'sigma[0]']

    def test_jacobian_matches_module_function(self):
        data = _synthetic_nested_logit_data()
        from pyRVtest.backends.logit import compute_analytical_jacobian
        alpha, sigma = -2.0, [0.3]
        backend = NestedLogitBackend(alpha=alpha, sigma=sigma, product_data=data)
        expected = compute_analytical_jacobian(alpha, sigma, data, nesting_ids_columns=['nesting_ids'])
        actual = backend.compute_jacobian()
        np.testing.assert_allclose(actual, expected, atol=1e-14, equal_nan=True)

    def test_perturbed_shifts_sigma(self):
        data = _synthetic_nested_logit_data()
        backend = NestedLogitBackend(alpha=-2.0, sigma=[0.3], product_data=data)
        before = backend.compute_jacobian().copy()
        with backend.perturbed(1, 0.05):  # perturb sigma[0]
            during = backend.compute_jacobian()
            assert not np.allclose(during, before, atol=1e-14, equal_nan=True)
        after = backend.compute_jacobian()
        np.testing.assert_allclose(after, before, atol=1e-14, equal_nan=True)
