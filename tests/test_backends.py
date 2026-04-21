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
    SupportsDemandAdjustment, UserSuppliedBackend,
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
        """For the logit DGP (alpha non-zero, no rho/pi), theta == ['alpha']."""
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

    def test_satisfies_demand_adjustment_structurally(self):
        """v0.4 step 4c: LogitBackend now structurally satisfies SupportsDemandAdjustment.

        The three protocol methods (demand_moments, xi_gradient, jacobian_gradient) are
        defined on the class. When the demand-adjustment state was not supplied at
        construction, the methods raise a clear ValueError — see
        ``test_demand_adjustment_methods_raise_without_state`` below.
        """
        data = _synthetic_logit_data()
        backend = LogitBackend(alpha=-2.0, product_data=data)
        assert isinstance(backend, SupportsDemandAdjustment)

    def test_demand_adjustment_methods_raise_without_state(self):
        """Without beta / x_columns / demand_instrument_columns, the methods error clearly."""
        data = _synthetic_logit_data()
        backend = LogitBackend(alpha=-2.0, product_data=data)
        with pytest.raises(ValueError, match="demand-adjustment state"):
            backend.demand_moments()
        with pytest.raises(ValueError, match="demand-adjustment state"):
            backend.xi_gradient()
        # jacobian_gradient does NOT depend on demand-adjustment state, so it must
        # still work; this is asserted elsewhere.

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
        backend = NestedLogitBackend(alpha=-2.0, rho=[0.3], product_data=data)
        assert isinstance(backend, DemandBackend)

    def test_n_parameters_is_two_for_single_nest(self):
        data = _synthetic_nested_logit_data()
        backend = NestedLogitBackend(alpha=-2.0, rho=[0.3], product_data=data)
        assert backend.n_parameters == 2
        assert backend.theta_names == ['alpha', 'rho[0]']

    def test_jacobian_matches_module_function(self):
        data = _synthetic_nested_logit_data()
        from pyRVtest.backends.logit import compute_analytical_jacobian
        alpha, rho = -2.0, [0.3]
        backend = NestedLogitBackend(alpha=alpha, rho=rho, product_data=data)
        expected = compute_analytical_jacobian(alpha, rho, data, nesting_ids_columns=['nesting_ids'])
        actual = backend.compute_jacobian()
        np.testing.assert_allclose(actual, expected, atol=1e-14, equal_nan=True)

    def test_perturbed_shifts_rho(self):
        data = _synthetic_nested_logit_data()
        backend = NestedLogitBackend(alpha=-2.0, rho=[0.3], product_data=data)
        before = backend.compute_jacobian().copy()
        with backend.perturbed(1, 0.05):  # perturb rho[0]
            during = backend.compute_jacobian()
            assert not np.allclose(during, before, atol=1e-14, equal_nan=True)
        after = backend.compute_jacobian()
        np.testing.assert_allclose(after, before, atol=1e-14, equal_nan=True)


# ---------------------------------------------------------------------------
# UserSuppliedBackend
# ---------------------------------------------------------------------------

class TestUserSuppliedBackend:
    """Escape-hatch backend with a precomputed Jacobian."""

    def _build(self, T=5, J=3):
        market_ids = np.repeat(np.arange(T), J)
        # NaN-padded (N, J_max) Jacobian. Since all markets have J=3 == J_max
        # we don't need NaN padding, but the structure must still be (N, J).
        jacobian = np.random.default_rng(seed=42).normal(size=(T * J, J))
        return jacobian, market_ids

    def test_satisfies_core_protocol(self):
        jac, mids = self._build()
        backend = UserSuppliedBackend(jacobian=jac, market_ids=mids)
        assert isinstance(backend, DemandBackend)

    def test_does_not_satisfy_demand_adjustment(self):
        """UserSuppliedBackend deliberately does not implement SupportsDemandAdjustment."""
        jac, mids = self._build()
        backend = UserSuppliedBackend(jacobian=jac, market_ids=mids)
        assert not isinstance(backend, SupportsDemandAdjustment)

    def test_zero_parameters_by_default(self):
        jac, mids = self._build()
        backend = UserSuppliedBackend(jacobian=jac, market_ids=mids)
        assert backend.n_parameters == 0
        assert backend.theta_names == []

    def test_compute_jacobian_returns_stored_array(self):
        jac, mids = self._build()
        backend = UserSuppliedBackend(jacobian=jac, market_ids=mids)
        np.testing.assert_array_equal(backend.compute_jacobian(), jac)

    def test_compute_hessian_returns_none_by_default(self):
        jac, mids = self._build()
        backend = UserSuppliedBackend(jacobian=jac, market_ids=mids)
        assert backend.compute_hessian(market_id=0) is None

    def test_perturb_raises_when_no_parameters(self):
        jac, mids = self._build()
        backend = UserSuppliedBackend(jacobian=jac, market_ids=mids)
        with pytest.raises(NotImplementedError, match='without theta_names'):
            with backend.perturbed(0, 0.1):
                pass

    def test_perturb_raises_when_no_callback(self):
        jac, mids = self._build()
        backend = UserSuppliedBackend(
            jacobian=jac, market_ids=mids, theta_names=['alpha']
        )
        with pytest.raises(NotImplementedError, match='perturb_callback'):
            with backend.perturbed(0, 0.1):
                pass

    def test_perturb_with_callback_yields_new_backend(self):
        jac, mids = self._build()

        def callback(idx: int, delta: float) -> UserSuppliedBackend:
            new_jac = jac * (1.0 + delta)
            return UserSuppliedBackend(jacobian=new_jac, market_ids=mids, theta_names=['alpha'])

        backend = UserSuppliedBackend(
            jacobian=jac, market_ids=mids,
            theta_names=['alpha'], perturb_callback=callback
        )
        with backend.perturbed(0, 0.01) as perturbed_backend:
            # Perturbed backend returns the scaled Jacobian
            np.testing.assert_allclose(
                perturbed_backend.compute_jacobian(), jac * 1.01, atol=1e-14
            )
        # Original backend is unchanged
        np.testing.assert_array_equal(backend.compute_jacobian(), jac)

    def test_construction_validates_shapes(self):
        with pytest.raises(ValueError, match='2-D'):
            UserSuppliedBackend(
                jacobian=np.zeros(6), market_ids=np.arange(6),
            )
        with pytest.raises(ValueError, match='must match'):
            UserSuppliedBackend(
                jacobian=np.zeros((6, 3)), market_ids=np.arange(10),
            )


# ---------------------------------------------------------------------------
# Backend equivalence: the demand_backend path in _compute_markups produces
# the same markups as the existing pyblp_results path (v0.4 step 3e).
# ---------------------------------------------------------------------------

class TestBackendEquivalenceInComputeMarkups:
    """Verify the new demand_backend parameter of _compute_markups.

    The plan's step 3e wires backends into the markups pipeline. To
    guard against drift, these tests call _compute_markups BOTH ways
    on the same DGP and assert byte-identical markups.
    """

    def test_pyblp_backend_matches_pyblp_results_path(self, pyblp_logit_results):
        """_compute_markups(pyblp_results=X) == _compute_markups(demand_backend=PyBLPBackend(X))."""
        from pyRVtest.markups import _compute_markups
        from pyRVtest.problem import Models, Products

        data_dict = {
            'market_ids': pyblp_logit_results.problem.products.market_ids,
            'firm_ids': pyblp_logit_results.problem.products.firm_ids,
            'shares': pyblp_logit_results.problem.products.shares,
            'prices': pyblp_logit_results.problem.products.prices,
        }
        product_data = pd.DataFrame({k: np.asarray(v).flatten() for k, v in data_dict.items()})

        # Build a minimal Models object with a Bertrand formulation
        models_formulations = (
            pyRVtest.ModelFormulation(
                model_downstream='bertrand', ownership_downstream='firm_ids',
            ),
        )
        models = Models(models_formulations, product_data)

        # Path A: existing pyblp_results path
        markups_A, _, _ = _compute_markups(
            product_data=product_data, pyblp_results=pyblp_logit_results,
            model_downstream=models['models_downstream'],
            ownership_downstream=models['ownership_downstream'],
        )

        # Path B: new demand_backend path
        backend = PyBLPBackend(pyblp_logit_results)
        markups_B, _, _ = _compute_markups(
            product_data=product_data, pyblp_results=pyblp_logit_results,
            model_downstream=models['models_downstream'],
            ownership_downstream=models['ownership_downstream'],
            demand_backend=backend,
        )

        # Markups must match to machine precision
        np.testing.assert_allclose(
            markups_A[0], markups_B[0], atol=1e-14,
            err_msg="PyBLPBackend path diverges from direct pyblp_results path"
        )

    def test_pyblp_backend_vertical_matches_pyblp_results_path(self, pyblp_logit_results):
        """v0.4 step 4a: vertical (upstream+downstream) markups byte-identical via PyBLPBackend.

        Vertical models require `_compute_markups` to pull a Hessian. Before step 4a the
        only path was `construct_passthrough_matrix(pyblp_results, t, ...)`; step 4a routes
        through `demand_backend.compute_hessian(market_id=t)` when a backend is supplied.
        PyBLPBackend.compute_hessian calls the same `pyblp_results.compute_demand_hessians`
        internally, so the two paths must be byte-identical.
        """
        from pyRVtest.markups import _compute_markups
        from pyRVtest.problem import Models

        data_dict = {
            'market_ids': pyblp_logit_results.problem.products.market_ids,
            'firm_ids': pyblp_logit_results.problem.products.firm_ids,
            'shares': pyblp_logit_results.problem.products.shares,
            'prices': pyblp_logit_results.problem.products.prices,
        }
        product_data = pd.DataFrame({k: np.asarray(v).flatten() for k, v in data_dict.items()})
        # Upstream ownership: pretend a single upstream manufacturer serves all products.
        product_data['upstream_firm_ids'] = np.ones(len(product_data), dtype=int)
        product_data['vi_col'] = np.zeros(len(product_data), dtype=int)

        models_formulations = (
            pyRVtest.ModelFormulation(
                model_downstream='bertrand', ownership_downstream='firm_ids',
                model_upstream='monopoly', ownership_upstream='upstream_firm_ids',
                vertical_integration='vi_col',
            ),
        )
        models = Models(models_formulations, product_data)

        # Path A: legacy pyblp_results vertical path
        markups_A, down_A, up_A = _compute_markups(
            product_data=product_data, pyblp_results=pyblp_logit_results,
            model_downstream=models['models_downstream'],
            ownership_downstream=models['ownership_downstream'],
            model_upstream=models['models_upstream'],
            ownership_upstream=models['ownership_upstream'],
            vertical_integration=models['vertical_integration'],
        )

        # Path B: new demand_backend vertical path
        backend = PyBLPBackend(pyblp_logit_results)
        markups_B, down_B, up_B = _compute_markups(
            product_data=product_data, pyblp_results=pyblp_logit_results,
            model_downstream=models['models_downstream'],
            ownership_downstream=models['ownership_downstream'],
            model_upstream=models['models_upstream'],
            ownership_upstream=models['ownership_upstream'],
            vertical_integration=models['vertical_integration'],
            demand_backend=backend,
        )

        np.testing.assert_allclose(
            markups_A[0], markups_B[0], atol=1e-14,
            err_msg="Vertical: PyBLPBackend diverges from legacy pyblp_results path"
        )
        np.testing.assert_allclose(
            down_A[0], down_B[0], atol=1e-14,
            err_msg="Vertical downstream: PyBLPBackend diverges from legacy path"
        )
        np.testing.assert_allclose(
            up_A[0], up_B[0], atol=1e-14,
            err_msg="Vertical upstream: PyBLPBackend diverges from legacy path"
        )

    def test_user_supplied_backend_without_hessian_raises_on_vertical(self):
        """v0.4 step 4a: UserSuppliedBackend without hessian_fn raises on vertical models.

        Vertical (upstream) markups require a Hessian. UserSuppliedBackend.compute_hessian
        returns None by default. The vertical code path in `_compute_markups` raises a
        clear error naming the backend and offering the fix.
        """
        from pyRVtest.markups import _compute_markups
        from pyRVtest.problem import Models

        rng = np.random.default_rng(seed=99)
        T, J = 4, 3
        N = T * J
        market_ids = np.repeat(np.arange(T), J)
        downstream_firm_ids = np.tile([0, 1, 1], T)
        upstream_firm_ids = np.ones(N, dtype=int)
        utilities = rng.normal(size=N)
        shares = np.zeros(N)
        for t in range(T):
            idx = np.where(market_ids == t)[0]
            exp_u = np.exp(utilities[idx])
            shares[idx] = exp_u / (1.0 + exp_u.sum())
        prices = rng.uniform(1.0, 3.0, size=N)
        df = pd.DataFrame({
            'market_ids': market_ids,
            'firm_ids': downstream_firm_ids,
            'upstream_firm_ids': upstream_firm_ids,
            'shares': shares,
            'prices': prices,
            'vi_col': np.zeros(N, dtype=int),
        })
        product_data = df.to_records(index=False)

        # Build a stacked NaN-padded Jacobian so UserSuppliedBackend doesn't blow up earlier.
        jacobian = np.full((N, J), np.nan)
        for t in range(T):
            idx = np.where(market_ids == t)[0]
            s_t = shares[idx]
            D_t = -2.0 * (np.diag(s_t) - np.outer(s_t, s_t))
            jacobian[idx[:, None], np.arange(len(idx))[None, :]] = D_t

        backend = UserSuppliedBackend(jacobian=jacobian, market_ids=market_ids)

        models_formulations = (
            pyRVtest.ModelFormulation(
                model_downstream='bertrand', ownership_downstream='firm_ids',
                model_upstream='monopoly', ownership_upstream='upstream_firm_ids',
                vertical_integration='vi_col',
            ),
        )
        models = Models(models_formulations, product_data)

        with pytest.raises(ValueError, match="returned None"):
            _compute_markups(
                product_data=product_data, pyblp_results=None,
                model_downstream=models['models_downstream'],
                ownership_downstream=models['ownership_downstream'],
                model_upstream=models['models_upstream'],
                ownership_upstream=models['ownership_upstream'],
                vertical_integration=models['vertical_integration'],
                demand_backend=backend,
            )


class TestComputeMarkupsAcceptsDataFrameProductData:
    """v0.4 step 4g: regression test for DataFrame product_data.

    Before step 4g, `_compute_markups`'s inline analytical-Hessian branch
    called `shares_t.flatten()`, which broke when the branch received a
    pandas Series (from DataFrame product_data, e.g., a direct caller
    bypassing `Problem`). The branch has since been removed; the surviving
    path wraps `shares_t` with `np.asarray` for robustness. This test
    exercises the vertical-markups path with a DataFrame product_data and
    verifies it runs cleanly.
    """

    def test_vertical_markups_with_dataframe_product_data(self):
        from pyRVtest.markups import _compute_markups
        from pyRVtest.problem import Models

        rng = np.random.default_rng(seed=4242)
        T, J = 5, 4
        N = T * J
        market_ids = np.repeat(np.arange(T), J)
        downstream_firm_ids = np.tile([0, 0, 1, 1], T)
        upstream_firm_ids = np.ones(N, dtype=int)
        utilities = rng.normal(size=N)
        shares = np.zeros(N)
        for t in range(T):
            idx = np.where(market_ids == t)[0]
            e = np.exp(utilities[idx])
            shares[idx] = e / (1.0 + e.sum())
        prices = rng.uniform(1.0, 3.0, size=N)
        df = pd.DataFrame({
            'market_ids': market_ids, 'firm_ids': downstream_firm_ids,
            'upstream_firm_ids': upstream_firm_ids,
            'shares': shares, 'prices': prices,
            'vi_col': np.zeros(N, dtype=int),
        })

        alpha = -2.0
        backend = LogitBackend(alpha=alpha, product_data=df)
        models_formulations = (
            pyRVtest.ModelFormulation(
                model_downstream='bertrand', ownership_downstream='firm_ids',
                model_upstream='monopoly', ownership_upstream='upstream_firm_ids',
                vertical_integration='vi_col',
            ),
        )
        models = Models(models_formulations, df)

        # Should run without AttributeError on .flatten().
        markups, _, _ = _compute_markups(
            product_data=df, pyblp_results=None,
            model_downstream=models['models_downstream'],
            ownership_downstream=models['ownership_downstream'],
            model_upstream=models['models_upstream'],
            ownership_upstream=models['ownership_upstream'],
            vertical_integration=models['vertical_integration'],
            demand_backend=backend,
        )
        assert markups[0] is not None
        assert markups[0].shape == (N, 1)
        assert np.all(np.isfinite(markups[0]))


class TestAnalyticalNestedLogitHessian:
    """v0.4 step 7: analytical dD/ds for plain and 1-level nested logit
    must match the prior finite-diff implementation on random fixtures.

    For the plain-logit case, target atol=1e-8. For the 1-level nested
    case, target atol=1e-7 (the finite-diff reference is itself noisy
    at O(eps) around eps=1e-7). The 2-level case falls back to
    finite-diff and is only checked for basic shape / finiteness.
    """

    @staticmethod
    def _finite_diff_hessian(alpha, rho, s, nesting, eps=1e-7):
        """Reference: the pre-step-7 implementation (centered finite-diff)."""
        from pyRVtest.backends.logit import _logit_jacobian, _nested_logit_jacobian
        J = len(s)
        if len(rho) == 0 or all(r == 0 for r in rho):
            D_func = lambda s_: _logit_jacobian(alpha, s_)
        else:
            D_func = lambda s_: _nested_logit_jacobian(alpha, rho, s_, nesting)
        D = D_func(s)
        dD_ds = np.zeros((J, J, J))
        for r in range(J):
            sp = s.copy(); sp[r] += eps / 2
            sm = s.copy(); sm[r] -= eps / 2
            dD_ds[:, :, r] = (D_func(sp) - D_func(sm)) / eps
        return np.einsum('jkr,rl->jkl', dD_ds, D)

    @staticmethod
    def _random_shares(rng, J):
        """Draw a (J,) share vector with shares summing to < 1 (room for s0)."""
        w = rng.dirichlet(np.ones(J + 1))
        return w[:J]

    @pytest.mark.parametrize("seed,J", [(0, 3), (1, 4), (2, 5), (3, 6), (4, 4)])
    def test_plain_logit_matches_finite_diff(self, seed, J):
        from pyRVtest.backends.logit import compute_analytical_hessian
        rng = np.random.default_rng(seed)
        s = self._random_shares(rng, J)
        alpha = -2.0 - rng.uniform()

        H_new = compute_analytical_hessian(alpha, [], s, [])
        H_ref = self._finite_diff_hessian(alpha, [], s, [])

        assert H_new.shape == (J, J, J)
        assert np.allclose(H_new, H_ref, atol=1e-8)

    def test_plain_logit_matches_finite_diff_zero_rho(self):
        """rho=[0.0] is treated as plain logit (same convention as the
        Jacobian); the analytical branch must still fire."""
        from pyRVtest.backends.logit import compute_analytical_hessian
        rng = np.random.default_rng(7)
        J = 5
        s = self._random_shares(rng, J)
        alpha = -2.5
        nest_ids = np.array([0, 0, 1, 1, 1])

        H_new = compute_analytical_hessian(alpha, [0.0], s, [nest_ids])
        H_ref = self._finite_diff_hessian(alpha, [0.0], s, [nest_ids])
        assert np.allclose(H_new, H_ref, atol=1e-8)

    @pytest.mark.parametrize(
        "seed,J,rho,nest_ids",
        [
            (10, 4, 0.3, np.array([0, 0, 1, 1])),
            (11, 6, 0.5, np.array([0, 0, 0, 1, 1, 1])),
            (12, 4, 0.7, np.array([0, 1, 0, 1])),
            (13, 6, 0.1, np.array([0, 1, 2, 0, 1, 2])),
            (14, 6, 0.9, np.array([0, 0, 1, 1, 2, 2])),
        ],
    )
    def test_one_level_nested_matches_finite_diff(self, seed, J, rho, nest_ids):
        from pyRVtest.backends.logit import compute_analytical_hessian
        rng = np.random.default_rng(seed)
        s = self._random_shares(rng, J)
        alpha = -2.0

        H_new = compute_analytical_hessian(alpha, [rho], s, [nest_ids])
        H_ref = self._finite_diff_hessian(alpha, [rho], s, [nest_ids])

        assert H_new.shape == (J, J, J)
        assert np.allclose(H_new, H_ref, atol=1e-7)

    def test_two_level_still_works(self):
        """Two-level nested logit falls back to finite-diff; verify the
        function runs and produces a valid (J, J, J) tensor."""
        from pyRVtest.backends.logit import compute_analytical_hessian
        rng = np.random.default_rng(100)
        J = 6
        s = self._random_shares(rng, J)
        alpha = -2.0
        nest1 = np.array([0, 0, 1, 1, 2, 2])
        nest2 = np.array([0, 0, 0, 0, 1, 1])

        H = compute_analytical_hessian(alpha, [0.3, 0.6], s, [nest1, nest2])
        assert H.shape == (J, J, J)
        assert np.all(np.isfinite(H))
