"""PyBLPBackend: wraps pyblp.ProblemResults behind the DemandBackend protocol.

v0.4 step 3b. Encapsulates the private-attribute access (`_sigma`, `_pi`,
`_beta`, `_rho`, `_delta`) that was previously scattered across
`Problem._compute_demand_adjustment_gradient`. The rest of pyRVtest
sees only `compute_jacobian`, `compute_hessian`, `perturbed`, and the
demand-adjustment inputs (`demand_moments`, `xi_gradient`,
`jacobian_gradient`).

This file makes the class IMPORTABLE and unit-testable. Sub-step 3e
rewires `_compute_markups` and the demand-adjustment pipeline to use
it.

Parameter enumeration order matches the existing
`_compute_demand_adjustment_gradient`: sigma (flat in column-major order
over K2 × K2), then pi (flat over K2 × D), then alpha (the price
coefficient in beta), then rho (each entry for nested logit). This
preserves gradient-index compatibility with the current code.
"""

from __future__ import annotations

import contextlib
import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .. import options
from ..solve.demand_adjustment import _residualize_on_xd


__all__ = ['PyBLPBackend']


_NDArray = NDArray[Any]

# Parameter kind: identifies which PyBLP private attribute holds the parameter.
ParamIndex = Tuple[str, int, Optional[int]]  # (kind, i, j)


class PyBLPBackend:
    """Encapsulate a pyblp.ProblemResults behind the DemandBackend protocol.

    Attributes
    ----------
    _results : pyblp.ProblemResults
        The wrapped demand estimation result.
    _cache_jacobian : bool
        If True (default), the stacked Jacobian is cached after the
        first call and invalidated on `perturbed(...)`.

    Examples
    --------
    >>> from pyRVtest.backends import PyBLPBackend  # doctest: +SKIP
    >>> # Requires a fitted pyblp.ProblemResults object; see docs/tutorial.rst
    >>> # for an end-to-end example that builds one. A representative call is:
    >>> backend = PyBLPBackend(demand_results=pyblp_results)  # doctest: +SKIP
    >>> backend.n_parameters  # doctest: +SKIP
    """

    def __init__(self, demand_results: Any, cache_jacobian: bool = True) -> None:
        self._results = demand_results
        self._cache_jacobian = cache_jacobian
        self._jacobian_cache: Optional[_NDArray] = None
        self._theta_names: List[str]
        self._theta_indices: List[ParamIndex]
        self._theta_names, self._theta_indices = self._enumerate_theta()

    # -----------------------------------------------------------------
    # Parameter enumeration
    # -----------------------------------------------------------------

    def _enumerate_theta(self) -> Tuple[List[str], List[ParamIndex]]:
        """Flat list of (name, accessor) for non-zero demand parameters.

        Mirrors the iteration order in the current PyBLP finite-diff code:
        sigma (column-major flat), pi (column-major flat), alpha (price
        coefficient in beta), rho (one entry per nest).
        """
        names: List[str] = []
        indices: List[ParamIndex] = []
        r = self._results
        K2 = r.problem.K2
        D = r.problem.D

        # sigma: (K2 × K2) nonlinear random-coefficient variance parameters
        if K2 > 0:
            for i, j in np.ndindex(K2, K2):
                if r.sigma[i, j] != 0:
                    names.append(f'sigma[{i},{j}]')
                    indices.append(('sigma', int(i), int(j)))

        # pi: (K2 × D) demographic-interaction parameters
        if K2 > 0 and D > 0:
            for i, j in np.ndindex(K2, D):
                if r.pi[i, j] != 0:
                    names.append(f'pi[{i},{j}]')
                    indices.append(('pi', int(i), int(j)))

        # alpha: price coefficient in beta (identified by beta_label 'prices')
        price_idx = [k for k, v in enumerate(r.beta_labels) if v == 'prices']
        for k in price_idx:
            names.append('alpha')
            indices.append(('beta', int(k), None))

        # rho: nested-logit nesting parameters
        rho_arr = np.atleast_1d(np.asarray(r.rho).flatten())
        for k in range(rho_arr.size):
            names.append(f'rho[{k}]')
            indices.append(('rho', int(k), None))

        return names, indices

    @property
    def n_parameters(self) -> int:
        return len(self._theta_names)

    @property
    def theta_names(self) -> List[str]:
        return list(self._theta_names)

    # -----------------------------------------------------------------
    # Primitives
    # -----------------------------------------------------------------

    def compute_jacobian(self, market_id: Any = None) -> _NDArray:
        """Return ∂s/∂p, stacked or per-market.

        Stacked shape: (N, J_max) with NaN padding for jagged markets.
        Per-market: (J_t, J_t) dense block.
        """
        if self._jacobian_cache is None or not self._cache_jacobian:
            with contextlib.redirect_stdout(open(os.devnull, 'w')):
                full = self._results.compute_demand_jacobians()
            if self._cache_jacobian:
                self._jacobian_cache = full
        else:
            full = self._jacobian_cache

        if market_id is None:
            return full

        product_market_ids = self._results.problem.products.market_ids.flatten()
        idx = np.where(product_market_ids == market_id)[0]
        block = full[idx]
        # Drop NaN-padded columns corresponding to other markets
        block = block[:, ~np.isnan(block).all(axis=0)]
        return block

    def compute_hessian(self, market_id: Any) -> Optional[_NDArray]:
        """Return ∂²s/∂p² for one market. Shape (J_t, J_t, J_t)."""
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            return self._results.compute_demand_hessians(market_id=market_id)

    # -----------------------------------------------------------------
    # Demand adjustment (SupportsDemandAdjustment)
    # -----------------------------------------------------------------

    def demand_moments(self) -> Tuple[_NDArray, _NDArray, _NDArray]:
        """Return (xi, Z_D, W_D).

        W_D is the weight matrix ACTUALLY used in estimation (`r.W`),
        NOT the next-step efficient weight (`r.updated_W`). DMSS eq. (77)
        specifies `r.W`; using `updated_W` was Bug 3 in b3b08a3.
        """
        r = self._results
        Z_D = r.problem.products.ZD
        W_D = self._select_weight_matrix()
        xi = r.xi
        return xi, Z_D, W_D

    def _select_weight_matrix(self) -> _NDArray:
        """Respect pyRVtest.options.demand_adjustment_weight toggle.

        Default 'W' = DMSS-correct. 'updated_W' = pre-b3b08a3 reproduction
        flag (see options.py for rationale and pRV.MEMO_coauthor_updates).
        """
        weight_choice = options.demand_adjustment_weight
        if weight_choice == 'W':
            return self._results.W
        if weight_choice == 'updated_W':
            return self._results.updated_W
        raise ValueError(
            f"Expected pyRVtest.options.demand_adjustment_weight to be 'W' "
            f"(DMSS-correct) or 'updated_W' (legacy pre-b3b08a3 reproduction). "
            f"Received {weight_choice!r}. "
            f"Fix: set pyRVtest.options.demand_adjustment_weight = 'W'."
        )

    def xi_gradient(self) -> _NDArray:
        """Return d(xi)/d(theta) with exogenous-regressor concentration.

        Mirrors the current `_compute_demand_adjustment_gradient` logic:
        build the raw PyBLP xi-by-theta Jacobian, append -prices column if
        prices is a linear parameter, absorb demand-side FEs, then
        2SLS-residualize on X_D (the non-price linear regressors) to
        profile beta out per DMSS Appendix C.
        """
        r = self._results
        XD = r.problem.products.X1
        XD_column_names = r.problem.products.dtype.fields['X1'][2]
        price_in_linear_parameters = 'prices' in XD_column_names

        if price_in_linear_parameters:
            XD = np.delete(XD, XD_column_names.index('prices'), 1)

        partial_y_theta: _NDArray
        if price_in_linear_parameters:
            partial_y_theta = np.append(
                r.xi_by_theta_jacobian, -r.problem.products.prices, axis=1
            )
        else:
            partial_y_theta = r.xi_by_theta_jacobian

        if r.problem.ED > 0:
            absorbed, _ = r.problem._absorb_demand_ids(partial_y_theta)
            n_theta = len(r.theta) + int(price_in_linear_parameters)
            N = r.problem.products.prices.shape[0]
            partial_y_theta = np.reshape(absorbed, [N, n_theta])

        ZD = r.problem.products.ZD
        WD = self._select_weight_matrix()
        # v0.4 step 4b: shared 2SLS-residualize helper (single source of truth).
        return _residualize_on_xd(partial_y_theta, XD, ZD, WD)

    def jacobian_gradient(self, market_id: Any) -> _NDArray:
        """Finite-difference approximation of d(D)/d(theta) in one market.

        Uses `perturbed` context manager for +/- epsilon/2 steps. `PyBLP`
        does not expose analytical derivatives of its demand Jacobian
        w.r.t. theta, so this backend uses finite-difference.

        .. note::
            **Precision analysis.**

            * **Plain logit (K2=0, rho.size=0)**: ``compute_delta()`` in
              ``perturbed`` restores observed shares, so D is linear in
              alpha at the evaluation point. Finite-diff therefore equals
              the exact derivative up to ULP. ``Problem._construct_demand_backend``
              keeps plain logit on ``PyBLPBackend`` since there is no
              precision gain from routing to ``LogitBackend``.

            * **Single-scalar-rho nested logit (K2=0, r.rho.size=1)**:
              pyblp's single scalar ``rho`` matches the AFSSZ L=1 nested
              logit ``sigma_1``. D is *nonlinear* in rho, so pyblp's
              finite-diff has genuine O(eps^2) truncation error.
              ``Problem._construct_demand_backend`` auto-routes this case
              to ``NestedLogitBackend(sigma=[rho])`` for exact
              ``d(D)/d(rho)``.

            * **Per-nest rho nested logit (K2=0, r.rho.size>1)**: pyblp's
              per-nest Cardell-Nevo parametrization gives per-nest
              ``rho_h`` parameters. The AFSSZ L-level formulation in
              ``_nested_logit_jacobian`` has one ``sigma_l`` per *level*,
              not per nest. ``Problem._construct_demand_backend`` keeps
              these on ``PyBLPBackend`` (finite-diff with O(eps^2) error
              on ``d(D)/d(rho_h)``) because the two parametrizations do
              not align.

            * **BLP (K2>0)**: the BLP contraction mapping makes D a
              nonlinear function of sigma, pi, (and rho when used
              jointly). Finite-diff has genuine O(eps^2) truncation error.
              Deriving analytical d(D_BLP)/d(theta) through the
              contraction is a research exercise; this package does not
              currently do it.
        """
        eps = options.finite_differences_epsilon
        base_shape = self.compute_jacobian(market_id).shape  # (J_t, J_t)
        J_t = base_shape[0]
        grad: _NDArray = np.zeros((J_t, J_t, self.n_parameters), dtype=options.dtype)
        for k in range(self.n_parameters):
            with self.perturbed(k, +eps / 2):
                jac_plus = self.compute_jacobian(market_id)
            with self.perturbed(k, -eps / 2):
                jac_minus = self.compute_jacobian(market_id)
            grad[:, :, k] = (jac_plus - jac_minus) / eps
        return grad

    # -----------------------------------------------------------------
    # Perturbation context manager
    # -----------------------------------------------------------------

    @contextmanager
    def perturbed(self, theta_index: int, delta: float) -> Iterator['PyBLPBackend']:
        """Apply `delta` to theta[theta_index]; restore state on exit.

        State restoration uses try/finally so the PyBLP results object is
        never left in a perturbed state on exception (same guarantee as
        the try/finally added in commit a9e37f8 for the inline code).
        """
        kind, i, j = self._theta_indices[theta_index]
        saved = self._save_state()
        try:
            self._apply_perturbation(kind, i, j, delta)
            self._recompute_delta()
            self._jacobian_cache = None  # invalidate
            yield self
        finally:
            self._restore_state(saved)
            self._jacobian_cache = None

    def _save_state(self) -> Dict[str, Any]:
        r = self._results
        return {
            'sigma': np.asarray(r.sigma).copy(),
            'pi': np.asarray(r.pi).copy(),
            'beta': np.asarray(r.beta).copy(),
            'rho': np.asarray(r.rho).copy() if np.asarray(r.rho).size > 0 else None,
            'delta': np.asarray(r.delta).copy(),
        }

    def _apply_perturbation(self, kind: str, i: int, j: Optional[int], delta: float) -> None:
        r = self._results
        if kind == 'sigma':
            assert j is not None
            r._sigma[i, j] = r._sigma[i, j] + delta
        elif kind == 'pi':
            assert j is not None
            r._pi[i, j] = r._pi[i, j] + delta
        elif kind == 'beta':
            r._beta[i] = r._beta[i] + delta
        elif kind == 'rho':
            # rho may be stored as a 1D array or a 2D column vector.
            if np.asarray(r._rho).ndim == 1:
                r._rho[i] = r._rho[i] + delta
            else:
                r._rho[i, 0] = r._rho[i, 0] + delta
        else:
            raise ValueError(
                f"pyRVtest internal error: expected parameter kind in "
                f"{{'sigma', 'pi', 'beta', 'rho'}} for perturbation dispatch; "
                f"received {kind!r}."
            )

    def _recompute_delta(self) -> None:
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            self._results._delta = self._results.compute_delta()

    def _restore_state(self, saved: Dict[str, Any]) -> None:
        r = self._results
        r._sigma[:] = saved['sigma']
        r._pi[:] = saved['pi']
        r._beta[:] = saved['beta']
        if saved['rho'] is not None:
            r._rho[:] = saved['rho']
        r._delta = saved['delta']
