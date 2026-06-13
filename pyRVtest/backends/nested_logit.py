"""NestedLogitBackend: analytical DemandBackend for L-level nested logit.

Split from `pyRVtest/backends/logit.py` for
user-facing clarity: tracebacks and generated API docs point at the
right file. `LogitBackend` stays in `logit.py` because it handles the
plain-logit (rho=[]) case; this module imports it as the parent class.

Theta order: [alpha, rho_1, rho_2, ..., rho_L].
`perturbed(0, delta)` shifts alpha; `perturbed(1+i, delta)` shifts rho[i].

The underlying closed-form derivatives come from AFSSZ equation (6) and
are implemented as module-level helpers in `pyRVtest/backends/logit.py`
(`compute_analytical_jacobian`, `compute_analytical_hessian`,
`_nested_logit_jacobian`, `_nested_logit_jacobian_derivative`).
`NestedLogitBackend` is a class wrapper that holds parameters and
shares, dispatches to those helpers, and implements the
`DemandBackend` protocol.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, List, Mapping, Optional

import numpy as np
from pyblp.utilities.basics import Array

from .logit import (
    LogitBackend,
    _infer_nesting_columns,
    compute_analytical_hessian,
    compute_analytical_jacobian,
)


__all__ = ['NestedLogitBackend']


class NestedLogitBackend(LogitBackend):
    """Analytical DemandBackend for L-level nested logit.

    Parameters are ordered: theta = [alpha, rho_1, rho_2, ..., rho_L].
    `perturbed(0, delta)` shifts alpha; `perturbed(1+i, delta)` shifts rho[i].

    rho values of exactly 0 are filtered at construction to match
    the ``compute_analytical_jacobian`` convention (a zero rho collapses that
    nesting level to plain logit, so it contributes nothing to theta).
    ``demand_moments`` / ``xi_gradient`` / ``jacobian_gradient`` are inherited
    unchanged from ``LogitBackend``; they key on ``self._rho`` (which
    ``LogitBackend.__init__`` defaults to ``[]`` and this class overrides with
    the filtered user-supplied list).

    Examples
    --------
    >>> import numpy as np
    >>> from pyRVtest.backends import NestedLogitBackend
    >>> product_data = {
    ...     'market_ids': np.array([0, 0, 0, 1, 1, 1]),
    ...     'shares': np.array([0.2, 0.2, 0.2, 0.3, 0.2, 0.2]),
    ...     'prices': np.array([1.0, 1.5, 2.0, 1.2, 1.8, 2.4]),
    ...     'nesting_ids': np.array([1, 1, 2, 1, 2, 2]),
    ... }
    >>> backend = NestedLogitBackend(
    ...     alpha=-1.0, rho=[0.5], product_data=product_data,
    ...     nesting_ids_columns=['nesting_ids'],
    ... )
    >>> backend.n_parameters
    2
    >>> backend.theta_names
    ['alpha', 'rho[0]']
    """

    def __init__(
            self, alpha: float, rho: List[float], product_data: Mapping,
            nesting_ids_columns: Optional[List[str]] = None,
            beta: Optional[Array] = None,
            x_columns: Optional[List[str]] = None,
            demand_instrument_columns: Optional[List[str]] = None,
            W_demand: Optional[Array] = None,
    ) -> None:
        super().__init__(
            alpha=alpha, product_data=product_data,
            beta=beta, x_columns=x_columns,
            demand_instrument_columns=demand_instrument_columns, W_demand=W_demand,
        )
        # Filter nonzero: a rho of exactly 0 is plain logit at that level.
        self._rho = [float(s) for s in rho if s > 0]
        self._nesting_ids_columns = nesting_ids_columns

    @property
    def n_parameters(self) -> int:
        return 1 + len(self._rho)

    @property
    def theta_names(self) -> List[str]:
        return ['alpha'] + [f'rho[{i}]' for i in range(len(self._rho))]

    def compute_jacobian(self, market_id: Any = None) -> Array:
        if self._jacobian_cache is None:
            self._jacobian_cache = compute_analytical_jacobian(
                self._alpha, self._rho, self._product_data,
                nesting_ids_columns=self._nesting_ids_columns,
                n_jobs=self._n_jobs,
            )
        full = self._jacobian_cache
        if market_id is None:
            return full
        # Reuse the inherited O(N log N) groupby cache (built once) instead of
        # an O(N) np.where per call -- the base LogitBackend.compute_jacobian
        # already does this; this override had regressed it back to per-call
        # scanning, making it O(N * n_markets) across demand adjustment.
        idx = self._ensure_market_indices()[market_id]
        block = full[idx]
        block = block[:, ~np.isnan(block).all(axis=0)]
        return block

    def compute_hessian(self, market_id: Any) -> Array:
        mids = np.asarray(self._product_data['market_ids']).flatten()
        idx = np.where(mids == market_id)[0]
        shares = np.asarray(self._product_data['shares']).flatten()[idx]
        rho_active = [r for r in self._rho if r > 0]
        nesting_t: List[Array] = []
        if rho_active:
            cols = self._nesting_ids_columns
            if cols is None:
                cols = _infer_nesting_columns(self._product_data, len(rho_active))
            for col in cols:
                arr = np.asarray(self._product_data[col]).flatten()
                nesting_t.append(arr[idx])
        return compute_analytical_hessian(self._alpha, rho_active, shares, nesting_t)

    @contextmanager
    def perturbed(self, theta_index: int, delta: float) -> Iterator['NestedLogitBackend']:
        n = self.n_parameters
        if theta_index < 0 or theta_index >= n:
            raise IndexError(
                f"Expected theta_index in [0, {n}) for NestedLogitBackend (has {n} "
                f"parameter(s): alpha followed by the rho nesting levels). "
                f"Received theta_index={theta_index}. "
                f"Fix: pass an integer in the valid range; see backend.theta_names "
                f"for names."
            )
        saved_alpha = self._alpha
        saved_rho = list(self._rho)
        saved_cache = self._jacobian_cache
        try:
            if theta_index == 0:
                self._alpha = saved_alpha + delta
            else:
                rho_index = theta_index - 1
                self._rho[rho_index] = saved_rho[rho_index] + delta
            self._jacobian_cache = None
            yield self
        finally:
            self._alpha = saved_alpha
            self._rho = saved_rho
            self._jacobian_cache = saved_cache
