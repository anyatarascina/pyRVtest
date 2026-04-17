"""UserSuppliedBackend: escape hatch for researcher-supplied demand systems.

v0.4 step 3d. Wraps a precomputed stacked Jacobian (and optionally a
per-market Hessian callable) into the DemandBackend protocol.

This backend implements the CORE protocol (compute_jacobian,
compute_hessian, perturbed, n_parameters, theta_names) but deliberately
does NOT implement SupportsDemandAdjustment. Researchers who want
first-stage adjustment with a custom demand system supply the adjustment
inputs through a separate interface (planned for step 15's worked
example with an Almagro-Sood-style labor supply).

Typical usage:

    jacobian = my_custom_demand_code(product_data, theta)
    backend = UserSuppliedBackend(jacobian=jacobian, market_ids=...)
    Problem(..., demand_backend=backend).solve()

The `perturbed` context manager raises `NotImplementedError` by default
because the backend has no built-in way to re-evaluate the demand system
at a perturbed theta. Users who need finite-diff corrections supply a
`perturb_callback` at construction time (receives a copy of self and
the perturbation, returns the perturbed Jacobian).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator, List, Optional

import numpy as np
from pyblp.utilities.basics import Array


__all__ = ['UserSuppliedBackend']


class UserSuppliedBackend:
    """Wrap a user-computed demand Jacobian as a DemandBackend.

    Parameters
    ----------
    jacobian
        Stacked (N, J_max) NaN-padded demand Jacobian, same format as
        pyblp.ProblemResults.compute_demand_jacobians() returns.
    market_ids
        (N,) array aligning each row of `jacobian` to a market id.
        Required for per-market slicing in compute_jacobian(market_id=...).
    hessian_fn
        Optional callable `market_id -> (J_t, J_t, J_t) Hessian`. If
        None, compute_hessian returns None and vertical-integration
        passthrough falls back to finite differences on the Jacobian.
    theta_names
        Optional human-readable parameter names. If provided, defines
        n_parameters = len(theta_names). If None, n_parameters = 0 (the
        backend is treated as having fixed parameters — perturb will
        raise).
    perturb_callback
        Optional callable `(theta_index: int, delta: float) -> UserSuppliedBackend`
        that returns a new backend with the perturbed parameter applied.
        If None and n_parameters > 0, perturbed raises
        NotImplementedError with a message asking the user to supply
        this callback.

    Examples
    --------
    >>> import numpy as np
    >>> from pyRVtest.backends import UserSuppliedBackend
    >>> # Stacked (N, J_max) Jacobian across two 2-product markets.
    >>> jac = np.array([[-1.0, 0.1], [0.1, -1.0], [-0.8, 0.2], [0.2, -0.8]])
    >>> market_ids = np.array([0, 0, 1, 1])
    >>> backend = UserSuppliedBackend(jacobian=jac, market_ids=market_ids)
    >>> backend.n_parameters
    0
    >>> backend.theta_names
    []
    >>> backend.compute_jacobian(market_id=0)
    array([[-1. ,  0.1],
           [ 0.1, -1. ]])
    >>> backend.compute_hessian(market_id=0) is None
    True
    """

    def __init__(
            self,
            jacobian: Array,
            market_ids: Array,
            hessian_fn: Optional[Callable[[Any], Array]] = None,
            theta_names: Optional[List[str]] = None,
            perturb_callback: Optional[Callable[[int, float], 'UserSuppliedBackend']] = None,
    ) -> None:
        if jacobian.ndim != 2:
            raise ValueError(
                f"Expected jacobian to be a 2-D (N, J_max) NaN-padded stacked "
                f"demand Jacobian. "
                f"Received an array with shape {jacobian.shape} (ndim={jacobian.ndim}). "
                f"Fix: stack per-market Jacobians into an (N, J_max) array, padding "
                f"with NaN where a market has fewer than J_max products."
            )
        market_ids_1d = np.asarray(market_ids).flatten()
        if market_ids_1d.shape[0] != jacobian.shape[0]:
            raise ValueError(
                f"Expected market_ids length to match the number of jacobian rows "
                f"(one market id per product). "
                f"Received market_ids of length {market_ids_1d.shape[0]} and "
                f"jacobian with {jacobian.shape[0]} rows; these must match. "
                f"Fix: rebuild market_ids so it has one entry per product row of "
                f"the stacked Jacobian."
            )
        self._jacobian = jacobian
        self._market_ids = market_ids_1d
        self._hessian_fn = hessian_fn
        self._theta_names: List[str] = list(theta_names) if theta_names is not None else []
        self._perturb_callback = perturb_callback

    @property
    def n_parameters(self) -> int:
        return len(self._theta_names)

    @property
    def theta_names(self) -> List[str]:
        return list(self._theta_names)

    def compute_jacobian(self, market_id: Any = None) -> Array:
        if market_id is None:
            return self._jacobian
        idx = np.where(self._market_ids == market_id)[0]
        block = self._jacobian[idx]
        block = block[:, ~np.isnan(block).all(axis=0)]
        return block

    def compute_hessian(self, market_id: Any) -> Optional[Array]:
        if self._hessian_fn is None:
            return None
        return self._hessian_fn(market_id)

    @contextmanager
    def perturbed(self, theta_index: int, delta: float) -> Iterator['UserSuppliedBackend']:
        if self.n_parameters == 0:
            raise NotImplementedError(
                "Expected UserSuppliedBackend to carry declared parameters "
                "(theta_names=[...]) before perturbed() can be called; the "
                "current instance was constructed without theta_names. "
                "Received 0 parameters. "
                "Fix: pass theta_names=[...] and perturb_callback=... at "
                "construction to enable finite-diff demand-adjustment corrections."
            )
        if self._perturb_callback is None:
            raise NotImplementedError(
                "Expected UserSuppliedBackend to carry a perturb_callback when "
                "parameters are declared. "
                "Received theta_names set but perturb_callback=None. "
                "Fix: pass perturb_callback=lambda idx, delta: <new backend> "
                "at construction to enable perturbed() for finite-diff demand "
                "adjustment."
            )
        if theta_index < 0 or theta_index >= self.n_parameters:
            raise IndexError(
                f"Expected theta_index in [0, {self.n_parameters}) for this "
                f"UserSuppliedBackend. "
                f"Received theta_index={theta_index}. "
                f"Fix: pass an integer in the valid range."
            )
        perturbed_backend = self._perturb_callback(theta_index, delta)
        try:
            yield perturbed_backend
        finally:
            # The callback returns a new backend; the original self is
            # unchanged, so there is no state to restore.
            pass
