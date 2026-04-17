"""Protocols for demand backends.

v0.4 step 3a. Defines two Protocol classes (PEP 544 structural subtyping):

  - `DemandBackend`: the core interface. Every backend implements this.
    Provides `compute_jacobian`, `compute_hessian`, `perturbed` context
    manager, and parameter enumeration (`n_parameters`, `theta_names`).

  - `SupportsDemandAdjustment`: the optional mixin. Backends that support
    the first-stage-correction adjustment (DMSS 2024 Appendix C eq. 77)
    additionally implement `demand_moments`, `xi_gradient`, and
    `jacobian_gradient`. The testing engine uses
    `isinstance(backend, SupportsDemandAdjustment)` to decide whether
    to request these.

The split (plan §2 item 2) lets `UserSuppliedBackend` opt out of the
demand-adjustment machinery cleanly — users who supply a bare Jacobian
don't need to also supply xi, Z_D, W_D, or the BLP contraction.

Subsequent sub-steps (3b/3c/3d/3e) populate concrete classes.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Hashable, Iterator, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
from numpy.typing import NDArray


__all__ = ['DemandBackend', 'SupportsDemandAdjustment']

# Type alias: pyRVtest arrays are float64 for numeric quantities but object
# dtype for id columns. NDArray[Any] = np.ndarray[Any, np.dtype[Any]], which
# keeps the dtype axis loose. Aliased locally so the Protocol signatures are
# concise without `from numpy.typing import NDArray` at every call site.
_NDArray = NDArray[Any]


@runtime_checkable
class DemandBackend(Protocol):
    """Core protocol for demand primitives consumed by pyRVtest.

    A backend provides the demand Jacobian (∂s/∂p), optionally the
    Hessian (∂²s/∂p²) for vertical-integration passthrough, and the
    machinery to perturb parameters and re-evaluate (used by the
    finite-difference demand adjustment; analytical backends may
    override with closed-form derivatives).

    Attributes
    ----------
    n_parameters
        Total count of free demand parameters. For logit with just
        alpha this is 1; for nested logit with k nests it's
        1 + k; for BLP it's len(sigma.flat) + len(pi.flat) + ...
    theta_names
        Human-readable names for each parameter in the flat vector.
        Used in error messages and for symbolic debugging.

    Examples
    --------
    ``DemandBackend`` is a ``runtime_checkable`` protocol, so concrete
    backends can be tested with ``isinstance``:

    >>> import numpy as np
    >>> from pyRVtest.backends import DemandBackend, LogitBackend
    >>> product_data = {
    ...     'market_ids': np.array([0, 0, 1, 1]),
    ...     'shares': np.array([0.3, 0.3, 0.4, 0.4]),
    ...     'prices': np.array([1.0, 2.0, 1.5, 2.5]),
    ... }
    >>> backend = LogitBackend(alpha=-1.0, product_data=product_data)
    >>> isinstance(backend, DemandBackend)
    True
    """

    # --- identification ---

    @property
    def n_parameters(self) -> int:
        ...

    @property
    def theta_names(self) -> List[str]:
        ...

    # --- primitives (always supported) ---

    def compute_jacobian(self, market_id: Optional[Hashable] = None) -> _NDArray:
        """Return ∂s/∂p.

        If market_id is None, returns the stacked Jacobian across all
        markets with NaN-padding for jagged market sizes (the current
        `_compute_markups` format). If market_id is given, returns the
        dense (J_t, J_t) block for that market.
        """
        ...

    def compute_hessian(self, market_id: Hashable) -> Optional[_NDArray]:
        """Return ∂²s/∂p² for one market, shape (J_t, J_t, J_t).

        Returns None if the backend does not support Hessians. Used by
        the vertical-integration passthrough-matrix construction.
        Analytical backends return a closed-form; the PyBLP backend
        delegates to `compute_demand_hessians`.
        """
        ...

    # --- context manager for perturbation ---

    @contextmanager
    def perturbed(self, theta_index: int, delta: float) -> Iterator['DemandBackend']:
        """Yield self with theta[theta_index] += delta. Must restore on exit.

        Cache invariation and state restoration are the backend's
        responsibility. Used by the finite-difference fallback in
        `compute_demand_adjustment` (step 4).
        """
        ...


@runtime_checkable
class SupportsDemandAdjustment(Protocol):
    """Optional mixin: backends that support the first-stage correction.

    pyRVtest's demand adjustment (DMSS 2024 Appendix C eq. 77) requires
    access to the GMM residuals (xi), instruments (Z_D), weight matrix
    (W_D), and the gradients d(xi)/d(theta) and d(Jacobian)/d(theta).
    Backends that can produce these advertise themselves via this
    protocol; the testing engine checks
    `isinstance(backend, SupportsDemandAdjustment)` before trying to
    pull these values.

    `UserSuppliedBackend` deliberately does NOT implement this mixin.
    Users supplying a bare Jacobian can still run the RV test without
    demand adjustment; if they want adjustment, they must supply the
    full backend-adjustment API separately.

    Examples
    --------
    Backends announce their support via ``isinstance`` against the
    protocol. ``UserSuppliedBackend`` deliberately opts out:

    >>> import numpy as np
    >>> from pyRVtest.backends import SupportsDemandAdjustment, UserSuppliedBackend
    >>> backend = UserSuppliedBackend(
    ...     jacobian=np.array([[-1.0]]),
    ...     market_ids=np.array([0]),
    ... )
    >>> isinstance(backend, SupportsDemandAdjustment)
    False
    """

    def demand_moments(self) -> Tuple[_NDArray, _NDArray, _NDArray]:
        """Return (xi, Z_D, W_D) for the psi first-stage correction.

        xi: structural demand residuals, shape (N, 1).
        Z_D: demand instruments, shape (N, d_z_D).
        W_D: GMM weight matrix actually used in estimation
             (NOT the "updated_W" next-step weight; see b3b08a3 for
             the bug that arose from using updated_W instead).
        """
        ...

    def xi_gradient(self) -> _NDArray:
        """Return d(xi)/d(theta), shape (N, n_parameters).

        In the DMSS framework, xi is the demand-side residual whose
        gradient in theta enters Lambda. For BLP-like systems this is
        a PyBLP-internal quantity; for closed-form logit it's derived
        analytically.
        """
        ...

    def jacobian_gradient(self, market_id: Hashable) -> _NDArray:
        """Return d(D_jk)/d(theta_i) for one market, shape (J_t, J_t, n_parameters).

        The market-level Jacobian's derivative w.r.t. demand parameters.
        Used to compute G_m = -(1/N) z' ∇_theta ĥ(θ) via chain rule.
        """
        ...
