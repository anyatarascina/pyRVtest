"""Demand-adjustment first-stage correction (DMSS 2024 Appendix C eq. 77).

Hosts the unified `compute_demand_adjustment` implementation (v0.4 step 4d)
that replaces `Problem._compute_analytical_demand_adjustment` and
`Problem._compute_demand_adjustment_gradient`. These are the paths that
had the three b3b08a3 bugs (sign error, missing concentration adjustment,
updated_W vs W).

v0.4 step 4b lands the `_residualize_on_xd` helper below — the 2SLS
profile-out formula used by every `SupportsDemandAdjustment` backend to
concentrate out the linear (non-price) coefficients `beta` from
`dxi/dtheta`. Previously duplicated in `PyBLPBackend.xi_gradient` and
`Problem._compute_analytical_demand_adjustment`; keeping a single copy
eliminates the divergence risk that caused one of the three b3b08a3 bugs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


__all__ = ['_residualize_on_xd']


_NDArray = NDArray[Any]


def _residualize_on_xd(
        dxi_dtheta: _NDArray, X_D: _NDArray, Z_D: _NDArray, W_D: _NDArray
) -> _NDArray:
    r"""2SLS-residualize ``dxi_dtheta`` on ``X_D`` using ``Z_D`` as instruments.

    Implements the profiled-moment gradient from DMSS Appendix C eq. (77):

    .. math::

        \frac{\partial \xi_{\text{profiled}}}{\partial \theta}
        = \frac{\partial \xi}{\partial \theta}
        - X_D \big( X_D' Z_D W_D Z_D' X_D \big)^{-1} X_D' Z_D W_D Z_D'
          \frac{\partial \xi}{\partial \theta}

    This concentrates out the linear demand-side coefficients (``beta``,
    including the intercept but excluding ``alpha`` which is already in
    ``dxi_dtheta``) at fixed nonlinear parameters ``(alpha, rho, sigma)``.

    If ``X_D`` has zero columns (no exogenous regressors to concentrate
    out), ``dxi_dtheta`` is returned unchanged.

    Parameters
    ----------
    dxi_dtheta
        Raw partial derivative of :math:`\xi` w.r.t. :math:`\theta`, shape
        ``(N, n_theta)``. For the PyBLP path this is the ``xi_by_theta``
        Jacobian with ``-prices`` appended if ``alpha`` is a linear
        parameter. For the analytical logit path this is
        ``(-prices, -log_within_nest_shares)`` stacked.
    X_D
        Non-price demand-side linear regressors (excluding ``prices``),
        shape ``(N, K_x)``. Demand fixed effects should already be
        absorbed.
    Z_D
        Demand instruments, shape ``(N, K_z)``.
    W_D
        GMM weight matrix used in estimation, shape ``(K_z, K_z)``.
        Per DMSS eq. (77) this is the estimation-time weight
        (``r.W``, not ``r.updated_W``).

    Returns
    -------
    ndarray
        Residualized gradient, shape ``(N, n_theta)``.

    Notes
    -----
    The formula is equivalent to ``dxi_dtheta - X_D @ inv(X_D'
    Z_D W_D Z_D' X_D) @ X_D' Z_D W_D Z_D' @ dxi_dtheta`` but uses a
    factored intermediate (``XtZW = X_D' Z_D W_D``) for clarity. Floating-
    point results differ by at most a few ULP from the one-line form due
    to matrix-product non-associativity; unit tests check this to
    ``atol=1e-11``.
    """
    if X_D.shape[1] == 0:
        return dxi_dtheta
    XtZW = X_D.T @ Z_D @ W_D
    M_xx = XtZW @ Z_D.T @ X_D
    projection_coeffs = np.linalg.inv(M_xx) @ (XtZW @ Z_D.T @ dxi_dtheta)
    return dxi_dtheta - X_D @ projection_coeffs
