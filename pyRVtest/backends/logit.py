"""Analytical demand Jacobian/Hessian for plain logit + shared module-level helpers.

v0.4 step 3c landed the initial version; this file was split post-step-3
for user-facing clarity (tracebacks + API docs). Contents now:

  - Module-level functions for logit and nested-logit math
    (`compute_analytical_jacobian`, `_logit_jacobian`,
    `_nested_logit_jacobian`, `_nested_logit_jacobian_derivative`,
    `compute_analytical_hessian`, `_infer_nesting_columns`). Originally
    lived in `pyRVtest/demand_jacobian.py`, moved here in v0.4 step 3c;
    the shim module was removed in v0.4 step 4g.

  - `LogitBackend` class: plain-logit (sigma=[]) DemandBackend wrapper
    that also implements `SupportsDemandAdjustment`. Parameter count
    = 1 (just alpha) in the plain-logit case; NestedLogitBackend
    overrides `self._sigma` to get 1 + L parameters.

`NestedLogitBackend` moved to `pyRVtest/backends/nested_logit.py` and
subclasses `LogitBackend`. Users who want nested logit do:

    from pyRVtest.backends import NestedLogitBackend
    # or, for submodule access:
    from pyRVtest.backends.nested_logit import NestedLogitBackend

v0.4 step 4c adds `SupportsDemandAdjustment` methods (`demand_moments`,
`xi_gradient`, `jacobian_gradient`) to `LogitBackend`. They key on
`self._sigma` — length-0 for plain logit, length-L for nested via
`NestedLogitBackend`. `NestedLogitBackend` inherits these methods
without overriding them (decision 2 for step 4, option ii).

Based on Berry (1994) for plain logit (sigma=[]) and the general
L-level nested-logit formulas in AFSSZ equation (6).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator, List, Mapping, Optional, Tuple

import numpy as np
from pyblp.utilities.basics import Array

from ..solve.demand_adjustment import _residualize_on_xd

# v0.4 step 18: per-module logger. Silence this subsystem specifically with
# ``logging.getLogger("pyRVtest.backends.logit").setLevel(logging.WARNING)``.
logger = logging.getLogger(__name__)


__all__ = [
    # Module-level analytical-jacobian functions.
    'compute_analytical_jacobian',
    'compute_analytical_hessian',
    '_logit_jacobian',
    '_nested_logit_jacobian',
    '_nested_logit_jacobian_derivative',
    '_infer_nesting_columns',
    # Plain-logit class (NestedLogitBackend now lives in nested_logit.py).
    'LogitBackend',
]


# ===========================================================================
# Module-level analytical-jacobian helpers (originally in
# pyRVtest/demand_jacobian.py before v0.4 step 3c moved them here).
# ===========================================================================


def compute_analytical_jacobian(
        alpha: float, sigma: List[float], product_data: Mapping,
        nesting_ids_columns: Optional[List[str]] = None
) -> Array:
    """Compute the (N, J_max) NaN-padded demand Jacobian for logit/nested logit.

    Parameters
    ----------
    alpha : float
        Price coefficient from demand estimation (typically negative).
    sigma : list of float
        Nesting parameters from finest to coarsest. Empty list or [0.0] for plain logit.
        [sigma_1] for 1-level nested logit. [sigma_1, sigma_2] for 2-level, etc.
    product_data : Mapping
        Product data with 'market_ids', 'shares', and nesting ID columns.
    nesting_ids_columns : list of str, optional
        Column names for nesting IDs, ordered finest to coarsest. If None and sigma
        is non-empty, inferred from columns named 'nesting_ids*' in product_data.

    Returns
    -------
    Array
        (N, J_max) NaN-padded stacked Jacobian matching PyBLP format.

    Examples
    --------
    >>> import numpy as np
    >>> from pyRVtest.backends.logit import compute_analytical_jacobian
    >>> product_data = {
    ...     'market_ids': np.array([0, 0, 1, 1]),
    ...     'shares': np.array([0.3, 0.3, 0.4, 0.4]),
    ... }
    >>> jac = compute_analytical_jacobian(alpha=-1.0, sigma=[], product_data=product_data)
    >>> jac.shape
    (4, 2)
    """
    # Validate alpha and sigma
    if not isinstance(alpha, (int, float)) or alpha >= 0:
        raise ValueError("alpha must be a negative scalar (price coefficient from demand estimation).")
    if not isinstance(sigma, (list, tuple)):
        raise TypeError("sigma must be a list of nesting parameters (empty list for plain logit).")
    for i, s_val in enumerate(sigma):
        if not isinstance(s_val, (int, float)):
            raise TypeError(f"sigma[{i}] must be a number, got {type(s_val)}.")
        if s_val < 0 or s_val >= 1:
            raise ValueError(
                f"sigma[{i}] = {s_val} is out of range. Each sigma must be in [0, 1). "
                f"Note: sigma = 0 corresponds to plain logit (Berry, 1994 convention)."
            )

    market_ids = np.asarray(product_data['market_ids']).flatten()
    shares = np.asarray(product_data['shares']).flatten()
    markets = np.unique(market_ids)
    N = len(market_ids)

    # Determine max products per market for NaN padding
    J_max = max(np.sum(market_ids == t) for t in markets)

    # Treat sigma values of exactly 0 as plain logit (no nesting effect)
    sigma = [s_val for s_val in sigma if s_val > 0]
    L = len(sigma)

    # Resolve nesting ID columns
    nesting_arrays = []
    if L > 0:
        if nesting_ids_columns is None:
            nesting_ids_columns = _infer_nesting_columns(product_data, L)
        if len(nesting_ids_columns) != L:
            raise ValueError(
                f"Number of nesting ID columns ({len(nesting_ids_columns)}) must match "
                f"number of non-zero sigma values ({L})."
            )
        for col in nesting_ids_columns:
            nesting_arrays.append(np.asarray(product_data[col]).flatten())

    # Build Jacobian market by market
    jacobian = np.full((N, J_max), np.nan)

    for t in markets:
        idx = np.where(market_ids == t)[0]
        J_t = len(idx)
        s_t = shares[idx]

        if L == 0:
            D_t = _logit_jacobian(alpha, s_t)
        else:
            nesting_t = [arr[idx] for arr in nesting_arrays]
            D_t = _nested_logit_jacobian(alpha, sigma, s_t, nesting_t)

        jacobian[idx[:, None], np.arange(J_t)[None, :]] = D_t

    return jacobian


def _logit_jacobian(alpha: float, s: Array) -> Array:
    """Compute J x J Jacobian for plain logit.

    D[j,k] = -alpha * s_j * s_k         for j != k
    D[j,j] = alpha * s_j * (1 - s_j)
    """
    J = len(s)
    s_col = s.reshape(J, 1)
    return alpha * (np.diag(s) - s_col @ s_col.T)


def _nested_logit_jacobian(alpha: float, sigma: List[float], s: Array,
                           nesting: List[Array]) -> Array:
    """Compute J x J Jacobian for general L-level nested logit.

    For products j, k with shares s_j, s_k, the derivative is:

        ds_k/dp_j = alpha * s_j * (
            -1/(1 - sigma_1) * I(j=k)
            + sum_{l=1}^{L} (sigma_l - sigma_{l-1}) / ((1-sigma_l)(1-sigma_{l-1}))
                * I(same nest at level l) * s_{k|nest_l}
            + s_k
        )

    where sigma_0 = 0 (convention), and s_{k|nest_l} = s_k / s_{nest_l} is the
    conditional share of k within its level-l nest.
    """
    J = len(s)
    L = len(sigma)

    sigma_ext = [0.0] + list(sigma)

    D = np.zeros((J, J))

    # Diagonal term: +1/(1 - sigma_1) for j == k
    np.fill_diagonal(D, 1.0 / (1.0 - sigma_ext[1]))

    # Outer share term: -s_k for all (j, k)
    D -= s[np.newaxis, :]

    # Level correction terms (subtracted)
    for l in range(L):
        sig_l = sigma_ext[l + 1]
        sig_prev = sigma_ext[l]

        nest_ids_l = nesting[l]
        unique_nests = np.unique(nest_ids_l)

        same_nest = nest_ids_l[:, None] == nest_ids_l[None, :]

        nest_shares = np.zeros(J)
        for g in unique_nests:
            mask = nest_ids_l == g
            nest_shares[mask] = s[mask].sum()

        cond_shares = s / nest_shares

        coef = (sig_l - sig_prev) / ((1.0 - sig_l) * (1.0 - sig_prev))

        D -= coef * same_nest * cond_shares[np.newaxis, :]

    D = alpha * s[:, np.newaxis] * D

    return D


def _nested_logit_jacobian_derivative(alpha: float, sigma: List[float], s: Array,
                                      nesting: List[Array], deriv_index: int) -> Array:
    """Compute d(D)/d(sigma_{deriv_index}) analytically for the L-level nested logit Jacobian."""
    J = len(s)
    L = len(sigma)
    m = deriv_index

    sigma_ext = [0.0] + list(sigma)
    sig_m = sigma_ext[m + 1]

    dM = np.zeros((J, J))

    if m == 0:
        np.fill_diagonal(dM, 1.0 / (1.0 - sig_m) ** 2)

    nest_ids_m = nesting[m]
    same_nest_m = nest_ids_m[:, None] == nest_ids_m[None, :]
    nest_shares_m = np.zeros(J)
    for g in np.unique(nest_ids_m):
        mask = nest_ids_m == g
        nest_shares_m[mask] = s[mask].sum()
    cond_shares_m = s / nest_shares_m
    d_coef_m = 1.0 / (1.0 - sig_m) ** 2
    dM -= d_coef_m * same_nest_m * cond_shares_m[np.newaxis, :]

    if m + 1 < L:
        nest_ids_mp1 = nesting[m + 1]
        same_nest_mp1 = nest_ids_mp1[:, None] == nest_ids_mp1[None, :]
        nest_shares_mp1 = np.zeros(J)
        for g in np.unique(nest_ids_mp1):
            mask = nest_ids_mp1 == g
            nest_shares_mp1[mask] = s[mask].sum()
        cond_shares_mp1 = s / nest_shares_mp1
        d_coef_mp1 = -1.0 / (1.0 - sig_m) ** 2
        dM -= d_coef_mp1 * same_nest_mp1 * cond_shares_mp1[np.newaxis, :]

    return alpha * s[:, np.newaxis] * dM


def compute_analytical_hessian(alpha: float, sigma: List[float], s: Array,
                               nesting: List[Array]) -> Array:
    """Compute the (J, J, J) demand Hessian d^2s_j/(dp_k dp_l) for a single market.

    Uses a closed-form ``dD/ds`` for plain logit (``sigma == []`` or all-zero
    sigma) and 1-level nested logit (single non-zero sigma). For 2+ nesting
    levels, falls back to a centered finite difference of the Jacobian w.r.t.
    shares (v0.4 step 7 scope limitation).

    The Hessian is assembled via the chain rule

        H[j, k, l] = sum_r (dD[j, k] / ds[r]) * D[r, l]

    where ``D = dS/dp`` is the demand Jacobian and ``dD/ds`` is the
    derivative of that Jacobian w.r.t. the share vector.

    Examples
    --------
    >>> import numpy as np
    >>> from pyRVtest.backends.logit import compute_analytical_hessian
    >>> s = np.array([0.3, 0.3])
    >>> h = compute_analytical_hessian(alpha=-1.0, sigma=[], s=s, nesting=[])
    >>> h.shape
    (2, 2, 2)
    """
    J = len(s)

    # Filter out zero sigmas: matches the convention in
    # compute_analytical_jacobian (zero sigma => plain logit at that level).
    effective_sigma = [sig for sig in sigma if sig > 0]
    L_eff = len(effective_sigma)

    if L_eff == 0:
        D = _logit_jacobian(alpha, s)
        dD_ds = _logit_dD_ds(alpha, s)
    elif L_eff == 1:
        # Pick the nesting array aligned with the single non-zero sigma.
        # Assumes the non-zero sigma's position in `sigma` maps to the same
        # position in `nesting` (the standard convention in this module).
        nest_idx = next(i for i, sig in enumerate(sigma) if sig > 0)
        nest_ids = nesting[nest_idx]
        rho = effective_sigma[0]
        D = _nested_logit_jacobian(alpha, effective_sigma, s, [nest_ids])
        dD_ds = _nested_logit_one_level_dD_ds(alpha, rho, s, nest_ids)
    else:
        # L >= 2: keep the centered finite-difference fallback.
        D_func = lambda s_: _nested_logit_jacobian(alpha, sigma, s_, nesting)
        D = D_func(s)
        eps = 1e-7
        dD_ds = np.zeros((J, J, J))
        for r in range(J):
            s_plus = s.copy()
            s_plus[r] += eps / 2
            s_minus = s.copy()
            s_minus[r] -= eps / 2
            dD_ds[:, :, r] = (D_func(s_plus) - D_func(s_minus)) / eps

    hessian = np.einsum('jkr,rl->jkl', dD_ds, D)

    return hessian


def _logit_dD_ds(alpha: float, s: Array) -> Array:
    """Closed-form dD/ds for plain logit.

    For ``D[j, k] = alpha * (s_j * delta_{jk} - s_j * s_k)``,

        dD[j, k] / ds[r] = alpha * (delta_{jr} * delta_{jk}
                                    - delta_{jr} * s_k
                                    - delta_{kr} * s_j)

    Returns a (J, J, J) tensor with axes (j, k, r).
    """
    J = len(s)
    eye = np.eye(J)
    out = np.zeros((J, J, J))

    # Term 1: alpha * delta_{jr} * delta_{jk}  -> nonzero only when j == k == r.
    diag_idx = np.arange(J)
    out[diag_idx, diag_idx, diag_idx] = alpha

    # Term 2: -alpha * delta_{jr} * s_k. Nonzero when j == r; value -alpha * s_k.
    # Shape broadcast: (J, J, J) with axes (j, k, r).
    out -= alpha * eye[:, None, :] * s[None, :, None]

    # Term 3: -alpha * delta_{kr} * s_j. Nonzero when k == r; value -alpha * s_j.
    out -= alpha * eye[None, :, :] * s[:, None, None]

    return out


def _nested_logit_one_level_dD_ds(alpha: float, rho: float, s: Array,
                                  nest_ids: Array) -> Array:
    """Closed-form dD/ds for 1-level nested logit.

    The Jacobian is ``D[j, k] = alpha * s_j * M[j, k]`` with

        M[j, k] = (1/(1-rho)) * delta_{jk}  -  s_k
                  -  (rho/(1-rho)) * I_same(j,k) * (s_k / S_g(k))

    where ``S_g(k)`` is the sum of shares in ``k``'s nest. Differentiating
    w.r.t. s_r and using the product rule on ``s_j * M[j, k]`` gives

        dD[j,k]/ds[r] = alpha * ( delta_{jr} * M[j, k]
                                 + s_j * dM[j, k]/ds[r] )

    with

        dM[j, k]/ds[r] = - delta_{kr}
                        - c * I_same(j,k) * [ delta_{kr} / S_g(k)
                                             - s_k * I_same(r,k) / S_g(k)^2 ]

    where ``c = rho/(1-rho)`` and ``I_same(a, b)`` is 1 if ``a`` and ``b``
    share a nest, else 0. The ``I_same(r, k)`` factor makes the last piece
    active exactly when both ``j`` and ``r`` share ``k``'s nest.

    Returns a (J, J, J) tensor with axes (j, k, r).
    """
    J = len(s)
    eye = np.eye(J)

    same_nest = (nest_ids[:, None] == nest_ids[None, :])  # (J, J)

    # Nest-sum per product: S_g(k) = sum of shares in k's nest.
    nest_sum = np.zeros(J)
    for g in np.unique(nest_ids):
        mask = nest_ids == g
        nest_sum[mask] = s[mask].sum()

    c = rho / (1.0 - rho)
    inv_1m = 1.0 / (1.0 - rho)

    # M[j, k] — same expression used inside _nested_logit_jacobian,
    # reconstructed here for clarity. D = alpha * s[:, None] * M.
    cond_share = s / nest_sum  # s_{k|g(k)}
    M = np.zeros((J, J))
    np.fill_diagonal(M, inv_1m)
    M = M - s[None, :] - c * same_nest * cond_share[None, :]

    # ---- Build dM/ds, shape (J, J, J) with axes (j, k, r). ----
    dM = np.zeros((J, J, J))

    # Piece A: -delta_{kr} (independent of j). Nonzero when k == r.
    dM -= eye[None, :, :]  # (1, J, J) broadcast over j

    # Piece B: -c * I_same(j,k) * delta_{kr} / S_g(k).
    # Nonzero when k == r; value depends on (j, k) via same_nest and S_g(k).
    #   shape of (same_nest / nest_sum): (J, J) indexed as (j, k)
    B_jk = same_nest / nest_sum[None, :]  # (J, J)
    dM -= c * B_jk[:, :, None] * eye[None, :, :]

    # Piece C: + c * I_same(j,k) * s_k * I_same(r,k) / S_g(k)^2.
    # Nonzero when j, r both share k's nest.
    #   same_nest[:, k] gives the k'th column — who shares k's nest.
    #   We need same_nest_{j,k} (axis jk) AND same_nest_{r,k} (axis rk).
    # same_nest_rk: for each (r, k), 1 if r in nest(k). Same matrix as same_nest,
    # indexed transposed: same_nest_rk[r, k] = same_nest[r, k].
    s_over_S2 = s / (nest_sum ** 2)  # (J,), indexed by k
    dM += c * (same_nest[:, :, None] * same_nest[None, :, :].swapaxes(1, 2)
               * s_over_S2[None, :, None])

    # ---- Assemble dD/ds = alpha * (delta_{jr} * M[j, k] + s_j * dM[j, k]/ds[r]). ----
    dD_ds = np.zeros((J, J, J))

    # Term 1: alpha * delta_{jr} * M[j, k]. Nonzero when j == r.
    # Put M[j, k] on axes (j, k) and delta_{jr} on (j, r):
    dD_ds += alpha * eye[:, None, :] * M[:, :, None]

    # Term 2: alpha * s_j * dM/ds.
    dD_ds += alpha * s[:, None, None] * dM

    return dD_ds


def _infer_nesting_columns(product_data: Mapping, L: int) -> List[str]:
    """Infer nesting ID columns from product_data and validate hierarchy."""
    if hasattr(product_data, 'dtype') and product_data.dtype.names:
        all_cols = list(product_data.dtype.names)
    elif hasattr(product_data, 'columns'):
        all_cols = list(product_data.columns)
    else:
        all_cols = list(product_data.keys()) if hasattr(product_data, 'keys') else []

    if L == 1:
        if 'nesting_ids' in all_cols:
            return ['nesting_ids']
        candidates = [c for c in all_cols if c.startswith('nesting_ids')]
        if len(candidates) == 1:
            return candidates
        raise ValueError(
            f"Cannot find nesting ID column. Expected 'nesting_ids' in product_data. "
            f"Pass nesting_ids_columns explicitly in demand_params."
        )

    candidates = [c for c in all_cols if c.startswith('nesting_ids')]
    if len(candidates) < L:
        raise ValueError(
            f"Found {len(candidates)} nesting ID columns but sigma has {L} levels. "
            f"Pass nesting_ids_columns explicitly in demand_params."
        )

    data_arrays = [(col, np.asarray(product_data[col]).flatten()) for col in candidates[:L + 2]]
    data_arrays.sort(key=lambda x: len(np.unique(x[1])), reverse=True)
    ordered = [col for col, _ in data_arrays[:L]]

    for i in range(len(ordered) - 1):
        finer = np.asarray(product_data[ordered[i]]).flatten()
        coarser = np.asarray(product_data[ordered[i + 1]]).flatten()
        for val in np.unique(finer):
            coarse_vals = np.unique(coarser[finer == val])
            if len(coarse_vals) > 1:
                raise ValueError(
                    f"Nesting columns are not hierarchical: '{ordered[i]}' value '{val}' "
                    f"maps to multiple '{ordered[i + 1]}' values: {coarse_vals}. "
                    f"Pass nesting_ids_columns explicitly in demand_params."
                )

    n_groups = [len(np.unique(np.asarray(product_data[col]).flatten())) for col in ordered]
    logger.info(
        f"Inferred nesting order (finest to coarsest): {ordered}\n"
        f"  Groups per level: {dict(zip(ordered, n_groups))}\n"
        f"  To specify manually, pass nesting_ids_columns in demand_params."
    )

    return ordered


# ===========================================================================
# Class wrappers satisfying the DemandBackend protocol.
# ===========================================================================


class LogitBackend:
    """Analytical DemandBackend for plain logit (no nesting).

    Wraps `compute_analytical_jacobian` and `compute_analytical_hessian`
    with the DemandBackend class API. Only parameter is alpha (the price
    coefficient). `perturbed(0, delta)` shifts alpha by delta.

    v0.4 step 4c: if the demand-adjustment state (``beta``, ``x_columns``,
    ``demand_instrument_columns``, optionally ``W_demand``) is provided
    at construction, the backend implements ``SupportsDemandAdjustment``.
    Without it, ``demand_moments`` / ``xi_gradient`` / ``jacobian_gradient``
    raise a clear error listing the missing fields. ``NestedLogitBackend``
    inherits these methods by overriding ``self._sigma`` and
    ``self._nesting_ids_columns`` in its own ``__init__``.

    Examples
    --------
    >>> import numpy as np
    >>> from pyRVtest.backends import LogitBackend
    >>> product_data = {
    ...     'market_ids': np.array([0, 0, 1, 1]),
    ...     'shares': np.array([0.3, 0.3, 0.4, 0.4]),
    ...     'prices': np.array([1.0, 2.0, 1.5, 2.5]),
    ... }
    >>> backend = LogitBackend(alpha=-1.0, product_data=product_data)
    >>> backend.n_parameters
    1
    >>> backend.theta_names
    ['alpha']
    >>> J_t = backend.compute_jacobian(market_id=0)
    >>> J_t.shape
    (2, 2)
    """

    def __init__(
            self,
            alpha: float,
            product_data: Mapping,
            beta: Optional[Array] = None,
            x_columns: Optional[List[str]] = None,
            demand_instrument_columns: Optional[List[str]] = None,
            W_demand: Optional[Array] = None,
    ) -> None:
        self._alpha = float(alpha)
        self._product_data = product_data
        # Plain logit by default; NestedLogitBackend overrides.
        self._sigma: List[float] = []
        self._nesting_ids_columns: Optional[List[str]] = None
        # Demand-adjustment state. All optional; SupportsDemandAdjustment
        # methods raise if accessed before these are set.
        self._beta: Optional[Array] = (
            np.asarray(beta).flatten() if beta is not None else None
        )
        self._x_columns: Optional[List[str]] = list(x_columns) if x_columns else None
        self._demand_instrument_columns: Optional[List[str]] = (
            list(demand_instrument_columns) if demand_instrument_columns else None
        )
        self._W_demand: Optional[Array] = (
            np.asarray(W_demand) if W_demand is not None else None
        )
        self._jacobian_cache: Optional[Array] = None

    @property
    def n_parameters(self) -> int:
        return 1

    @property
    def theta_names(self) -> List[str]:
        return ['alpha']

    def compute_jacobian(self, market_id: Any = None) -> Array:
        if self._jacobian_cache is None:
            self._jacobian_cache = compute_analytical_jacobian(
                self._alpha, [], self._product_data, nesting_ids_columns=None
            )
        full = self._jacobian_cache
        if market_id is None:
            return full
        mids = np.asarray(self._product_data['market_ids']).flatten()
        idx = np.where(mids == market_id)[0]
        block = full[idx]
        block = block[:, ~np.isnan(block).all(axis=0)]
        return block

    def compute_hessian(self, market_id: Any) -> Array:
        mids = np.asarray(self._product_data['market_ids']).flatten()
        idx = np.where(mids == market_id)[0]
        shares = np.asarray(self._product_data['shares']).flatten()[idx]
        return compute_analytical_hessian(self._alpha, [], shares, [])

    @contextmanager
    def perturbed(self, theta_index: int, delta: float) -> Iterator['LogitBackend']:
        if theta_index != 0:
            raise IndexError(
                f"LogitBackend has 1 parameter (alpha); theta_index must be 0, got {theta_index}."
            )
        saved_alpha = self._alpha
        saved_cache = self._jacobian_cache
        try:
            self._alpha = saved_alpha + delta
            self._jacobian_cache = None
            yield self
        finally:
            self._alpha = saved_alpha
            self._jacobian_cache = saved_cache

    # -----------------------------------------------------------------
    # SupportsDemandAdjustment (v0.4 step 4c)
    # -----------------------------------------------------------------

    def demand_moments(self) -> Tuple[Array, Array, Array]:
        """Return (xi, Z_D, W_D) for the DMSS eq. (77) first-stage correction.

        xi is the demand residual from Berry (1994) inversion:
        ``log(s_j) - log(s_0) - X_D beta - alpha p - sum_l sigma_l log(s_{j|g_l})``.
        Z_D and W_D come from the stored demand-adjustment state.
        """
        self._require_demand_adjustment_state()
        shares = np.asarray(self._product_data['shares']).flatten()
        prices = np.asarray(self._product_data['prices']).flatten()
        market_ids = np.asarray(self._product_data['market_ids']).flatten()
        N = shares.shape[0]
        s0 = self._compute_s0(shares, market_ids)
        X_D = self._build_X_D()
        Z_D = self._build_Z_D()
        xi = np.log(shares) - np.log(s0) - X_D @ self._beta - self._alpha * prices
        for sig, log_wn in zip(self._sigma, self._log_within_nest_shares(shares, market_ids)):
            xi = xi - sig * log_wn
        W_D = self._compute_W_D(Z_D, N)
        return xi, Z_D, W_D

    def xi_gradient(self) -> Array:
        """Return ∂xi/∂theta profiled on X_D (2SLS residualize on exogenous regressors).

        Shape ``(N, 1 + L)`` where ``L == len(self._sigma)``. For plain logit (L=0)
        this is ``(-prices,)`` residualized. For nested logit the additional columns
        are ``-log(s_{j|g_l})`` for each nesting level l.
        """
        self._require_demand_adjustment_state()
        shares = np.asarray(self._product_data['shares']).flatten()
        prices = np.asarray(self._product_data['prices']).flatten()
        market_ids = np.asarray(self._product_data['market_ids']).flatten()
        N = prices.shape[0]
        L = len(self._sigma)
        dxi_dtheta = np.zeros((N, 1 + L))
        dxi_dtheta[:, 0] = -prices
        for level, log_wn in enumerate(self._log_within_nest_shares(shares, market_ids)):
            dxi_dtheta[:, 1 + level] = -log_wn
        X_D = self._build_X_D()
        Z_D = self._build_Z_D()
        W_D = self._compute_W_D(Z_D, N)
        return _residualize_on_xd(dxi_dtheta, X_D, Z_D, W_D)

    def jacobian_gradient(self, market_id: Any) -> Array:
        """Return ∂D/∂theta for one market, shape ``(J_t, J_t, 1 + L)``.

        The alpha column uses ``D / alpha`` (D is linear in alpha).
        Sigma columns use the analytical derivative
        ``_nested_logit_jacobian_derivative(alpha, sigma, s_t, nesting_t, l)``.
        """
        mids = np.asarray(self._product_data['market_ids']).flatten()
        idx = np.where(mids == market_id)[0]
        shares = np.asarray(self._product_data['shares']).flatten()
        s_t = shares[idx]
        J_t = idx.shape[0]
        L = len(self._sigma)
        grad = np.zeros((J_t, J_t, 1 + L))
        # d(D)/d(alpha) = D/alpha (D = alpha * f(s, sigma))
        D_t = self.compute_jacobian(market_id=market_id)
        grad[:, :, 0] = D_t / self._alpha
        if L > 0:
            cols = self._nesting_ids_columns
            if cols is None:
                cols = _infer_nesting_columns(self._product_data, L)
            nesting_t: List[Array] = []
            for col in cols:
                arr = np.asarray(self._product_data[col]).flatten()
                nesting_t.append(arr[idx])
            for level in range(L):
                grad[:, :, 1 + level] = _nested_logit_jacobian_derivative(
                    self._alpha, self._sigma, s_t, nesting_t, level
                )
        return grad

    # -----------------------------------------------------------------
    # Internal helpers for SupportsDemandAdjustment
    # -----------------------------------------------------------------

    def _require_demand_adjustment_state(self) -> None:
        """Raise a clear error if the backend was constructed without the fields
        that SupportsDemandAdjustment needs.
        """
        missing: List[str] = []
        if self._beta is None:
            missing.append('beta')
        if self._x_columns is None:
            missing.append('x_columns')
        if self._demand_instrument_columns is None:
            missing.append('demand_instrument_columns')
        if missing:
            raise ValueError(
                f"{type(self).__name__} was constructed without the demand-adjustment "
                f"state: {', '.join(missing)}. Pass these kwargs at construction to "
                f"enable demand_moments / xi_gradient / jacobian_gradient."
            )

    def _build_X_D(self) -> Array:
        assert self._x_columns is not None  # narrowed by _require_demand_adjustment_state
        return np.column_stack([
            np.asarray(self._product_data[col]).flatten() for col in self._x_columns
        ])

    def _build_Z_D(self) -> Array:
        assert self._demand_instrument_columns is not None
        return np.column_stack([
            np.asarray(self._product_data[col]).flatten()
            for col in self._demand_instrument_columns
        ])

    def _compute_W_D(self, Z_D: Array, N: int) -> Array:
        if self._W_demand is not None:
            return self._W_demand
        return np.linalg.inv((1 / N) * Z_D.T @ Z_D)

    @staticmethod
    def _compute_s0(shares: Array, market_ids: Array) -> Array:
        """Outside-good share per observation, computed per market."""
        s0 = np.zeros(shares.shape[0])
        for t in np.unique(market_ids):
            mask = market_ids == t
            s0[mask] = 1.0 - shares[mask].sum()
        return s0

    def _log_within_nest_shares(
            self, shares: Array, market_ids: Array
    ) -> List[Array]:
        """``log(s_{j|g_l})`` vectors, one per nesting level.

        Returns a list of length ``L = len(self._sigma)``. For plain logit
        (L=0) returns ``[]``. Matches the inline loop in
        ``Problem._compute_analytical_demand_adjustment``.
        """
        L = len(self._sigma)
        if L == 0:
            return []
        cols = self._nesting_ids_columns
        if cols is None:
            cols = _infer_nesting_columns(self._product_data, L)
        result: List[Array] = []
        N = shares.shape[0]
        for col in cols:
            nest_ids = np.asarray(self._product_data[col]).flatten()
            within = np.zeros(N)
            for t in np.unique(market_ids):
                idx = np.where(market_ids == t)[0]
                nest_t = nest_ids[idx]
                s_t = shares[idx]
                for g in np.unique(nest_t):
                    mask = nest_t == g
                    nest_sum = s_t[mask].sum()
                    within[idx[mask]] = s_t[mask] / nest_sum
            result.append(np.log(within))
        return result


