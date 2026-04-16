"""Analytical demand Jacobian/Hessian for plain logit + shared module-level helpers.

v0.4 step 3c landed the initial version; this file was split post-step-3
for user-facing clarity (tracebacks + API docs). Contents now:

  - Module-level functions for logit and nested-logit math
    (`compute_analytical_jacobian`, `_logit_jacobian`,
    `_nested_logit_jacobian`, `_nested_logit_jacobian_derivative`,
    `compute_analytical_hessian`, `_infer_nesting_columns`) — moved
    verbatim from `pyRVtest/demand_jacobian.py`. That module is now a
    thin shim that re-exports these names for backward compatibility
    with existing callers (pyRVtest/problem.py, pyRVtest/markups.py,
    tests).

  - `LogitBackend` class: plain-logit (sigma=[]) DemandBackend wrapper.
    Parameter count = 1 (just alpha).

`NestedLogitBackend` moved to `pyRVtest/backends/nested_logit.py` and
subclasses `LogitBackend`. Users who want nested logit do:

    from pyRVtest.backends import NestedLogitBackend
    # or, for submodule access:
    from pyRVtest.backends.nested_logit import NestedLogitBackend

SupportsDemandAdjustment is NOT implemented here — that lands in step 4
when we unify the two demand-adjustment paths.

Based on Berry (1994) for plain logit (sigma=[]) and the general
L-level nested-logit formulas in AFSSZ equation (6).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, List, Mapping, Optional

import numpy as np
from pyblp.utilities.basics import Array, output


__all__ = [
    # Module-level functions (re-exported by pyRVtest.demand_jacobian shim)
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
# Module-level functions (moved verbatim from pyRVtest/demand_jacobian.py).
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
    """Compute the (J, J, J) demand Hessian d^2s_j/(dp_k dp_l) for a single market."""
    J = len(s)
    eps = 1e-7

    if len(sigma) == 0 or all(sig == 0 for sig in sigma):
        D_func = lambda s_: _logit_jacobian(alpha, s_)
    else:
        D_func = lambda s_: _nested_logit_jacobian(alpha, sigma, s_, nesting)

    D = D_func(s)

    dD_ds = np.zeros((J, J, J))
    for r in range(J):
        s_plus = s.copy()
        s_plus[r] += eps / 2
        s_minus = s.copy()
        s_minus[r] -= eps / 2
        dD_ds[:, :, r] = (D_func(s_plus) - D_func(s_minus)) / eps

    hessian = np.einsum('jkr,rl->jkl', dD_ds, D)

    return hessian


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
    output(
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
    """

    def __init__(self, alpha: float, product_data: Mapping) -> None:
        self._alpha = float(alpha)
        self._product_data = product_data
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


