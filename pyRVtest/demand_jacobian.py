"""Analytical demand Jacobian for logit and multi-level nested logit.

Computes the J_t x J_t matrix of demand derivatives ds/dp per market without
PyBLP, using the closed-form expressions from Berry (1994). Supports plain logit
(sigma=[]) and general L-level nested logit (sigma=[sigma_1, ..., sigma_L] from
finest to coarsest nesting level).
"""

import numpy as np
from pyblp.utilities.basics import Array, output
from typing import List, Mapping, Optional, Sequence


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

    # Convention: sigma_0 = 0 (outside the nesting structure = plain logit)
    sigma_ext = [0.0] + list(sigma)  # [sigma_0, sigma_1, ..., sigma_L]

    # Build Jacobian following AFSSZ equation (6) generalized to L levels.
    #
    # For the 1-level case, AFSSZ gives:
    #   ds_j/dp_j = alpha * s_j * (1/(1-sig) - sig/(1-sig) * s_{j|g} - s_j)
    #   ds_k/dp_j = -alpha * s_j * (sig/(1-sig) * s_{k|g} + s_k)  [same nest]
    #   ds_k/dp_j = -alpha * s_j * s_k                              [diff nests]
    #
    # Rewriting in a unified form for all (j,k):
    #   ds_k/dp_j = alpha * s_j * (
    #       1/(1-sigma_1) * I(j==k)
    #       - sum_l (sigma_l - sigma_{l-1}) / ((1-sigma_l)(1-sigma_{l-1})) * I(same nest l) * s_{k|l}
    #       - s_k
    #   )
    # where sigma_0 = 0.

    D = np.zeros((J, J))

    # Diagonal term: +1/(1 - sigma_1) for j == k
    np.fill_diagonal(D, 1.0 / (1.0 - sigma_ext[1]))

    # Outer share term: -s_k for all (j, k)
    D -= s[np.newaxis, :]

    # Level correction terms (subtracted)
    for l in range(L):
        sig_l = sigma_ext[l + 1]  # sigma at this level
        sig_prev = sigma_ext[l]   # sigma at previous (finer) level

        nest_ids_l = nesting[l]
        unique_nests = np.unique(nest_ids_l)

        # same_nest[j, k] = True if j and k are in the same nest at level l
        same_nest = nest_ids_l[:, None] == nest_ids_l[None, :]

        # Nest shares: s_{g_l} = sum of s_j for j in nest g_l
        nest_shares = np.zeros(J)
        for g in unique_nests:
            mask = nest_ids_l == g
            nest_shares[mask] = s[mask].sum()

        # Conditional shares: s_{k|g_l} = s_k / s_{g_l}
        cond_shares = s / nest_shares

        # Coefficient for this level
        coef = (sig_l - sig_prev) / ((1.0 - sig_l) * (1.0 - sig_prev))

        # Subtract contribution: coef * I(same nest at l) * s_{k|nest_l}
        D -= coef * same_nest * cond_shares[np.newaxis, :]

    # Multiply by alpha * s_j (row-wise)
    D = alpha * s[:, np.newaxis] * D

    return D


def _nested_logit_jacobian_derivative(alpha: float, sigma: List[float], s: Array,
                                      nesting: List[Array], deriv_index: int) -> Array:
    """Compute d(D)/d(sigma_{deriv_index}) analytically for the L-level nested logit Jacobian.

    The Jacobian inner matrix (before alpha * s_j multiplication) is:
        M[j,k] = 1/(1-sigma_1) * I(j==k)
                - sum_l coef_l * same_nest_l[j,k] * cond_shares_l[k]
                - s_k
    where coef_l = (sigma_l - sigma_{l-1}) / ((1-sigma_l)(1-sigma_{l-1})).

    Differentiating w.r.t. sigma_m (0-indexed as deriv_index):
    - Diagonal: d[1/(1-sigma_1)]/d(sigma_m) = 1/(1-sigma_1)^2 if m==0, else 0
    - Level l=m: d(coef_m)/d(sigma_m) = 1/(1-sigma_m)^2
    - Level l=m+1: d(coef_{m+1})/d(sigma_m) = -1/(1-sigma_m)^2
    - All other levels: 0
    - Outer share term: 0

    Parameters
    ----------
    deriv_index : int
        0-indexed index into the sigma list. d(D)/d(sigma_{deriv_index+1}) in
        the 1-indexed convention of the paper.
    """
    J = len(s)
    L = len(sigma)
    m = deriv_index  # 0-indexed level we're differentiating w.r.t.

    sigma_ext = [0.0] + list(sigma)  # [sigma_0=0, sigma_1, ..., sigma_L]
    sig_m = sigma_ext[m + 1]

    dM = np.zeros((J, J))

    # Diagonal term: d[1/(1-sigma_1)]/d(sigma_m)
    if m == 0:  # differentiating w.r.t. sigma_1 (finest level)
        np.fill_diagonal(dM, 1.0 / (1.0 - sig_m) ** 2)

    # Level m contribution: d(coef_m)/d(sigma_m) = 1/(1-sigma_m)^2
    nest_ids_m = nesting[m]
    same_nest_m = nest_ids_m[:, None] == nest_ids_m[None, :]
    nest_shares_m = np.zeros(J)
    for g in np.unique(nest_ids_m):
        mask = nest_ids_m == g
        nest_shares_m[mask] = s[mask].sum()
    cond_shares_m = s / nest_shares_m
    d_coef_m = 1.0 / (1.0 - sig_m) ** 2
    dM -= d_coef_m * same_nest_m * cond_shares_m[np.newaxis, :]

    # Level m+1 contribution (if it exists): d(coef_{m+1})/d(sigma_m) = -1/(1-sigma_m)^2
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

    # Multiply by alpha * s_j (same outer factor as the Jacobian itself)
    return alpha * s[:, np.newaxis] * dM


def _infer_nesting_columns(product_data: Mapping, L: int) -> List[str]:
    """Infer nesting ID columns from product_data and validate hierarchy.

    Looks for 'nesting_ids' (L=1) or 'nesting_ids_1', 'nesting_ids_2', etc.
    When multiple columns are found, infers order by counting unique groups
    (more groups = finer level) and validates that finer levels are refinements
    of coarser levels.
    """
    # Get available column names
    if hasattr(product_data, 'dtype') and product_data.dtype.names:
        all_cols = list(product_data.dtype.names)
    elif hasattr(product_data, 'columns'):
        all_cols = list(product_data.columns)
    else:
        all_cols = list(product_data.keys()) if hasattr(product_data, 'keys') else []

    # Find nesting columns
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

    # Multi-level: look for nesting_ids_1, nesting_ids_2, ... or nesting_ids + others
    candidates = [c for c in all_cols if c.startswith('nesting_ids')]
    if len(candidates) < L:
        raise ValueError(
            f"Found {len(candidates)} nesting ID columns but sigma has {L} levels. "
            f"Pass nesting_ids_columns explicitly in demand_params."
        )

    # Sort by number of unique groups (more groups = finer level)
    data_arrays = [(col, np.asarray(product_data[col]).flatten()) for col in candidates[:L + 2]]
    data_arrays.sort(key=lambda x: len(np.unique(x[1])), reverse=True)

    # Take the top L
    ordered = [col for col, _ in data_arrays[:L]]

    # Validate hierarchy: finer must be a refinement of coarser
    for i in range(len(ordered) - 1):
        finer = np.asarray(product_data[ordered[i]]).flatten()
        coarser = np.asarray(product_data[ordered[i + 1]]).flatten()
        # Every unique value of finer should map to exactly one value of coarser
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
