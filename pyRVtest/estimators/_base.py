"""Shared 2SLS machinery for the in-package demand estimators."""

from __future__ import annotations

import logging
from typing import Any, List, Mapping, Optional, Tuple

import numpy as np
from pyblp.utilities.basics import Array

from ..exceptions import ValidationError
from ..formulation import Formulation


logger = logging.getLogger(__name__)


__all__ = [
    'compute_delta',
    'two_stage_least_squares',
    'build_design_from_formulation',
    'extract_array_column',
]


def extract_array_column(product_data: Mapping[str, Any], name: str) -> Array:
    """Pull a 1-D float array out of product_data.

    Accepts pandas DataFrames, structured numpy arrays, and dict-likes.
    Raises ``ValidationError`` (with a Fix line) if the column is absent.
    """
    try:
        col = product_data[name]
    except (KeyError, ValueError, IndexError) as exc:
        raise ValidationError(
            f"Expected product_data to contain a {name!r} column. "
            f"Received product_data without it. "
            f"Fix: add {name!r} to product_data, or pass the actual column "
            f"name via the corresponding ``*_column`` kwarg."
        ) from exc
    return np.asarray(col).flatten().astype(float)


def compute_delta(shares: Array, market_ids: Array) -> Array:
    """Berry (1994) mean-utility inversion: ``log(s_j) - log(s_{0,t})``.

    Computes the outside-good share per market as ``1 - sum_j s_{jt}``
    and returns ``log(s) - log(s_0)``.
    """
    if np.any(shares <= 0):
        raise ValidationError(
            "Expected all inside-good shares to be strictly positive for "
            "Berry inversion. "
            f"Received {(shares <= 0).sum()} non-positive share value(s). "
            "Fix: drop or aggregate products with zero shares before passing "
            "product_data to the estimator."
        )
    s0 = np.zeros(shares.shape[0])
    bad_markets: List[Any] = []
    for t in np.unique(market_ids):
        mask = market_ids == t
        inside_total = shares[mask].sum()
        if inside_total >= 1.0:
            bad_markets.append(t)
            continue
        s0[mask] = 1.0 - inside_total
    if bad_markets:
        raise ValidationError(
            f"Expected inside-good shares to sum to < 1 in every market "
            f"(so the outside good has positive share). "
            f"Received {len(bad_markets)} market(s) where inside shares "
            f"sum to >= 1: {bad_markets[:5]}{'...' if len(bad_markets) > 5 else ''}. "
            f"Fix: rescale shares or aggregate products so each market leaves "
            f"positive mass for the outside good."
        )
    return np.log(shares) - np.log(s0)


def build_design_from_formulation(
        formulation: Optional[Formulation], product_data: Mapping[str, Any], role: str,
) -> Tuple[Array, List[str]]:
    """Build a design matrix + column-name list from a Formulation.

    ``role`` is included in the error message so users see which kwarg
    failed validation. If ``formulation`` is ``None`` the function
    returns an ``(N, 0)`` matrix and an empty column list (used when the
    user has no exogenous regressors beyond the endogenous ones).
    """
    if formulation is None:
        return np.zeros((_extract_size(product_data), 0)), []
    if not isinstance(formulation, Formulation):
        raise ValidationError(
            f"Expected {role} to be a pyRVtest.Formulation instance. "
            f"Received {type(formulation).__name__}. "
            f"Fix: wrap the formula string in pyRVtest.Formulation(...)."
        )
    matrix, columns, _ = formulation._build_matrix(product_data)
    column_names = [str(c) for c in columns]
    return np.asarray(matrix, dtype=float), column_names


def _extract_size(product_data: Mapping[str, Any]) -> int:
    """Return N (number of rows) from any product_data shape."""
    if hasattr(product_data, '__len__'):
        try:
            return int(len(product_data))
        except TypeError:
            pass
    if hasattr(product_data, 'shape'):
        return int(product_data.shape[0])
    for v in (product_data.values() if hasattr(product_data, 'values') else []):
        return int(np.asarray(v).flatten().shape[0])
    raise ValidationError(
        "Expected product_data to support len() or expose a column from "
        "which row count can be inferred. "
        "Received an unrecognized container. "
        "Fix: pass product_data as a pandas DataFrame, structured numpy "
        "array, or dict-of-arrays."
    )


def two_stage_least_squares(
        delta: Array,
        D: Array,
        W: Array,
        weight: Optional[Array] = None,
) -> Tuple[Array, Array]:
    r"""Standard linear 2SLS for ``delta = D @ theta + xi``.

    Solves
    :math:`\hat\theta = (D' W \Omega W' D)^{-1} D' W \Omega W' \delta`
    where ``Omega`` is ``weight`` (default :math:`(W'W/N)^{-1}` — standard
    2SLS). Returns ``(theta_hat, omega)``. ``omega`` is the weight matrix
    actually used so callers can stash it in ``demand_params['W_demand']``
    and the analytical backend reuses the same one for the demand-
    adjustment correction (DMSS Appendix C eq. 77).
    """
    N = delta.shape[0]
    K_d = D.shape[1]
    K_w = W.shape[1]

    if K_w < K_d:
        raise ValidationError(
            f"Expected the instrument matrix W (concatenation of exogenous "
            f"regressors and excluded instruments) to have at least as many "
            f"columns as the design matrix D for identification. "
            f"Received W with {K_w} columns and D with {K_d} columns. "
            f"Fix: add additional excluded instruments to formulation_Z so the "
            f"order condition for 2SLS is satisfied."
        )

    if weight is None:
        WtW = W.T @ W
        rank_W = np.linalg.matrix_rank(W)
        if rank_W < K_w:
            raise ValidationError(
                "Expected the instrument matrix W (exogenous regressors + "
                "excluded instruments) to have full column rank for 2SLS. "
                f"Received W with shape {W.shape} but rank {rank_W}. "
                "Fix: drop collinear instruments from formulation_Z or "
                "collinear regressors from formulation_X."
            )
        try:
            omega = np.linalg.inv(WtW / N)
        except np.linalg.LinAlgError as exc:
            raise ValidationError(
                "Expected the instrument matrix W (exogenous regressors + "
                "excluded instruments) to have full column rank for 2SLS. "
                f"Received W'W/N that is singular (LinAlgError: {exc}). "
                "Fix: drop collinear instruments from formulation_Z or "
                "collinear regressors from formulation_X."
            ) from exc
    else:
        omega = np.asarray(weight, dtype=float)
        if omega.shape != (K_w, K_w):
            raise ValidationError(
                f"Expected W_demand to be a ({K_w}, {K_w}) ndarray matching the "
                f"width of the instrument matrix. "
                f"Received shape {omega.shape}. "
                f"Fix: omit W_demand to use the default (W'W/N)^-1, or pass a "
                f"correctly-sized weight matrix."
            )

    DtW = D.T @ W
    M = DtW @ omega @ DtW.T
    rhs = DtW @ omega @ (W.T @ delta)
    try:
        theta = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValidationError(
            "Expected the projected design matrix D' W Omega W' D to be "
            "invertible for 2SLS. "
            f"Received a singular matrix (LinAlgError: {exc}). "
            "Fix: check the rank condition (excluded instruments must "
            "explain endogenous regressors after partialling out X)."
        ) from exc

    return theta, omega
