"""Labor-side instrument constructors.

v0.4 step 13. Mechanical, vectorized helpers for labor-market testing
specifications. Each helper accepts ``product_data`` (``pandas.DataFrame``,
``dict``, or structured record array -- anything supporting ``d[col]``
column access) and returns a numpy array aligned row-for-row with the input.

Functions:

  - :func:`hausman` -- leave-one-market-out mean (optionally restricted to
    same period).
  - :func:`bartik` -- simple shift-share ``weight_j * leave-one-out-mean(
    shock)``.
  - :func:`concentration_hhi` -- Herfindahl-Hirschman index broadcast to
    every product in a market.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


__all__ = ['hausman', 'bartik', 'concentration_hhi']


def _column(product_data: Any, name: str) -> np.ndarray:
    """Return ``product_data[name]`` as a 1-D ``ndarray``."""
    col = product_data[name]
    arr = np.asarray(col)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def hausman(
        product_data: Any,
        column: str,
        market_id_column: str = 'market_ids',
        period_column: Optional[str] = None) -> np.ndarray:
    r"""Hausman leave-one-market-out mean of ``column``.

    For each product :math:`j` in market :math:`t`, return the mean of
    ``column`` across all products in markets :math:`s \neq t`. If
    ``period_column`` is supplied, restrict the average to products in the
    same period as :math:`j` (but still in a different market).

    A common application: the wage in other markets used as an instrument
    for the local wage.

    Parameters
    ----------
    product_data : structured array-like
        Row per product.
    column : str
        Column to average.
    market_id_column : str, optional
        Market ID column name. Defaults to ``'market_ids'``.
    period_column : str, optional
        If supplied, the leave-one-out mean is restricted to the same period
        as each row. Defaults to ``None`` (no period restriction).

    Returns
    -------
    ndarray, shape ``(N,)``

    Notes
    -----
    If a product is the only one in its (market, period) bucket
    universe-wide, the leave-one-out mean is undefined and returned as
    ``np.nan``.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'market_ids': [0, 0, 1, 1, 2, 2],
    ...     'x':          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ... })
    >>> hausman(df, 'x')
    array([4.5, 4.5, 3.5, 3.5, 2.5, 2.5])
    """
    x = _column(product_data, column).astype(float, copy=False)
    mkt = _column(product_data, market_id_column)
    n = x.shape[0]
    out = np.full(n, np.nan, dtype=float)

    if period_column is None:
        for t in np.unique(mkt):
            idx_t = np.where(mkt == t)[0]
            other = np.where(mkt != t)[0]
            if other.size == 0:
                continue
            out[idx_t] = float(x[other].mean())
        return out

    period = _column(product_data, period_column)
    for p in np.unique(period):
        p_mask = period == p
        for t in np.unique(mkt[p_mask]):
            row_mask = p_mask & (mkt == t)
            other_mask = p_mask & (mkt != t)
            if not other_mask.any():
                continue
            out[row_mask] = float(x[other_mask].mean())
    return out


def bartik(
        product_data: Any,
        weight_column: str,
        shock_column: str,
        market_id_column: str = 'market_ids') -> np.ndarray:
    r"""Simple shift-share (Bartik) instrument.

    For each product :math:`j`, compute

    .. math::

        B_j = w_j \cdot \frac{1}{N - 1} \sum_{k \neq j} s_k

    where :math:`w_j` is ``weight_column`` at row :math:`j`, :math:`s_k` is
    ``shock_column`` at row :math:`k`, and the sum runs over all other rows
    in ``product_data`` (leave-one-out across the full panel).

    The ``market_id_column`` argument is accepted for API consistency with
    other helpers in this module but is not used by this simple variant;
    it is reserved for future market-grouped Bartik variants.

    This matches the most common applied usage of the Bartik lever: a local
    weight (e.g. industry employment share) times a national-level shock
    measured by the leave-one-out mean. If your paper's Bartik construction
    is different (e.g., inner-product over industries), use this module as
    a starting point and implement the specific formula in user code.

    Parameters
    ----------
    product_data : structured array-like
        Row per product.
    weight_column : str
        Weight column (e.g., local employment share).
    shock_column : str
        Shock column (e.g., industry productivity shifter). The leave-one-out
        average of this column is multiplied by the per-row weight.
    market_id_column : str, optional
        Accepted for consistency; not used by this simple variant. Defaults
        to ``'market_ids'``.

    Returns
    -------
    ndarray, shape ``(N,)``

    Notes
    -----
    The sum is the global leave-one-out mean: :math:`\frac{1}{N-1}
    \sum_{k \neq j} s_k`. If ``N == 1`` the result is ``np.nan``.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'market_ids': [0, 0, 1, 1],
    ...     'w':          [1.0, 2.0, 3.0, 4.0],
    ...     's':          [10.0, 20.0, 30.0, 40.0],
    ... })
    >>> bartik(df, 'w', 's')
    array([30.        , 53.33333333, 70.        , 80.        ])
    """
    w = _column(product_data, weight_column).astype(float, copy=False)
    s = _column(product_data, shock_column).astype(float, copy=False)
    # Touch market_id_column to validate it exists even though we don't use it.
    _column(product_data, market_id_column)

    n = s.shape[0]
    if n <= 1:
        return np.full(n, np.nan, dtype=float)
    total = float(s.sum())
    # Leave-one-out mean of s_k for k != j.
    loo_mean = (total - s) / (n - 1)
    return w * loo_mean


def concentration_hhi(
        product_data: Any,
        share_column: str = 'shares',
        firm_id_column: str = 'firm_ids',
        market_id_column: str = 'market_ids') -> np.ndarray:
    r"""Herfindahl-Hirschman Index, broadcast to each product in a market.

    For each market :math:`t`, compute

    .. math::

        \mathrm{HHI}_t = \sum_f \left( \sum_{j \in t,\; \mathrm{firm}(j) = f}
        s_j \right)^2

    i.e., the sum over firms of squared firm-level market shares. Every
    product in market :math:`t` receives the same value ``HHI_t``.

    Shares are taken at face value (no rescaling). If your ``share_column``
    is in percent rather than in :math:`[0, 1]`, the returned HHI is in
    percent-squared units accordingly.

    Parameters
    ----------
    product_data : structured array-like
        Row per product.
    share_column : str, optional
        Column with product-level shares. Defaults to ``'shares'``.
    firm_id_column : str, optional
        Firm ID column name. Defaults to ``'firm_ids'``.
    market_id_column : str, optional
        Market ID column name. Defaults to ``'market_ids'``.

    Returns
    -------
    ndarray, shape ``(N,)``
        HHI of each product's market.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'market_ids': [0, 0, 0, 1, 1],
    ...     'firm_ids':   [1, 1, 2, 1, 2],
    ...     'shares':     [0.3, 0.2, 0.5, 0.6, 0.4],
    ... })
    >>> concentration_hhi(df)
    array([0.5 , 0.5 , 0.5 , 0.52, 0.52])
    """
    shares = _column(product_data, share_column).astype(float, copy=False)
    firm = _column(product_data, firm_id_column)
    mkt = _column(product_data, market_id_column)
    n = shares.shape[0]
    out = np.zeros(n, dtype=float)

    for t in np.unique(mkt):
        idx = np.where(mkt == t)[0]
        if idx.size == 0:
            continue
        firms_t = firm[idx]
        shares_t = shares[idx]
        hhi_t = 0.0
        for f in np.unique(firms_t):
            f_share = float(shares_t[firms_t == f].sum())
            hhi_t += f_share * f_share
        out[idx] = hhi_t
    return out
