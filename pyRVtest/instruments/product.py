"""Product-side instrument constructors.

v0.4 step 13. Mechanical, vectorized helpers researchers can call to build
columns they pass to :class:`pyRVtest.Problem` via ``instrument_formulation``.

Each helper accepts ``product_data`` (a ``pandas.DataFrame``, ``dict``, or
structured ``numpy`` record array -- anything supporting ``d[col]`` lookup
and whose columns are ``ndarray``-like) and returns a numpy array (or a dict
of arrays) aligned row-for-row with ``product_data``.

Functions:

  - :func:`rival_sums` -- sum of a column over market-t rivals.
  - :func:`differentiation_ivs` -- Berry-Levinsohn-Pakes sum-of-squared
    differences of characteristics, split by own-firm vs rival-firm.
  - :func:`blp_instruments` -- classic BLP set: own_sum, own_count,
    rival_sum, rival_count per requested column.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

import numpy as np


__all__ = ['rival_sums', 'differentiation_ivs', 'blp_instruments']


def _column(product_data: Any, name: str) -> np.ndarray:
    """Return ``product_data[name]`` as a 1-D ``ndarray``.

    Accepts anything with ``__getitem__`` (DataFrame, dict, structured array).
    """
    col = product_data[name]
    arr = np.asarray(col)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def rival_sums(
        product_data: Any,
        column: str,
        firm_id_column: str = 'firm_ids',
        market_id_column: str = 'market_ids') -> np.ndarray:
    r"""Sum of ``column`` over rivals in the same market.

    For each product :math:`j` in market :math:`t`, returns

    .. math::

        \sum_{k \in t,\; \mathrm{firm}(k) \neq \mathrm{firm}(j)} x_k

    i.e., the sum of ``column`` over all products in the same market that
    are owned by a different firm than ``j``.

    Parameters
    ----------
    product_data : structured array-like
        Row per product. Must contain ``column``, ``firm_id_column``, and
        ``market_id_column``.
    column : str
        Name of the column to sum over.
    firm_id_column : str, optional
        Column in ``product_data`` with firm IDs. Defaults to ``'firm_ids'``.
    market_id_column : str, optional
        Column in ``product_data`` with market IDs. Defaults to
        ``'market_ids'``.

    Returns
    -------
    ndarray, shape ``(N,)``
        For each row ``j``, the sum of ``column`` over rivals in the same
        market. A product whose firm owns every product in the market gets
        ``0``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> df = pd.DataFrame({
    ...     'market_ids':  [0, 0, 0, 1, 1, 1],
    ...     'firm_ids':    [1, 1, 2, 1, 2, 2],
    ...     'x':           [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ... })
    >>> rival_sums(df, 'x')
    array([ 3.,  3.,  3., 11.,  4.,  4.])
    """
    x = _column(product_data, column).astype(float, copy=False)
    firm = _column(product_data, firm_id_column)
    mkt = _column(product_data, market_id_column)

    out = np.zeros_like(x, dtype=float)
    for t in np.unique(mkt):
        idx = np.where(mkt == t)[0]
        if idx.size == 0:
            continue
        total = float(x[idx].sum())
        firms_t = firm[idx]
        # Sum of x over same-firm products (including self).
        for j_pos, j in enumerate(idx):
            own_mask = firms_t == firms_t[j_pos]
            own_sum = float(x[idx[own_mask]].sum())
            out[j] = total - own_sum
    return out


def differentiation_ivs(
        product_data: Any,
        column: str,
        firm_id_column: str = 'firm_ids',
        market_id_column: str = 'market_ids') -> Dict[str, np.ndarray]:
    r"""Berry-Levinsohn-Pakes differentiation instruments.

    For each product :math:`j` in market :math:`t`, compute summary statistics
    of :math:`(x_j - x_k)^2` over same-market products :math:`k`, split by
    whether :math:`k` is owned by the same firm as :math:`j` or by a rival
    firm (excluding :math:`j` itself).

    Two instruments are returned:

    - ``sum_squared_diff_rival`` -- :math:`\sum_{k: \mathrm{firm}(k) \neq
      \mathrm{firm}(j),\; \mathrm{market}(k)=t} (x_j - x_k)^2`.
    - ``sum_squared_diff_same_firm`` -- :math:`\sum_{k: \mathrm{firm}(k) =
      \mathrm{firm}(j),\; k \neq j,\; \mathrm{market}(k)=t} (x_j - x_k)^2`.

    Parameters
    ----------
    product_data : structured array-like
        Row per product.
    column : str
        Characteristic column to take squared differences of.
    firm_id_column : str, optional
        Firm ID column name. Defaults to ``'firm_ids'``.
    market_id_column : str, optional
        Market ID column name. Defaults to ``'market_ids'``.

    Returns
    -------
    dict of str to ndarray
        ``{'sum_squared_diff_rival': (N,), 'sum_squared_diff_same_firm': (N,)}``.
        A row gets 0 in either slot if it has no qualifying partner (e.g., a
        single-firm market, or a product that is the only one its firm owns).

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'market_ids': [0, 0, 0],
    ...     'firm_ids':   [1, 1, 2],
    ...     'x':          [1.0, 3.0, 5.0],
    ... })
    >>> out = differentiation_ivs(df, 'x')
    >>> out['sum_squared_diff_rival']
    array([16.,  4., 20.])
    >>> out['sum_squared_diff_same_firm']
    array([4., 4., 0.])
    """
    x = _column(product_data, column).astype(float, copy=False)
    firm = _column(product_data, firm_id_column)
    mkt = _column(product_data, market_id_column)

    n = x.shape[0]
    out_rival = np.zeros(n, dtype=float)
    out_same = np.zeros(n, dtype=float)

    for t in np.unique(mkt):
        idx = np.where(mkt == t)[0]
        if idx.size == 0:
            continue
        xt = x[idx]
        firms_t = firm[idx]
        # Full matrix of squared differences within market t.
        diff = xt[:, None] - xt[None, :]
        sq = diff * diff
        for j_pos, j in enumerate(idx):
            same = firms_t == firms_t[j_pos]
            rival = ~same
            # Exclude self from same-firm sum.
            same_excl_self = same.copy()
            same_excl_self[j_pos] = False
            out_rival[j] = float(sq[j_pos, rival].sum())
            out_same[j] = float(sq[j_pos, same_excl_self].sum())
    return {
        'sum_squared_diff_rival': out_rival,
        'sum_squared_diff_same_firm': out_same,
    }


def blp_instruments(
        product_data: Any,
        columns: Iterable[str],
        firm_id_column: str = 'firm_ids',
        market_id_column: str = 'market_ids') -> Dict[str, np.ndarray]:
    r"""Classic BLP instrument set.

    For each column :math:`c` in ``columns`` and each product :math:`j` in
    market :math:`t`, return:

    - ``"<c>_own_sum"``: :math:`\sum_{k \in t,\; k \neq j,\; \mathrm{firm}(k)
      = \mathrm{firm}(j)} c_k`.
    - ``"<c>_rival_sum"``: :math:`\sum_{k \in t,\; \mathrm{firm}(k) \neq
      \mathrm{firm}(j)} c_k`.

    Plus two count columns that do not depend on ``c`` and are therefore
    returned just once:

    - ``"own_count"``: number of other products of :math:`j`'s firm in market
      :math:`t` (i.e., excluding :math:`j`).
    - ``"rival_count"``: number of rival products in market :math:`t`.

    Parameters
    ----------
    product_data : structured array-like
        Row per product.
    columns : sequence of str
        Characteristic columns to build own/rival sums for.
    firm_id_column : str, optional
        Firm ID column name. Defaults to ``'firm_ids'``.
    market_id_column : str, optional
        Market ID column name. Defaults to ``'market_ids'``.

    Returns
    -------
    dict of str to ndarray
        Keys are ``"{c}_own_sum"`` and ``"{c}_rival_sum"`` for each
        ``c in columns``, plus ``"own_count"`` and ``"rival_count"``.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'market_ids': [0, 0, 0, 0],
    ...     'firm_ids':   [1, 1, 2, 2],
    ...     'x':          [1.0, 2.0, 3.0, 4.0],
    ... })
    >>> out = blp_instruments(df, ['x'])
    >>> out['x_own_sum']
    array([2., 1., 4., 3.])
    >>> out['x_rival_sum']
    array([7., 7., 3., 3.])
    >>> out['own_count']
    array([1, 1, 1, 1])
    >>> out['rival_count']
    array([2, 2, 2, 2])
    """
    firm = _column(product_data, firm_id_column)
    mkt = _column(product_data, market_id_column)
    cols: List[str] = list(columns)

    n = firm.shape[0]
    own_count = np.zeros(n, dtype=int)
    rival_count = np.zeros(n, dtype=int)
    col_arrays: Mapping[str, np.ndarray] = {
        c: _column(product_data, c).astype(float, copy=False) for c in cols
    }
    own_sums = {c: np.zeros(n, dtype=float) for c in cols}
    rival_sums_arr = {c: np.zeros(n, dtype=float) for c in cols}

    for t in np.unique(mkt):
        idx = np.where(mkt == t)[0]
        if idx.size == 0:
            continue
        firms_t = firm[idx]
        for j_pos, j in enumerate(idx):
            same = firms_t == firms_t[j_pos]
            rival = ~same
            same_excl_self = same.copy()
            same_excl_self[j_pos] = False
            own_count[j] = int(same_excl_self.sum())
            rival_count[j] = int(rival.sum())
            for c in cols:
                xt = col_arrays[c][idx]
                own_sums[c][j] = float(xt[same_excl_self].sum())
                rival_sums_arr[c][j] = float(xt[rival].sum())

    out: Dict[str, np.ndarray] = {}
    for c in cols:
        out[f'{c}_own_sum'] = own_sums[c]
        out[f'{c}_rival_sum'] = rival_sums_arr[c]
    out['own_count'] = own_count
    out['rival_count'] = rival_count
    return out
