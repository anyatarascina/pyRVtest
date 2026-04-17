"""Product data structured as a record array.

v0.4 step 2: extracted verbatim from `pyRVtest/problem.py`. No logic
change. Type hints added to local variables where mypy --strict
requires explicit declarations. PyBLP internals (`Array`, `Data`,
`RecArray`, `Groups`, `extract_matrix`, `structure_matrices`) are
imported from `pyblp.utilities.basics`; they do not ship mypy stubs,
so a small number of `# type: ignore[...]` annotations appear where
mypy can't resolve pyblp return types. Each annotation carries a
short comment naming the gap (per plan §7 Open Question 7).

The `Products` class itself is a thin wrapper around `structure_matrices`
that validates and organizes product-level inputs (market_ids, shares,
prices, cost shifters w, instruments Z). Its `__new__` returns a
`RecArray`; callers use it as a structured view, not a user-facing
object.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from pyblp.utilities.basics import (
    Array, Data, Groups, RecArray, extract_matrix, structure_matrices
)

from . import options
from .formulation import Formulation


__all__ = ['Products']


class Products(object):
    r"""Product data structured as a record array.

    Attributes
    ----------
     market_ids : `ndarray`
        IDs that associate product_data with markets.
    cost_ids : `ndarray`
        IDs that associate product_data with cost-side fixed effects.
    nesting_ids : `ndarray`
        IDs that associate product_data with nesting groups.
    product_ids : `ndarray`
        IDs that identify product_data within markets.
    clustering_ids : `ndarray`
        IDs used to compute clustered standard errors.
    shares : `ndarray`
        Market shares, :math:`s`.
    prices : `ndarray`
        Product prices, :math:`p`.
    Z : `ndarray`
        Instruments, :math: `Z`.
    w : `ndarray`
        Cost-shifters, :math: `w`.

    Examples
    --------
    >>> import pandas as pd
    >>> import pyRVtest
    >>> df = pd.DataFrame({
    ...     'market_ids': [0, 0, 1, 1],
    ...     'shares':     [0.3, 0.3, 0.4, 0.4],
    ...     'prices':     [1.0, 2.0, 1.5, 2.5],
    ...     'w_shifter':  [0.1, 0.2, 0.3, 0.4],
    ...     'iv':         [0.5, 0.7, 0.9, 1.1],
    ... })
    >>> products = pyRVtest.Products(
    ...     cost_formulation=pyRVtest.Formulation('0 + w_shifter'),
    ...     instrument_formulation=pyRVtest.Formulation('0 + iv'),
    ...     product_data=df,
    ... )
    >>> products.shares.shape
    (4, 1)
    >>> products.prices.shape
    (4, 1)
    """

    market_ids: Array
    cost_ids: Array
    nesting_ids: Array
    product_ids: Array
    clustering_ids: Array
    shares: Array
    prices: Array
    Z: Array
    w: Array

    def __new__(
            cls, cost_formulation: Formulation, instrument_formulation: Sequence[Optional[Formulation]],
            product_data: Mapping[str, Any]) -> RecArray:
        """Structure product data."""

        # validate the cost formulations
        if not isinstance(cost_formulation, Formulation):
            raise TypeError("cost_formulation must be a Formulation instance or None.")
        if cost_formulation is None:
            raise ValueError("The formulation for marginal cost must be specified.")

        # build w
        w, w_formulation, w_data = cost_formulation._build_matrix(product_data)

        # check that prices are not in X1
        if 'prices' in w_data:
            raise NameError("prices cannot be included in the formulation for marginal cost.")

        # validate the instrument formulation
        if instrument_formulation is None:
            raise ValueError("The formulation for instruments for testing must be specified.")
        if hasattr(instrument_formulation, '__len__'):
            L = len(instrument_formulation)
            if not all(isinstance(f, Formulation) for f in instrument_formulation):
                raise TypeError("Each formulation in instrument_formulation must be a Formulation.")
        else:
            L = 1
            if not isinstance(instrument_formulation, Formulation):
                raise TypeError("instrument_formulation must be a single Formulation instance.")

        # build Z. `instrument_formulation` is typed as a Sequence for multi-instrument callers,
        # but the single-instrument branch also accepts a bare Formulation — mypy sees only the
        # declared Sequence type, so suppress the narrowing here with a scoped ignore.
        Z: Data = {}
        if L == 1:
            Z_l, Z_formulation_l, Z_data_l = (
                instrument_formulation._build_matrix(product_data)  # type: ignore[attr-defined]
            )
            for z in Z_formulation_l:
                if z in w_data:
                    raise NameError("Z must be excluded from marginal cost.")
            Z["Z0"] = Z_l
            Z["Z0_formulation"] = Z_formulation_l
            Z["Z0_data"] = Z_data_l
        elif L > 1:
            for l in range(L):
                f_l = instrument_formulation[l]
                assert f_l is not None  # L>1 branch guarantees all entries present and Formulation
                Z_l, Z_formulation_l, Z_data_l = f_l._build_matrix(product_data)
                for z in Z_formulation_l:
                    if z in w_data:
                        raise NameError("Z must be excluded from marginal cost.")
                Z["Z{0}".format(l)] = Z_l
                Z["Z{0}_formulation".format(l)] = Z_formulation_l
                Z["Z{0}_data".format(l)] = Z_data_l

        # load fixed effect IDs
        cost_ids: Optional[Array] = None
        if cost_formulation._absorbed_terms:
            cost_ids = cost_formulation._build_ids(product_data)

        # load other IDs
        market_ids = extract_matrix(product_data, 'market_ids')
        nesting_ids = extract_matrix(product_data, 'nesting_ids')
        product_ids = extract_matrix(product_data, 'product_ids')
        clustering_ids = extract_matrix(product_data, 'clustering_ids')
        if market_ids is None:
            raise KeyError("product_data must have a market_ids field.")
        if market_ids.shape[1] > 1:
            raise ValueError("The market_ids field of product_data must be one-dimensional.")
        if nesting_ids is not None and nesting_ids.shape[1] > 1:
            raise ValueError("The nesting_ids field of product_data must be one-dimensional.")
        if product_ids is not None and product_ids.shape[1] > 1:
            raise ValueError("The product_ids field of product_data must be one-dimensional.")
        if clustering_ids is not None:
            if clustering_ids.shape[1] > 1:
                raise ValueError("The clustering_ids field of product_data must be one-dimensional.")
            if np.unique(clustering_ids).size == 1:
                raise ValueError("The clustering_ids field of product_data must have at least two distinct IDs.")

        # load shares
        shares = extract_matrix(product_data, 'shares')
        if shares is None:
            raise KeyError("product_data must have a shares field.")
        if shares.shape[1] > 1:
            raise ValueError("The shares field of product_data must be one-dimensional.")
        if (shares <= 0).any() or (shares >= 1).any():
            raise ValueError(
                "The shares field of product_data must consist of values between zero and one, exclusive.")

        # verify that shares sum to less than one in each market
        market_groups = Groups(market_ids)
        bad_shares_index = market_groups.sum(shares) >= 1
        if np.any(bad_shares_index):
            bad_market_ids = market_groups.unique[bad_shares_index.flat]
            raise ValueError(f"Shares in these markets do not sum to less than 1: {bad_market_ids}.")

        # load prices
        prices = extract_matrix(product_data, 'prices')
        if prices is None:
            raise KeyError("product_data must have a prices field.")
        if prices.shape[1] > 1:
            raise ValueError("The prices field of product_data must be one-dimensional.")
        if (prices < 0).any():
            raise ValueError(
                "The prices field of product_data must consist of values >= zero, exclusive.")

        # structure product fields as a mapping. Keys are either plain strings
        # (e.g. 'market_ids') or (formulation_tuple, column_name) pairs for w/Z.
        ProductKey = Union[str, Tuple[Any, ...]]
        product_mapping: Dict[ProductKey, Tuple[Optional[Array], Any]] = {
            'market_ids': (market_ids, np.object_),
            'cost_ids': (cost_ids, np.object_),
            'nesting_ids': (nesting_ids, np.object_),
            'product_ids': (product_ids, np.object_),
            'clustering_ids': (clustering_ids, np.object_),
            'shares': (shares, options.dtype),
            'prices': (prices, options.dtype),
        }
        product_mapping[(tuple(w_formulation), 'w')] = (w, options.dtype)
        for l in range(L):
            key: ProductKey = (tuple(Z["Z{0}_formulation".format(l)]), 'Z{0}'.format(l))
            product_mapping[key] = (Z["Z{0}".format(l)], options.dtype)

        # structure and validate variables underlying instruments
        underlying_data: Dict[str, Tuple[Array, Any]] = {
            k: (v, options.dtype) for k, v in {**w_data}.items() if k != 'shares'
        }
        for l in range(L):
            underlying_data.update({
                k: (v, options.dtype) for k, v in {**Z["Z{0}_data".format(l)]}.items() if k != 'shares'
            })
        invalid_names = set(underlying_data) & {k if isinstance(k, str) else k[1] for k in product_mapping}
        if invalid_names:
            raise NameError(f"These reserved names in product_formulations are invalid: {list(invalid_names)}.")

        combined: Dict[ProductKey, Tuple[Optional[Array], Any]] = dict(product_mapping)
        for key_str, value in underlying_data.items():
            combined[key_str] = value
        return structure_matrices(combined)
