"""Product data structured as a record array.

extracted verbatim from `pyRVtest/problem.py`. No logic
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
from .exceptions import InstrumentDataError
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
            raise TypeError(
                f"Expected cost_formulation to be a pyRVtest.Formulation instance. "
                f"Received {type(cost_formulation).__name__}. "
                f"Fix: wrap your cost-shifter formula string in Formulation(...)."
            )
        if cost_formulation is None:
            raise ValueError(
                "Expected cost_formulation to specify the marginal-cost formula. "
                "Received None. "
                "Fix: pass cost_formulation=Formulation('0 + <cost-shifters>')."
            )

        # build w
        w, w_formulation, w_data = cost_formulation._build_matrix(product_data)

        # check that prices are not in X1
        if 'prices' in w_data:
            raise NameError(
                "Expected the cost_formulation to exclude the 'prices' column. "
                "Received a formulation that includes 'prices'. "
                "Fix: drop 'prices' from cost_formulation; it is the dependent "
                "variable, not a cost shifter."
            )

        # validate the instrument formulation
        if instrument_formulation is None:
            raise ValueError(
                "Expected instrument_formulation to specify testing instruments. "
                "Received None. "
                "Fix: pass instrument_formulation=Formulation('0 + <instruments>'), "
                "or a list of Formulations for multiple instrument sets."
            )
        if hasattr(instrument_formulation, '__len__'):
            L = len(instrument_formulation)
            if not all(isinstance(f, Formulation) for f in instrument_formulation):
                raise TypeError(
                    "Expected every entry in instrument_formulation to be a "
                    "pyRVtest.Formulation instance. "
                    "Received a sequence containing non-Formulation values. "
                    "Fix: wrap each instrument-set formula in Formulation(...) "
                    "before passing the list."
                )
        else:
            L = 1
            if not isinstance(instrument_formulation, Formulation):
                raise TypeError(
                    f"Expected instrument_formulation to be a single pyRVtest.Formulation "
                    f"or a sequence of them. "
                    f"Received {type(instrument_formulation).__name__}. "
                    f"Fix: pass Formulation('0 + <instruments>')."
                )

        # build Z. `instrument_formulation` is typed as a Sequence for multi-instrument callers,
        # but the single-instrument branch also accepts a bare Formulation — mypy sees only the
        # declared Sequence type, so suppress the narrowing here with a scoped ignore.
        Z: Data = {}
        if L == 1:
            Z_l, Z_formulation_l, Z_data_l = (
                instrument_formulation._build_matrix(product_data)  # type: ignore[union-attr]
            )
            for z in Z_formulation_l:
                if z in w_data:
                    raise InstrumentDataError(
                        "Expected instrument columns (Z) to be disjoint from the "
                        "marginal-cost shifters (w). "
                        "Received an instrument variable that also appears in "
                        "cost_formulation; Z must be excluded from marginal cost. "
                        "Fix: drop the duplicate variable from either cost_formulation "
                        "or instrument_formulation (variables cannot serve both roles)."
                    )
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
                        raise InstrumentDataError(
                            "Expected instrument columns (Z) to be disjoint from the "
                            "marginal-cost shifters (w). "
                            "Received an instrument variable that also appears in "
                            "cost_formulation; Z must be excluded from marginal cost. "
                            "Fix: drop the duplicate variable from either "
                            "cost_formulation or instrument_formulation (variables "
                            "cannot serve both roles)."
                        )
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
            raise KeyError(
                "Expected product_data to contain a 'market_ids' column. "
                "Received product_data without that key. "
                "Fix: add a 'market_ids' column identifying the market each product belongs to."
            )
        if market_ids.shape[1] > 1:
            raise ValueError(
                f"Expected the 'market_ids' column to be one-dimensional. "
                f"Received shape {market_ids.shape}. "
                f"Fix: pass a single vector of market identifiers, not a multi-column array."
            )
        if nesting_ids is not None and nesting_ids.shape[1] > 1:
            raise ValueError(
                f"Expected the 'nesting_ids' column to be one-dimensional. "
                f"Received shape {nesting_ids.shape}. "
                f"Fix: pass a single vector of nesting identifiers."
            )
        if product_ids is not None and product_ids.shape[1] > 1:
            raise ValueError(
                f"Expected the 'product_ids' column to be one-dimensional. "
                f"Received shape {product_ids.shape}. "
                f"Fix: pass a single vector of product identifiers."
            )
        if clustering_ids is not None:
            if clustering_ids.shape[1] > 1:
                raise ValueError(
                    f"Expected the 'clustering_ids' column to be one-dimensional. "
                    f"Received shape {clustering_ids.shape}. "
                    f"Fix: pass a single vector of clustering identifiers."
                )
            if np.unique(clustering_ids).size == 1:
                raise ValueError(
                    "Expected 'clustering_ids' to contain at least 2 distinct IDs "
                    "so cluster-robust covariance is identified. "
                    "Received a column with 1 unique ID. "
                    "Fix: drop the clustering_ids column, or supply a column with "
                    "multiple clusters."
                )

        # load shares
        shares = extract_matrix(product_data, 'shares')
        if shares is None:
            raise KeyError(
                "Expected product_data to contain a 'shares' column. "
                "Received product_data without that key. "
                "Fix: add a 'shares' column with per-product market shares in (0, 1)."
            )
        if shares.shape[1] > 1:
            raise ValueError(
                f"Expected the 'shares' column to be one-dimensional. "
                f"Received shape {shares.shape}. "
                f"Fix: pass a single vector of shares per product."
            )
        if (shares <= 0).any() or (shares >= 1).any():
            raise ValueError(
                "Expected every entry of 'shares' to lie strictly in the open "
                "interval (0, 1). "
                "Received at least one share outside that range. "
                "Fix: clip or drop products with non-interior shares, or check "
                "that shares are fractions, not percentages."
            )

        # verify that shares sum to less than one in each market
        market_groups = Groups(market_ids)
        bad_shares_index = market_groups.sum(shares) >= 1
        if np.any(bad_shares_index):
            bad_market_ids = market_groups.unique[bad_shares_index.flat]
            raise ValueError(
                f"Expected within-market shares to sum to less than 1 (leaving "
                f"room for the outside option). "
                f"Received markets whose shares sum to >= 1: {bad_market_ids}. "
                f"Fix: renormalize shares, or drop the affected markets."
            )

        # load prices
        prices = extract_matrix(product_data, 'prices')
        if prices is None:
            raise KeyError(
                "Expected product_data to contain a 'prices' column. "
                "Received product_data without that key. "
                "Fix: add a non-negative 'prices' column."
            )
        if prices.shape[1] > 1:
            raise ValueError(
                f"Expected the 'prices' column to be one-dimensional. "
                f"Received shape {prices.shape}. "
                f"Fix: pass a single vector of prices per product."
            )
        if (prices < 0).any():
            raise ValueError(
                "Expected every entry of 'prices' to be non-negative. "
                "Received at least one negative price. "
                "Fix: check the units of the prices column; all values must be >= 0."
            )

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
            raise NameError(
                f"Expected formulation variables to avoid reserved pyRVtest names "
                f"(market_ids, shares, prices, 'w', 'Z0', ...). "
                f"Received these reserved names in cost/instrument formulations: "
                f"{list(invalid_names)}. "
                f"Fix: rename the offending columns in product_data and in the "
                f"formulation strings."
            )

        combined: Dict[ProductKey, Tuple[Optional[Array], Any]] = dict(product_mapping)
        for key_str, value in underlying_data.items():
            combined[key_str] = value
        return structure_matrices(combined)
