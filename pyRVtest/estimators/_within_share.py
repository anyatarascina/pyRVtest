"""Helper for auto-constructing a within-nest instrument from product_data."""

from __future__ import annotations

import logging
from typing import Any, Mapping, Tuple

import numpy as np
from pyblp.utilities.basics import Array

from ..exceptions import ValidationError


logger = logging.getLogger(__name__)


__all__ = ['count_in_nest_iv']


def count_in_nest_iv(
        product_data: Mapping[str, Any],
        market_ids_column: str,
        nesting_ids_column: str,
        column_name: str = 'count_in_nest_iv',
) -> Tuple[Array, str]:
    r"""Build an excluded instrument for the within-nest log-share regressor.

    For one-level nested logit, the within-nest log-share
    :math:`\log s_{j|g}` is endogenous (it depends on :math:`\xi`). The
    standard "BLP-style" instrument for it is the count of products in
    the same nest within the same market: :math:`\#\{j' \in g_{jt}\}`.
    This is exogenous as long as nest membership is exogenous.

    Returns the array of counts and the suggested column name (so the
    caller can append it to ``product_data`` and to the formula
    string).
    """
    market_ids = np.asarray(product_data[market_ids_column]).flatten()
    nesting_ids = np.asarray(product_data[nesting_ids_column]).flatten()
    if market_ids.shape != nesting_ids.shape:
        raise ValidationError(
            f"Expected market_ids and nesting_ids to have the same length. "
            f"Received {market_ids.shape} and {nesting_ids.shape}. "
            f"Fix: confirm that {market_ids_column!r} and {nesting_ids_column!r} "
            f"are aligned with product_data rows."
        )
    counts = np.zeros(market_ids.shape[0], dtype=float)
    for t in np.unique(market_ids):
        idx = np.where(market_ids == t)[0]
        nest_t = nesting_ids[idx]
        for g in np.unique(nest_t):
            mask = nest_t == g
            counts[idx[mask]] = mask.sum()
    return counts, column_name
