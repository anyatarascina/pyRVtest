"""One-level nested-logit 2SLS estimator.

Fits

.. math::

    \\log s_{jt} - \\log s_{0t}
        = X_{jt}'\\beta + \\alpha p_{jt} + \\rho \\log s_{j|g_t} + \\xi_{jt}

by 2SLS with two endogenous regressors (:math:`p` and :math:`\\log s_{j|g}`)
and the user-supplied excluded instruments. Multi-level nesting is out
of scope for this implementation; the literature mostly uses one-level
nesting and the math + auto-IV story for higher levels is a follow-up.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
from pyblp.utilities.basics import Array

from ..exceptions import ValidationError
from ..formulation import Formulation
from ._base import (
    build_design_from_formulation,
    compute_delta,
    extract_array_column,
    two_stage_least_squares,
)
from ._within_share import count_in_nest_iv
from .logit import _reorder_weight_matrix


logger = logging.getLogger(__name__)


__all__ = ['NestedLogitEstimator']


class NestedLogitEstimator:
    r"""2SLS estimator for one-level nested logit demand.

    Estimates :math:`(\alpha, \beta, \rho)` in

    .. math::

        \log s_{jt} - \log s_{0t}
            = X_{jt}'\beta + \alpha p_{jt} + \rho \log s_{j|g_t} + \xi_{jt}

    Two endogenous regressors (:math:`p` and :math:`\log s_{j|g}`) means
    ``formulation_Z`` must supply at least two excluded instruments for
    the order condition. Set ``auto_construct_within_share_iv=True`` to
    have the estimator append a ``count_in_nest`` column to the
    instrument set automatically (its standard within-nest IV).

    Parameters
    ----------
    product_data
        DataFrame, structured numpy array, or dict-of-arrays.
    formulation_X
        Non-price demand-side regressors; ``None`` if none.
    formulation_Z
        Excluded instruments for the two endogenous regressors. Must
        contain :math:`\geq 2` columns unless
        ``auto_construct_within_share_iv=True`` (then :math:`\geq 1` is
        OK; the auto-IV is appended to satisfy the order condition).
    nesting_ids_column
        Column with the nest assignments. One nest per row.
    market_ids_column, shares_column, prices_column
        Column-name overrides; defaults are the pyRVtest standards.
    auto_construct_within_share_iv
        If True, append ``count_in_nest_iv`` to the instrument set. The
        new instrument column is also added to a copy of
        ``product_data`` that the caller receives via the
        ``product_data`` attribute on the estimator (so the user can
        feed it to ``Problem`` if needed).
    W_demand
        Optional :math:`K_w \times K_w` GMM weight matrix. Defaults to
        :math:`(W'W/N)^{-1}`.

    Examples
    --------
    >>> import pandas as pd
    >>> import pyRVtest
    >>> df = pd.read_csv('product_data.csv')  # doctest: +SKIP
    >>> est = pyRVtest.NestedLogitEstimator(  # doctest: +SKIP
    ...     product_data=df,
    ...     formulation_X=pyRVtest.Formulation('1 + size'),
    ...     formulation_Z=pyRVtest.Formulation('0 + cost1 + cost2'),
    ...     nesting_ids_column='nest',
    ... )
    >>> demand_params = est.solve()  # doctest: +SKIP
    """

    def __init__(
            self,
            product_data: Mapping[str, Any],
            formulation_X: Optional[Formulation],
            formulation_Z: Formulation,
            nesting_ids_column: str,
            market_ids_column: str = 'market_ids',
            shares_column: str = 'shares',
            prices_column: str = 'prices',
            auto_construct_within_share_iv: bool = False,
            W_demand: Optional[Array] = None,
    ) -> None:
        if formulation_Z is None:
            raise ValidationError(
                "Expected formulation_Z to be a pyRVtest.Formulation listing "
                "excluded instruments for price and the within-nest log-share. "
                "Received formulation_Z=None. "
                "Fix: pass at least two excluded instruments, "
                "or set auto_construct_within_share_iv=True and pass at least "
                "one excluded instrument for price."
            )
        if not nesting_ids_column:
            raise ValidationError(
                "Expected nesting_ids_column to be the name of the nest-ID "
                "column in product_data. "
                "Received an empty value. "
                "Fix: pass nesting_ids_column='nest' (or whichever column "
                "labels nest membership)."
            )
        self._product_data = product_data
        self._formulation_X = formulation_X
        self._formulation_Z = formulation_Z
        self._nesting_ids_column = nesting_ids_column
        self._market_ids_column = market_ids_column
        self._shares_column = shares_column
        self._prices_column = prices_column
        self._auto_iv = bool(auto_construct_within_share_iv)
        self._W_demand_user = W_demand

        self.alpha: Optional[float] = None
        self.beta: Optional[Array] = None
        self.rho: Optional[float] = None
        self.xi: Optional[Array] = None
        self.x_columns: Optional[List[str]] = None
        self.demand_instrument_columns: Optional[List[str]] = None
        self.W_demand: Optional[Array] = None
        self.product_data: Optional[Mapping[str, Any]] = None

    def solve(self) -> Dict[str, Any]:
        """Run 2SLS and return a populated ``demand_params`` dict.

        Returns
        -------
        dict
            Same keys as :meth:`~pyRVtest.LogitEstimator.solve` plus
            ``rho`` (single-element list ``[rho_hat]``) and
            ``nesting_ids_columns`` (single-element list with the
            nesting-id column name).
        """
        prices = extract_array_column(self._product_data, self._prices_column)
        shares = extract_array_column(self._product_data, self._shares_column)
        market_ids = np.asarray(self._product_data[self._market_ids_column]).flatten()
        nesting_ids = np.asarray(self._product_data[self._nesting_ids_column]).flatten()

        delta = compute_delta(shares, market_ids)
        log_within_nest = _log_within_nest_shares(shares, market_ids, nesting_ids)

        product_data_for_iv: Mapping[str, Any] = self._product_data
        if self._auto_iv:
            counts, count_col = count_in_nest_iv(
                self._product_data,
                market_ids_column=self._market_ids_column,
                nesting_ids_column=self._nesting_ids_column,
            )
            product_data_for_iv = _augment_with_column(
                self._product_data, count_col, counts,
            )

        X, x_columns = build_design_from_formulation(
            self._formulation_X, product_data_for_iv, role='formulation_X',
        )
        Z_excl, z_excl_columns = build_design_from_formulation(
            self._formulation_Z, product_data_for_iv, role='formulation_Z',
        )

        if self._auto_iv:
            counts_arr = np.asarray(product_data_for_iv[count_col]).flatten().astype(float)
            Z_excl = np.column_stack([Z_excl, counts_arr])
            z_excl_columns = list(z_excl_columns) + [count_col]

        if 'prices' in x_columns:
            raise ValidationError(
                "Expected formulation_X to exclude the 'prices' column "
                "(price is the endogenous regressor). "
                "Received formulation_X containing 'prices'. "
                "Fix: drop 'prices' from formulation_X."
            )

        K_x = X.shape[1]
        if Z_excl.shape[1] < 2:
            raise ValidationError(
                f"Expected at least 2 excluded instruments for nested logit "
                f"(price and within-nest log-share are both endogenous). "
                f"Received {Z_excl.shape[1]} excluded instrument(s). "
                f"Fix: add a second excluded instrument to formulation_Z, or "
                f"set auto_construct_within_share_iv=True to use the "
                f"count-of-products-in-nest as the second IV automatically."
            )

        # D = [X | prices | log_within_nest]; W = [X | Z_excl].
        # theta layout: theta[:K_x] = beta; theta[K_x] = alpha; theta[K_x+1] = rho.
        if K_x:
            D = np.column_stack([X, prices, log_within_nest])
            W = np.column_stack([X, Z_excl])
        else:
            D = np.column_stack([prices, log_within_nest])
            W = Z_excl

        theta, omega = two_stage_least_squares(
            delta=delta, D=D, W=W, weight=self._W_demand_user,
        )

        beta = theta[:K_x] if K_x else np.zeros(0)
        alpha = float(theta[K_x])
        rho = float(theta[K_x + 1])

        if alpha >= 0:
            warnings.warn(
                f"NestedLogitEstimator returned alpha={alpha:.4f} >= 0; "
                "this implies upward-sloping demand. "
                "Possible causes: weak instruments, mis-signed instruments, "
                "or model misspecification. "
                "Fix: check that excluded instruments shift price in the "
                "expected direction and have first-stage F > 10.",
                stacklevel=2,
            )
        if not (0 <= rho < 1):
            warnings.warn(
                f"NestedLogitEstimator returned rho={rho:.4f}, outside the "
                "[0, 1) range required for nested-logit utility maximization. "
                "Common causes: weak within-nest instrument, model "
                "misspecification (try plain logit), or insufficient "
                "within-nest variation. "
                "The estimate is reported as-is; the caller should decide "
                "whether to clip, refit with stronger instruments, or fall "
                "back to plain logit.",
                stacklevel=2,
            )

        xi = delta - D @ theta

        demand_instrument_columns = list(z_excl_columns) + list(x_columns)

        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.xi = xi
        self.x_columns = list(x_columns)
        self.demand_instrument_columns = demand_instrument_columns
        self.W_demand = _reorder_weight_matrix(
            omega, source_order=list(x_columns) + list(z_excl_columns),
            target_order=demand_instrument_columns,
        )
        self.product_data = product_data_for_iv

        logger.info(
            "NestedLogitEstimator: alpha=%.6f, rho=%.6f on N=%d obs, "
            "T=%d markets, K_x=%d, K_z_excluded=%d.",
            alpha, rho, prices.shape[0], len(np.unique(market_ids)),
            K_x, Z_excl.shape[1],
        )

        return {
            'alpha': alpha,
            'beta': beta,
            'rho': [rho],
            'x_columns': self.x_columns,
            'demand_instrument_columns': self.demand_instrument_columns,
            'W_demand': self.W_demand,
            'nesting_ids_columns': [self._nesting_ids_column],
        }


def _log_within_nest_shares(
        shares: Array, market_ids: Array, nesting_ids: Array,
) -> Array:
    """Compute ``log(s_{j|g_t})`` per observation."""
    N = shares.shape[0]
    out = np.zeros(N)
    for t in np.unique(market_ids):
        idx = np.where(market_ids == t)[0]
        nest_t = nesting_ids[idx]
        s_t = shares[idx]
        for g in np.unique(nest_t):
            mask = nest_t == g
            nest_total = s_t[mask].sum()
            if nest_total <= 0:
                raise ValidationError(
                    f"Expected each nest in each market to contain products "
                    f"with positive total share for log(s_{{j|g}}) to be "
                    f"defined. "
                    f"Received nest {g!r} in market {t!r} with total share "
                    f"{nest_total:.6g}. "
                    f"Fix: drop or aggregate empty nests before estimation."
                )
            out[idx[mask]] = np.log(s_t[mask] / nest_total)
    return out


def _augment_with_column(
        product_data: Mapping[str, Any], column_name: str, values: Array,
) -> Mapping[str, Any]:
    """Return a copy of product_data with an additional column appended.

    Mirrors the behaviour of ``Problem._augment_with_intercept_column``
    but is parametric over name/values. Never mutates the input.
    """
    import pandas as pd
    if hasattr(product_data, 'assign'):
        result_df: Mapping[str, Any] = product_data.assign(**{column_name: values})
        return result_df
    if hasattr(product_data, 'dtype') and product_data.dtype.names:
        df = pd.DataFrame({name: product_data[name] for name in product_data.dtype.names})
        df[column_name] = values
        result_recarray: Mapping[str, Any] = df
        return result_recarray
    if hasattr(product_data, 'keys'):
        out = {k: product_data[k] for k in product_data.keys()}
        out[column_name] = values
        return out
    raise TypeError(
        f"Expected product_data to be a pandas DataFrame, structured numpy "
        f"array, or dict-like mapping for auto-IV augmentation. "
        f"Received {type(product_data).__name__}."
    )
