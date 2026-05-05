"""Plain-logit 2SLS estimator (Berry 1994 inversion + linear IV)."""

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


logger = logging.getLogger(__name__)


__all__ = ['LogitEstimator']


class LogitEstimator:
    r"""2SLS estimator for plain-logit demand (Berry 1994 inversion).

    Estimates :math:`(\alpha, \beta)` in

    .. math::

        \log s_{jt} - \log s_{0t} = X_{jt}' \beta + \alpha p_{jt} + \xi_{jt}

    by 2SLS with ``prices`` as the only endogenous regressor and
    ``formulation_Z`` providing the excluded instruments. ``formulation_X``
    is the non-price portion of the demand-side regressors (typically a
    constant and observed product attributes).

    The default weight matrix is :math:`(W'W/N)^{-1}` where
    :math:`W = [X, Z_{\text{excluded}}]` is the full instrument matrix
    (this is standard 2SLS). The same weight is stored on the returned
    dict as ``W_demand`` so the demand-adjustment correction in
    :class:`~pyRVtest.Problem.solve` (DMSS Appendix C eq. 77) reuses
    consistent moment weights.

    Parameters
    ----------
    product_data
        DataFrame, structured numpy array, or dict-of-arrays. Must contain
        ``market_ids_column``, ``shares_column``, ``prices_column``, and
        every variable referenced by ``formulation_X`` and
        ``formulation_Z``.
    formulation_X
        :class:`~pyRVtest.Formulation` for the non-price demand-side
        regressors (e.g., ``Formulation('1 + size + brand_id')``). Pass
        ``None`` if the model has no non-price covariates.
    formulation_Z
        :class:`~pyRVtest.Formulation` for the excluded instruments
        (cost shifters or BLP-style rival-characteristics aggregates).
        Must have at least one column.
    market_ids_column
        Column name for market IDs. Default ``'market_ids'``.
    shares_column
        Column name for inside-good shares. Default ``'shares'``.
    prices_column
        Column name for prices. Default ``'prices'``.
    W_demand
        Optional :math:`K_w \times K_w` GMM weight matrix for the moment
        condition :math:`W' \xi = 0`, where :math:`K_w` is the total
        number of columns in :math:`[X, Z_{\text{excluded}}]`. If
        ``None``, defaults to :math:`(W'W/N)^{-1}` (standard 2SLS).

    Examples
    --------
    >>> import pandas as pd
    >>> import pyRVtest
    >>> df = pd.read_csv('product_data.csv')  # doctest: +SKIP
    >>> est = pyRVtest.LogitEstimator(  # doctest: +SKIP
    ...     product_data=df,
    ...     formulation_X=pyRVtest.Formulation('1 + size'),
    ...     formulation_Z=pyRVtest.Formulation('0 + cost_shifter'),
    ... )
    >>> demand_params = est.solve()  # doctest: +SKIP
    >>> demand_params['alpha']  # doctest: +SKIP
    -2.13...
    """

    def __init__(
            self,
            product_data: Mapping[str, Any],
            formulation_X: Optional[Formulation],
            formulation_Z: Formulation,
            market_ids_column: str = 'market_ids',
            shares_column: str = 'shares',
            prices_column: str = 'prices',
            W_demand: Optional[Array] = None,
    ) -> None:
        if formulation_Z is None:
            raise ValidationError(
                "Expected formulation_Z to be a pyRVtest.Formulation listing "
                "the excluded instruments (cost shifters or rival-character "
                "aggregates) for price. "
                "Received formulation_Z=None. "
                "Fix: pass at least one excluded instrument, "
                "e.g., formulation_Z=Formulation('0 + cost_shifter')."
            )
        self._product_data = product_data
        self._formulation_X = formulation_X
        self._formulation_Z = formulation_Z
        self._market_ids_column = market_ids_column
        self._shares_column = shares_column
        self._prices_column = prices_column
        self._W_demand_user = W_demand

        # Filled in by solve().
        self.alpha: Optional[float] = None
        self.beta: Optional[Array] = None
        self.xi: Optional[Array] = None
        self.x_columns: Optional[List[str]] = None
        self.demand_instrument_columns: Optional[List[str]] = None
        self.W_demand: Optional[Array] = None

    def solve(self) -> Dict[str, Any]:
        """Run the 2SLS estimation and return a populated ``demand_params`` dict.

        Returns
        -------
        dict
            Keys:

            - ``alpha`` (float) — estimated price coefficient.
            - ``beta`` (ndarray) — estimated coefficients on
              ``x_columns``, in the same order.
            - ``rho`` (list) — empty list ``[]`` for plain logit.
            - ``x_columns`` (list of str) — non-price regressor names.
            - ``demand_instrument_columns`` (list of str) — full
              :math:`Z_D` columns used in the demand-adjustment moment
              (excluded instruments concatenated with ``x_columns``).
            - ``W_demand`` (ndarray) — weight matrix used for 2SLS,
              reused by the demand-adjustment correction.

        The dict is directly consumable by
        ``Problem(demand_params=...)``.
        """
        prices = extract_array_column(self._product_data, self._prices_column)
        shares = extract_array_column(self._product_data, self._shares_column)
        market_ids_raw = self._product_data[self._market_ids_column]
        market_ids = np.asarray(market_ids_raw).flatten()

        delta = compute_delta(shares, market_ids)

        X, x_columns = build_design_from_formulation(
            self._formulation_X, self._product_data, role='formulation_X',
        )
        Z_excl, z_excl_columns = build_design_from_formulation(
            self._formulation_Z, self._product_data, role='formulation_Z',
        )

        if 'prices' in x_columns:
            raise ValidationError(
                "Expected formulation_X to exclude the 'prices' column "
                "(price is the endogenous regressor and is added by the "
                "estimator automatically). "
                "Received formulation_X containing 'prices'. "
                "Fix: drop 'prices' from formulation_X."
            )
        if 'prices' in z_excl_columns:
            raise ValidationError(
                "Expected formulation_Z to exclude the 'prices' column "
                "(price is the endogenous regressor, not an instrument for "
                "itself). "
                "Received formulation_Z containing 'prices'. "
                "Fix: drop 'prices' from formulation_Z."
            )

        K_x = X.shape[1]
        if Z_excl.shape[1] < 1:
            raise ValidationError(
                "Expected formulation_Z to provide at least one excluded "
                "instrument for price (the order condition for 2SLS). "
                "Received an empty Z matrix. "
                "Fix: pass formulation_Z=Formulation('0 + <instrument>')."
            )

        # D = [X | prices], W = [X | Z_excl]. Order matters for theta extraction:
        # the price coefficient sits at index K_x.
        D = np.column_stack([X, prices]) if K_x else prices.reshape(-1, 1)
        W = np.column_stack([X, Z_excl]) if K_x else Z_excl

        theta, omega = two_stage_least_squares(
            delta=delta, D=D, W=W, weight=self._W_demand_user,
        )

        beta = theta[:K_x] if K_x else np.zeros(0)
        alpha = float(theta[-1])

        if alpha >= 0:
            warnings.warn(
                f"LogitEstimator returned alpha={alpha:.4f} >= 0; "
                "this implies upward-sloping demand. "
                "Possible causes: weak instruments, mis-signed instruments, "
                "or model misspecification. "
                "Fix: check that excluded instruments shift price in the "
                "expected direction and have first-stage F > 10.",
                stacklevel=2,
            )

        xi = delta - D @ theta

        # demand_instrument_columns = excluded IVs first, then x_columns
        # (matches the convention from ``Problem._backend_from_demand_results``).
        demand_instrument_columns = list(z_excl_columns) + list(x_columns)

        self.alpha = alpha
        self.beta = beta
        self.xi = xi
        self.x_columns = list(x_columns)
        self.demand_instrument_columns = demand_instrument_columns
        # Reorder the stored omega so it lines up with [Z_excl, X] rather than
        # [X, Z_excl] (the order used in the 2SLS computation).
        self.W_demand = _reorder_weight_matrix(
            omega, source_order=list(x_columns) + list(z_excl_columns),
            target_order=demand_instrument_columns,
        )

        logger.info(
            "LogitEstimator: estimated alpha=%.6f on N=%d observations, "
            "T=%d markets, with K_x=%d exogenous regressors and K_z=%d "
            "excluded instruments.",
            alpha, prices.shape[0], len(np.unique(market_ids)),
            K_x, Z_excl.shape[1],
        )

        return {
            'alpha': alpha,
            'beta': beta,
            'rho': [],
            'x_columns': self.x_columns,
            'demand_instrument_columns': self.demand_instrument_columns,
            'W_demand': self.W_demand,
        }


def _reorder_weight_matrix(
        omega: Array, source_order: List[str], target_order: List[str],
) -> Array:
    """Permute rows + columns of ``omega`` so it matches ``target_order``.

    The 2SLS routine builds ``W = [X, Z_excl]``; the analytical backend
    expects ``Z_D = [Z_excl, X]``. The weight matrix corresponds to the
    moment ``W' xi = 0``, so a column permutation of ``W`` requires the
    same permutation on both axes of the weight matrix.
    """
    if list(source_order) == list(target_order):
        return np.ascontiguousarray(omega)
    perm = [source_order.index(name) for name in target_order]
    permuted = omega[np.ix_(perm, perm)]
    return np.ascontiguousarray(permuted)
