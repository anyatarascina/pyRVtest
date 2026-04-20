"""Mixed Cournot-Bertrand: Morrow-Skerlos (2011) Schur-complement formulation.

v0.4 step 5a. Extracts the math previously in
``pyRVtest/markups.py::_compute_mix_cournot_bertrand_markups`` and the
corresponding branch of ``_analytical_markup_derivative`` into a
polymorphic class.

User code specifies which products are Bertrand vs Cournot via a
``mix_flag`` column (boolean: True = Bertrand, False = Cournot). Within
a market the Bertrand products have their FOC modified by the Schur
complement to account for feedback from Cournot quantities.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypeAlias

from .base import ConductModel


__all__ = ['MixCournotBertrand']


_NDArray: TypeAlias = NDArray[Any]


class MixCournotBertrand(ConductModel):
    r"""Mixed-strategy oligopoly: a subset of products play Bertrand, the rest Cournot.

    Within each market, let :math:`B` index Bertrand products and :math:`C`
    index Cournot products. Cournot products use the standard quantity-
    setting FOC; Bertrand products use the price-setting FOC adjusted by
    the Schur complement ``D_BC @ D_CC^{-1} @ D_CB`` to account for the
    Cournot players' quantity response to Bertrand price choices. Formula
    from Morrow and Skerlos (2011).

    Parameters
    ----------
    mix_flag : str, required
        Column name for the per-product boolean indicating Bertrand (True)
        vs Cournot (False). Required; enforced in ``__init__``.
    ownership, kappa_specification, user_supplied_markups, taxes,
    cost_scaling, vertical_integration : see ``ConductModel``.

    Examples
    --------
    >>> from pyRVtest import MixCournotBertrand
    >>> model = MixCournotBertrand(mix_flag='is_bertrand', ownership='firm_ids')
    >>> model.mix_flag
    'is_bertrand'
    >>> model._model_name
    'mix_cournot_bertrand'
    >>> MixCournotBertrand(mix_flag=None)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    TypeError: Expected mix_flag to identify which products play Bertrand ...
    """

    _model_name = 'mix_cournot_bertrand'

    def __init__(
            self,
            mix_flag: Optional[str],
            **kwargs: Any,
    ) -> None:
        if mix_flag is None:
            raise TypeError(
                "Expected mix_flag to identify which products play Bertrand "
                "(True) vs Cournot (False) in each market; mix_flag is required "
                "for MixCournotBertrand. "
                "Received mix_flag=None. "
                "Fix: pass mix_flag='<column_name>' naming the per-product "
                "boolean indicator in product_data."
            )
        super().__init__(mix_flag=mix_flag, **kwargs)

    def _compute_markup(
            self, O: _NDArray, D: _NDArray, s: _NDArray,
    ) -> _NDArray:
        raise NotImplementedError(
            "pyRVtest internal error: expected the pipeline to call "
            "MixCournotBertrand._compute_markup_with_flag(O, D, s, mix_flag_t), "
            "which carries the per-market Bertrand/Cournot slice. "
            "Received a direct _compute_markup call without mix_flag. "
            "Fix: route this call through _compute_markup_with_flag (Problem "
            "supplies the flag during step 5b wiring)."
        )

    def _compute_markup_with_flag(
            self, O: _NDArray, D: _NDArray, s: _NDArray,
            mix_flag_t: _NDArray,
    ) -> _NDArray:
        """Compute markups using an already-sliced mix_flag for this market."""
        b = mix_flag_t.astype(bool)
        c = ~b
        s_arr = np.asarray(s).flatten()
        if not (b.any() and c.any()):
            # Fall back: if all products are one side, the market is pure
            # Bertrand or pure Cournot. Match the existing inline behavior
            # which zeros out markups in this degenerate configuration.
            return np.zeros((len(s_arr), 1))
        shares_B, shares_C = s_arr[b], s_arr[c]
        O_BB = O[np.ix_(b, b)]
        O_CC = O[np.ix_(c, c)]
        D_BB = D[np.ix_(b, b)]
        D_BC = D[np.ix_(b, c)]
        D_CB = D[np.ix_(c, b)]
        D_CC = D[np.ix_(c, c)]

        D_CC_inv = np.linalg.inv(D_CC)
        mkups_C = -(O_CC * D_CC_inv) @ shares_C
        mkups_B = np.linalg.solve(
            O_BB * (D_BC @ D_CC_inv @ D_CB + D_BB), -shares_B,
        )
        mkups = np.zeros((len(b), 1))
        mkups[b, 0] = mkups_B.flatten()
        mkups[c, 0] = mkups_C.flatten()
        return mkups

    def _markup_derivative(
            self, O: _NDArray, D: _NDArray, dD: _NDArray,
            s: _NDArray, mu: _NDArray,
    ) -> _NDArray:
        raise NotImplementedError(
            "pyRVtest internal error: expected the pipeline to call "
            "MixCournotBertrand._markup_derivative_with_flag (which carries "
            "the per-market mix_flag slice). "
            "Received a direct _markup_derivative call without mix_flag. "
            "Fix: route this call through _markup_derivative_with_flag."
        )

    def _markup_derivative_with_flag(
            self, O: _NDArray, D: _NDArray, dD: _NDArray,
            s: _NDArray, mu: _NDArray, mix_flag_t: _NDArray,
    ) -> _NDArray:
        """Markup derivative w.r.t. one theta, using pre-sliced mix_flag."""
        b = mix_flag_t.astype(bool)
        c = ~b
        J_t = len(b)
        if not (b.any() and c.any()):
            return np.zeros(J_t)
        s_arr = np.asarray(s).flatten()
        mu_arr = np.asarray(mu).flatten()
        D_BB = D[np.ix_(b, b)]
        D_BC = D[np.ix_(b, c)]
        D_CB = D[np.ix_(c, b)]
        D_CC = D[np.ix_(c, c)]
        D_CC_inv = np.linalg.inv(D_CC)
        O_BB = O[np.ix_(b, b)]
        O_CC = O[np.ix_(c, c)]
        dD_BB = dD[np.ix_(b, b)]
        dD_BC = dD[np.ix_(b, c)]
        dD_CB = dD[np.ix_(c, b)]
        dD_CC = dD[np.ix_(c, c)]

        # Cournot block
        dD_CC_inv = -D_CC_inv @ dD_CC @ D_CC_inv
        d_mu_C = -(O_CC * dD_CC_inv) @ s_arr[c]

        # Bertrand block via Schur complement
        Schur = D_BC @ D_CC_inv @ D_CB + D_BB
        dSchur = (
            dD_BC @ D_CC_inv @ D_CB
            + D_BC @ dD_CC_inv @ D_CB
            + D_BC @ D_CC_inv @ dD_CB
            + dD_BB
        )
        A_B = O_BB * Schur
        dA_B = O_BB * dSchur
        d_mu_B = -np.linalg.solve(A_B, dA_B @ mu_arr[b])

        d_mu = np.zeros(J_t)
        d_mu[b] = d_mu_B.flatten()
        d_mu[c] = d_mu_C.flatten()
        return d_mu
