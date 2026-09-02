"""Mixed Cournot-Bertrand: Feenstra-Levinsohn (1995) Nash-in-(p_B, q_C) formulation.

User code specifies which products are Bertrand vs Cournot via a
``mix_flag`` column (boolean: ``True`` = Bertrand, ``False`` = Cournot).
Within a market the Bertrand products face the residual demand that
obtains when Cournot quantities are held fixed, whose slope is the Schur
complement ``D_BB - D_BC D_CC^{-1} D_CB`` of the demand Jacobian.
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
    index Cournot products, and partition the demand Jacobian
    :math:`D = \partial s / \partial p` and the ownership matrix
    :math:`\Omega` accordingly. The equilibrium is Nash in
    :math:`(p_B, q_C)`: Bertrand firms choose prices taking rival prices and
    Cournot quantities as given, Cournot firms choose quantities taking rival
    quantities and Bertrand prices as given (Feenstra and Levinsohn, 1995,
    Proposition 3). Cournot products use the standard quantity-setting FOC
    on the Cournot block,

    .. math:: \Delta_C = -(\Omega_{CC} \odot D_{CC}^{-1}) s_C .

    When a Bertrand firm moves :math:`p_B`, the Cournot firms' prices adjust
    so that their quantities stay fixed, :math:`dp_C = -D_{CC}^{-1} D_{CB}\,dp_B`,
    so the Bertrand firm's residual-demand slope is the Schur complement
    :math:`S = D_{BB} - D_{BC} D_{CC}^{-1} D_{CB}` and

    .. math:: \Delta_B = -(\Omega_{BB} \odot S')^{-1} s_B .

    This is eq. (A22) of Feenstra and Levinsohn (1995) written on the
    Jacobian blocks. (Their Proposition 3(a) shows a *plus* sign because it
    is written on the cross-elasticity matrix :math:`E`, where
    :math:`D_{BC} \propto +E_{12}` but :math:`D_{CC} \propto -(I - E_{22})`;
    the sign is carried by :math:`(I - E_{22})^{-1}`.) Versions of pyRVtest
    before 0.4.0b9 used ``D_BB + D_BC D_CC^{-1} D_CB``, which understates the
    Bertrand players' markups (they must exceed pure-Bertrand markups).

    Parameters
    ----------
    mix_flag : str
        Column name for the per-product boolean indicating Bertrand (True)
        vs Cournot (False). Required; enforced in ``__init__``.

    Notes
    -----
    Other parameters (``ownership``, ``kappa_specification``,
    ``user_supplied_markups``, taxes, ``cost_scaling``,
    ``vertical_integration``) follow the :class:`ConductModel` base-class
    contract.

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
        # Residual-demand slope of the Bertrand players with Cournot
        # quantities held fixed (Schur complement of D w.r.t. the C block);
        # transposed to match the Bertrand convention ``O * D.T``.
        schur = D_BB - D_BC @ D_CC_inv @ D_CB
        mkups_B = np.linalg.solve(O_BB * schur.T, -shares_B)
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

        # Bertrand block via the Schur complement S = D_BB - D_BC D_CC^{-1} D_CB
        schur = D_BB - D_BC @ D_CC_inv @ D_CB
        d_schur = dD_BB - (
            dD_BC @ D_CC_inv @ D_CB +
            D_BC @ dD_CC_inv @ D_CB +
            D_BC @ D_CC_inv @ dD_CB
        )
        A_B = O_BB * schur.T
        dA_B = O_BB * d_schur.T
        d_mu_B = -np.linalg.solve(A_B, dA_B @ mu_arr[b])

        d_mu = np.zeros(J_t)
        d_mu[b] = d_mu_B.flatten()
        d_mu[c] = d_mu_C.flatten()
        return d_mu
