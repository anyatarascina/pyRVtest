"""Standard conduct models: Bertrand, Cournot, Monopoly, PerfectCompetition.

v0.4 step 5a. Each class implements ``_compute_markup`` and
``_markup_derivative`` as a thin wrapper around the math that previously
lived in ``pyRVtest/markups.py::evaluate_first_order_conditions`` and
``pyRVtest/solve/demand_adjustment.py::_analytical_markup_derivative``.

The math is unchanged; these classes reorganize it into polymorphic
dispatch. Step 5b wires ``Problem`` to consume these instances
directly; step 5c keeps ``ModelFormulation(model_downstream='bertrand',
...)`` working as a deprecation alias that constructs the right class.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .base import ConductModel


__all__ = ['Bertrand', 'Cournot', 'Monopoly', 'PerfectCompetition']


_NDArray = NDArray[Any]


def _as_column(s: _NDArray) -> _NDArray:
    """Ensure a share/markup vector is shape ``(J, 1)`` for matrix ops."""
    a = np.asarray(s)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    return a


class Bertrand(ConductModel):
    r"""Price-setting (Bertrand-Nash) conduct.

    First-order condition: ``(O * D') @ markup + s = 0``, giving
    ``markup = -(O * D')^{-1} s``. Implicit differentiation yields
    ``d(markup)/d(theta) = -(O * D')^{-1} (O * dD'/d(theta)) @ markup``.
    """

    _model_name = 'bertrand'

    def _compute_markup(self, O: _NDArray, D: _NDArray, s: _NDArray) -> _NDArray:
        s_col = _as_column(s)
        return -np.linalg.solve(O * D.T, s_col)

    def _markup_derivative(
            self, O: _NDArray, D: _NDArray, dD: _NDArray,
            s: _NDArray, mu: _NDArray,
    ) -> _NDArray:
        A = O * D.T
        dA = O * dD.T
        return -np.linalg.solve(A, dA @ mu)


class Cournot(ConductModel):
    r"""Quantity-setting (Cournot-Nash) conduct.

    Markup formula: ``markup = -(O * D^{-1}) @ s``. Using
    ``d(D^{-1})/d(theta) = -D^{-1} (dD/d(theta)) D^{-1}``, the implicit
    derivative is ``d(markup)/d(theta) = -(O * dD^{-1}/d(theta)) @ s``.
    """

    _model_name = 'cournot'

    def _compute_markup(self, O: _NDArray, D: _NDArray, s: _NDArray) -> _NDArray:
        s_col = _as_column(s)
        return -(O * np.linalg.inv(D)) @ s_col

    def _markup_derivative(
            self, O: _NDArray, D: _NDArray, dD: _NDArray,
            s: _NDArray, mu: _NDArray,
    ) -> _NDArray:
        D_inv = np.linalg.inv(D)
        dD_inv = -D_inv @ dD @ D_inv
        result: _NDArray = -(O * dD_inv) @ s
        return result


class Monopoly(ConductModel):
    r"""Full-collusion / single-firm conduct.

    Markup formula: ``markup = -D'^{-1} @ s`` (all ownership effectively
    unity). Implicit derivative: ``d(markup)/d(theta) = -D'^{-1}
    (dD'/d(theta)) @ markup``.
    """

    _model_name = 'monopoly'

    def _compute_markup(self, O: _NDArray, D: _NDArray, s: _NDArray) -> _NDArray:
        s_col = _as_column(s)
        return -np.linalg.solve(D, s_col)

    def _markup_derivative(
            self, O: _NDArray, D: _NDArray, dD: _NDArray,
            s: _NDArray, mu: _NDArray,
    ) -> _NDArray:
        dA = dD.T
        return -np.linalg.solve(D.T, dA @ mu)


class PerfectCompetition(ConductModel):
    """Marginal-cost pricing. Markups and their gradients are zero."""

    _model_name = 'perfect_competition'

    def _compute_markup(self, O: _NDArray, D: _NDArray, s: _NDArray) -> _NDArray:
        J = np.asarray(s).shape[0]
        return np.zeros((J, 1))

    def _markup_derivative(
            self, O: _NDArray, D: _NDArray, dD: _NDArray,
            s: _NDArray, mu: _NDArray,
    ) -> _NDArray:
        J = np.asarray(s).shape[0]
        return np.zeros(J)
