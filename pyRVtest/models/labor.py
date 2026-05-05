"""Labor-side conduct models: Monopsony, BertrandWages, CournotEmployment, NashBargaining.

Sign conventions flip relative to product-side models:
labor supply is upward-sloping in wages (``ds/dw > 0`` for
employment-on-wage), so the first-order conditions pick up markdowns
(``w - MRP``) rather than markups (``p - MC``). The math is a
sign-flipped mirror of ``pyRVtest/models/standard.py``.

Classes
-------

- :class:`Monopsony` — single-firm wage-setter (analogue of ``Monopoly``).
- :class:`BertrandWages` — wage-setting Bertrand (analogue of ``Bertrand``).
- :class:`CournotEmployment` — employment-setting Cournot
  (analogue of ``Cournot``).
- :class:`NashBargaining` — surplus-split bargaining over wages; full
  implementation deferred to v0.5 when labor data is available.

Each class accepts an ``ownership`` init kwarg (default ``'firm_ids'``)
matching the Bertrand pattern. ``NashBargaining`` additionally takes
``outside_option`` (required) and ``bargaining_weight`` (optional).

These classes are only valid when a ``Problem`` is constructed with
``market_side='labor'``; using them on a product-side problem raises
:class:`pyRVtest.exceptions.ValidationError` at ``Problem.__init__``.

The ``LaborSupplyBackend`` skeleton in
``pyRVtest/backends/labor/nested_logit_labor.py`` is intentionally a
stub in v0.4: plumbing exists and the protocol is honored so that v0.5
can populate the analytical Jacobian/Hessian without architectural
changes. See ``.claude/plans/v0.4-refactor.md`` §4.5.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Union

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypeAlias

from .base import ConductModel


__all__ = ['Monopsony', 'BertrandWages', 'CournotEmployment', 'NashBargaining']


_NDArray: TypeAlias = NDArray[Any]


def _as_column(s: _NDArray) -> _NDArray:
    """Ensure a share/markdown vector is shape ``(J, 1)`` for matrix ops."""
    a = np.asarray(s)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    return a


class Monopsony(ConductModel):
    r"""Single-firm wage-setter under upward-sloping labor supply.

    Sign-flipped analogue of :class:`Monopoly`. Markdown formula:
    ``markdown = D^{-1} s`` (no leading minus, because labor supply
    slopes up). The implicit derivative is
    ``d(markdown)/d(theta) = D^{-1} (dD/d(theta)) @ markdown``.

    Examples
    --------
    >>> import numpy as np
    >>> from pyRVtest.models.labor import Monopsony
    >>> model = Monopsony(ownership='firm_ids')
    >>> model._model_name
    'monopsony'
    >>> # Upward-sloping supply: diagonal entries of D are POSITIVE.
    >>> D = np.array([[2.0, -0.5], [-0.5, 2.0]])
    >>> s = np.array([0.3, 0.3])
    >>> model._compute_markup(np.eye(2), D, s).round(4)
    array([[0.2],
           [0.2]])
    """

    _model_name = 'monopsony'

    def __init__(
            self,
            ownership: Optional[str] = 'firm_ids',
            kappa_specification: Optional[Union[str, Callable[[Any, Any], float]]] = None,
            user_supplied_markups: Optional[str] = None,
            unit_tax: Optional[str] = None,
            advalorem_tax: Optional[str] = None,
            advalorem_payer: Optional[str] = None,
            cost_scaling: Optional[str] = None,
            vertical_integration: Optional[str] = None,
            mix_flag: Optional[str] = None,
    ) -> None:
        super().__init__(
            ownership=ownership,
            kappa_specification=kappa_specification,
            user_supplied_markups=user_supplied_markups,
            unit_tax=unit_tax,
            advalorem_tax=advalorem_tax,
            advalorem_payer=advalorem_payer,
            cost_scaling=cost_scaling,
            vertical_integration=vertical_integration,
            mix_flag=mix_flag,
        )

    def _compute_markup(self, O: _NDArray, D: _NDArray, s: _NDArray) -> _NDArray:
        s_col = _as_column(s)
        return np.linalg.solve(D, s_col)

    def _markup_derivative(
            self, O: _NDArray, D: _NDArray, dD: _NDArray,
            s: _NDArray, mu: _NDArray,
    ) -> _NDArray:
        dA = dD.T
        return np.linalg.solve(D.T, dA @ mu)


class BertrandWages(ConductModel):
    r"""Wage-setting Bertrand among labor competitors.

    Sign-flipped analogue of :class:`Bertrand`. Markdown formula:
    ``markdown = (O * D')^{-1} s``. Implicit derivative:
    ``d(markdown)/d(theta) = (O * D')^{-1} (O * dD'/d(theta)) @ markdown``.

    Examples
    --------
    >>> import numpy as np
    >>> from pyRVtest.models.labor import BertrandWages
    >>> model = BertrandWages(ownership='firm_ids')
    >>> model._model_name
    'bertrand_wages'
    >>> # Two single-firm employers, simple upward-sloping supply matrix.
    >>> O = np.eye(2)
    >>> D = np.array([[2.0, -0.5], [-0.5, 2.0]])
    >>> s = np.array([0.3, 0.3])
    >>> model._compute_markup(O, D, s).round(4)
    array([[0.15],
           [0.15]])
    """

    _model_name = 'bertrand_wages'

    def _compute_markup(self, O: _NDArray, D: _NDArray, s: _NDArray) -> _NDArray:
        s_col = _as_column(s)
        return np.linalg.solve(O * D.T, s_col)

    def _markup_derivative(
            self, O: _NDArray, D: _NDArray, dD: _NDArray,
            s: _NDArray, mu: _NDArray,
    ) -> _NDArray:
        A = O * D.T
        dA = O * dD.T
        return np.linalg.solve(A, dA @ mu)


class CournotEmployment(ConductModel):
    r"""Employment-setting Cournot among labor competitors.

    Sign-flipped analogue of :class:`Cournot`. Markdown formula:
    ``markdown = (O * D^{-1}) @ s``. Implicit derivative uses
    ``d(D^{-1})/d(theta) = -D^{-1} (dD/d(theta)) D^{-1}``.

    Examples
    --------
    >>> import numpy as np
    >>> from pyRVtest.models.labor import CournotEmployment
    >>> model = CournotEmployment(ownership='firm_ids')
    >>> model._model_name
    'cournot_employment'
    >>> O = np.eye(2)
    >>> D = np.array([[2.0, -0.5], [-0.5, 2.0]])
    >>> s = np.array([0.3, 0.3])
    >>> model._compute_markup(O, D, s).round(4)
    array([[0.16],
           [0.16]])
    """

    _model_name = 'cournot_employment'

    def _compute_markup(self, O: _NDArray, D: _NDArray, s: _NDArray) -> _NDArray:
        s_col = _as_column(s)
        return (O * np.linalg.inv(D)) @ s_col

    def _markup_derivative(
            self, O: _NDArray, D: _NDArray, dD: _NDArray,
            s: _NDArray, mu: _NDArray,
    ) -> _NDArray:
        D_inv = np.linalg.inv(D)
        dD_inv = -D_inv @ dD @ D_inv
        result: _NDArray = (O * dD_inv) @ s
        return result


class NashBargaining(ConductModel):
    r"""Nash bargaining over surplus between firms and workers.

    Requires an outside-option column (worker's reservation wage) and
    optionally a bargaining-weight column (worker share of surplus).

    Implementation is deliberately deferred to v0.5 when labor data is
    available. The class shape, required-column discovery, and
    registration under :class:`pyRVtest.ModelFormulation` land in v0.4
    step 14 so downstream plumbing (``Problem(market_side='labor')``,
    ``Models`` recarray, label branching) is exercised; the markdown
    math is pinned to the specific labor-paper formula once the
    pre-registered DGP ships.

    Parameters
    ----------
    outside_option
        Column name in product_data giving the worker's outside option
        (reservation wage) per observation.
    bargaining_weight
        Optional column name giving the worker's share of the bilateral
        surplus, :math:`\gamma \in [0, 1]`. If ``None``, the formula
        falls back to symmetric Nash (``gamma = 0.5``) at solve time.
    ownership
        Firm-id column for ownership-matrix construction. Defaults to
        ``'firm_ids'`` to match the rest of the labor-side model family.

    Raises
    ------
    NotImplementedError
        ``_compute_markup`` and ``_markup_derivative`` raise with a
        pointer to v0.5. See module docstring.
    """

    _model_name = 'nash_bargaining'

    def __init__(
            self,
            outside_option: str,
            bargaining_weight: Optional[str] = None,
            ownership: Optional[str] = 'firm_ids',
            kappa_specification: Optional[Union[str, Callable[[Any, Any], float]]] = None,
            user_supplied_markups: Optional[str] = None,
            unit_tax: Optional[str] = None,
            advalorem_tax: Optional[str] = None,
            advalorem_payer: Optional[str] = None,
            cost_scaling: Optional[str] = None,
            vertical_integration: Optional[str] = None,
            mix_flag: Optional[str] = None,
    ) -> None:
        if not isinstance(outside_option, str) or not outside_option:
            raise TypeError(
                f"Expected outside_option to be a non-empty column-name string. "
                f"Received {outside_option!r}. "
                f"Fix: pass the column name of the worker's reservation wage "
                f"(e.g., NashBargaining(outside_option='reservation_wage'))."
            )
        if bargaining_weight is not None and not isinstance(bargaining_weight, str):
            raise TypeError(
                f"Expected bargaining_weight to be a column-name string or None. "
                f"Received {type(bargaining_weight).__name__}. "
                f"Fix: pass the column name of the bargaining weight or omit "
                f"bargaining_weight for symmetric Nash bargaining."
            )
        super().__init__(
            ownership=ownership,
            kappa_specification=kappa_specification,
            user_supplied_markups=user_supplied_markups,
            unit_tax=unit_tax,
            advalorem_tax=advalorem_tax,
            advalorem_payer=advalorem_payer,
            cost_scaling=cost_scaling,
            vertical_integration=vertical_integration,
            mix_flag=mix_flag,
        )
        self.outside_option = outside_option
        self.bargaining_weight = bargaining_weight

    def _compute_markup(self, O: _NDArray, D: _NDArray, s: _NDArray) -> _NDArray:
        raise NotImplementedError(
            "NashBargaining markdowns are not implemented in v0.4. "
            "Full implementation deferred to v0.5 when labor data is "
            "available. See .claude/plans/v0.4-refactor.md §4.5 and the "
            "models/labor.py module docstring for the planned formula. "
            "Fix: use Monopsony, BertrandWages, or CournotEmployment for "
            "labor-side testing in v0.4, or supply a CustomConductModel "
            "with the bargaining formula you need."
        )

    def _markup_derivative(
            self, O: _NDArray, D: _NDArray, dD: _NDArray,
            s: _NDArray, mu: _NDArray,
    ) -> _NDArray:
        raise NotImplementedError(
            "NashBargaining markdown derivatives are not implemented in v0.4. "
            "Full implementation deferred to v0.5 when labor data is "
            "available. See .claude/plans/v0.4-refactor.md §4.5. "
            "Fix: use Monopsony, BertrandWages, or CournotEmployment for "
            "labor-side testing in v0.4."
        )
