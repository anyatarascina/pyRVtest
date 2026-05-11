"""Dearing et al. (2024) simple-markup conduct models.

Implements three closely related simple-markup conduct models from
Dearing, Magnolfi, Quint, Sullivan, and Waldfogel (2024), "Learning
Firm Conduct: Pass-Through as a Foundation for Instrument Relevance."

Models
------

``RuleOfThumb(phi)`` — Example 1, pp. 7-8. Price is a fixed multiple
    :math:`\\varphi \\geq 1` of marginal cost: :math:`p = \\varphi \\cdot
    mc`, which rearranges to markup :math:`\\Delta = (\\varphi - 1) / \\varphi
    \\cdot p`. Setting ``phi=2.0`` yields 50%-of-price markups;
    ``phi=1.0`` recovers marginal-cost pricing (perfect competition).

``ConstantMarkup(markup)`` — Example 7, pp. 23-24. A fixed per-product
    dollar markup :math:`\\Delta_{jt} = \\zeta_j` that does not vary
    across markets. Unlike the rule-of-thumb family, the markup does
    NOT come from the first-order condition — it is a model primitive
    supplied by the researcher as either a scalar (same dollar markup
    for every product) or a column name in ``product_data``
    (product-specific dollar markups).

Mechanism
---------

``RuleOfThumb`` uses the same ``constant_markup`` field in the
``Models`` recarray as ``ConstantMarkup``. ``Problem.__init__``
pre-computes ``(phi - 1) / phi * prices`` and stores it as the
per-row ``constant_markup`` array for that model.
:func:`~pyRVtest.markups.evaluate_first_order_conditions` then reads
the field and writes it directly as the markup vector, so
``markups = (phi - 1) / phi * p`` and ``mc = p / phi``.

``ConstantMarkup`` introduces an additive per-product dollar markup
that does not vary across markets. A branch
in :func:`~pyRVtest.markups.evaluate_first_order_conditions` reads the
per-row markup vector from the ``constant_markup`` field threaded
through the ``Models`` recarray, treating the model as if it had a
user-supplied markup without the column-name requirement.

References
----------

Dearing, Magnolfi, Quint, Sullivan, and Waldfogel (2024), "Learning
Firm Conduct: Pass-Through as a Foundation for Instrument Relevance."

"""

from __future__ import annotations

from typing import Any, Union

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypeAlias

from ..exceptions import ValidationError
from .base import ConductModel


__all__ = ['ConstantMarkup', 'RuleOfThumb']


_NDArray: TypeAlias = NDArray[Any]


class RuleOfThumb(ConductModel):
    r"""Rule-of-thumb markup: :math:`p = \varphi \cdot mc`.

    Dearing, Magnolfi, Quint, Sullivan, and Waldfogel (2024), Example 1
    (pp. 7-8). Firms set price as a fixed multiple :math:`\varphi \geq 1`
    of marginal cost, which rearranges to the implied markup
    :math:`\Delta = (\varphi - 1) / \varphi \cdot p`. Equivalently, the
    implied marginal cost is :math:`mc = p / \varphi`.

    ``Problem.__init__`` pre-computes ``(phi - 1) / phi * prices`` and
    stores it as the per-row ``constant_markup`` entry. The
    :func:`~pyRVtest.markups.evaluate_first_order_conditions` dispatch
    then writes it directly as the markup vector.

    Parameters
    ----------
    phi : float
        Markup multiplier, :math:`\varphi \geq 1`. ``phi = 1`` degenerates to
        marginal-cost pricing (zero markup); ``phi = 2`` yields
        50%-of-price markups.

    Other Parameters
    ----------------
    Accepts all other :class:`~pyRVtest.models.base.ConductModel`
    keyword arguments EXCEPT ``cost_scaling``, which is incompatible
    with ``RuleOfThumb``. Passing both raises
    :class:`~pyRVtest.exceptions.ValidationError`.

    Examples
    --------
    >>> from pyRVtest import RuleOfThumb
    >>> model = RuleOfThumb(phi=2.0)
    >>> model.phi
    2.0
    >>> model._model_name
    'constant_markup'
    """

    _model_name = 'constant_markup'

    def __init__(self, phi: float, **kwargs: Any) -> None:
        # Reject bool explicitly before isinstance(phi, (int, float)) accepts it.
        if isinstance(phi, bool) or not isinstance(phi, (int, float)):
            raise ValidationError(
                f"Expected phi to be a numeric scalar (float or int). "
                f"Received {type(phi).__name__} ({phi!r}). "
                f"Fix: pass a scalar, e.g. RuleOfThumb(phi=2.0)."
            )
        if not np.isfinite(phi) or phi < 1.0:
            raise ValidationError(
                f"Expected phi >= 1.0 (Dearing Example 1 requires the markup "
                f"multiplier to be at least one, so the implied marginal cost "
                f"p / phi does not exceed the observed price). "
                f"Received phi = {phi!r}. "
                f"Fix: pass phi >= 1.0 (phi=1 degenerates to marginal-cost "
                f"pricing; phi=2 yields 50%-of-price markups)."
            )
        if 'cost_scaling' in kwargs:
            raise ValidationError(
                f"Expected RuleOfThumb not to receive a cost_scaling argument "
                f"(the markup (phi-1)/phi * p is computed directly from prices "
                f"and does not use the cost_scaling mechanism). "
                f"Received cost_scaling={kwargs['cost_scaling']!r} together "
                f"with phi={phi!r}. "
                f"Fix: drop the cost_scaling argument, or switch to the "
                f"lower-level PerfectCompetition(cost_scaling=...) API."
            )
        super().__init__(**kwargs)
        self.phi: float = float(phi)

    def _compute_markup(self, O: _NDArray, D: _NDArray, s: _NDArray) -> _NDArray:
        """Not callable on ``RuleOfThumb``.

        The markup ``(phi-1)/phi * p`` requires observed prices, which are
        not in the ``(O, D, s)`` triple. ``Problem.__init__`` pre-computes
        the per-row vector and stores it in the ``constant_markup`` field;
        the live dispatch path reads it via ``_model_name = 'constant_markup'``.
        """
        raise NotImplementedError(
            f"RuleOfThumb._compute_markup is not callable directly — the "
            f"markup (phi-1)/phi * p requires observed prices, which are not "
            f"part of the (ownership, response_matrix, shares) triple. "
            f"Fix: the Problem pipeline pre-computes (phi-1)/phi * prices and "
            f"stores it in the 'constant_markup' field; call Problem.solve(...) "
            f"rather than invoking this hook directly."
        )

    def _markup_derivative(
            self, O: _NDArray, D: _NDArray, dD: _NDArray,
            s: _NDArray, mu: _NDArray,
    ) -> _NDArray:
        """Zero gradient (see :meth:`_compute_markup`)."""
        J = int(np.asarray(s).shape[0])
        return np.zeros(J)

    def __repr__(self) -> str:
        return f'RuleOfThumb(phi={self.phi!r})'


class ConstantMarkup(ConductModel):
    r"""Fixed per-product dollar markup: :math:`\Delta_{jt} = \zeta_j`.

    Dearing et al. (2024), Example 7, pp. 23-24. Each product carries a
    fixed dollar markup that is the same across markets. Unlike the
    rule-of-thumb family, the markup does NOT come from a first-order
    condition — it is a model primitive supplied by the researcher.

    Parameters
    ----------
    markup : float, int, or str
        Either a numeric scalar (same dollar markup for every product)
        or a column name in ``product_data`` holding per-product dollar
        markups. Scalars are broadcast to every row.

    Other Parameters
    ----------------
    Accepts all other :class:`~pyRVtest.models.base.ConductModel`
    keyword arguments. ``user_supplied_markups`` is not allowed together
    with ``markup`` (they are two names for the same mechanism; pick
    one).

    Notes
    -----
    The markup is threaded through the ``Models`` recarray via a new
    ``constant_markup`` field (step 12 plumbing). The live dispatch path
    in :func:`~pyRVtest.markups.evaluate_first_order_conditions`
    recognises ``model_type = 'constant_markup'`` and writes the
    per-row vector directly; the class-based
    :meth:`ConductModel._compute_markup` hook raises
    :class:`NotImplementedError` because the markup is not a function of
    the ``(O, D, s)`` triple.

    Examples
    --------
    >>> from pyRVtest import ConstantMarkup
    >>> m_scalar = ConstantMarkup(markup=0.5)
    >>> m_scalar.markup
    0.5
    >>> m_scalar._model_name
    'constant_markup'
    >>> m_col = ConstantMarkup(markup='eta_col')
    >>> m_col.markup
    'eta_col'
    """

    _model_name = 'constant_markup'

    def __init__(
            self,
            markup: Union[float, int, str],
            **kwargs: Any,
    ) -> None:
        if isinstance(markup, bool) or not isinstance(markup, (int, float, str)):
            raise ValidationError(
                f"Expected markup to be a numeric scalar or a column-name string. "
                f"Received {type(markup).__name__} ({markup!r}). "
                f"Fix: pass a scalar like ConstantMarkup(markup=0.5) or a "
                f"column name like ConstantMarkup(markup='eta_col')."
            )
        if isinstance(markup, str) and not markup:
            raise ValidationError(
                "Expected markup to be a non-empty column-name string. "
                "Received markup=''. "
                "Fix: pass a non-empty column name, or switch to a scalar."
            )
        if isinstance(markup, (int, float)) and not np.isfinite(markup):
            raise ValidationError(
                f"Expected markup scalar to be finite. "
                f"Received markup = {markup!r}. "
                f"Fix: pass a finite numeric scalar."
            )
        if kwargs.get('user_supplied_markups') is not None:
            raise ValidationError(
                f"Expected ConstantMarkup to own the markup specification "
                f"(either as a scalar or as a column name via `markup=...`). "
                f"Received both markup={markup!r} and "
                f"user_supplied_markups={kwargs['user_supplied_markups']!r}. "
                f"Fix: drop user_supplied_markups — ConstantMarkup IS a "
                f"user-supplied-markup model under a dedicated class name."
            )
        super().__init__(**kwargs)
        self.markup: Union[float, int, str] = markup

    def _compute_markup(self, O: _NDArray, D: _NDArray, s: _NDArray) -> _NDArray:
        """Not callable on ``ConstantMarkup``.

        The markup comes from the instance's ``markup`` attribute (a
        scalar or a column pulled from ``product_data`` at
        ``Problem.__init__``), not from the ``(O, D, s)`` triple. The
        live path reads it via the ``'constant_markup'`` branch in
        :func:`~pyRVtest.markups.evaluate_first_order_conditions`.
        """
        raise NotImplementedError(
            f"ConstantMarkup._compute_markup is not callable directly — the "
            f"markup is a model primitive ({self.markup!r}), not a function of "
            f"(ownership, response_matrix, shares). "
            f"Fix: the Problem pipeline routes ConstantMarkup through the "
            f"'constant_markup' dispatch branch; call Problem.solve(...) "
            f"rather than invoking this hook directly."
        )

    def _markup_derivative(
            self, O: _NDArray, D: _NDArray, dD: _NDArray,
            s: _NDArray, mu: _NDArray,
    ) -> _NDArray:
        """Zero gradient: a constant markup does not depend on demand parameters."""
        J = int(np.asarray(s).shape[0])
        return np.zeros(J)

    def __repr__(self) -> str:
        return f'ConstantMarkup(markup={self.markup!r})'
