"""Dearing et al. (2026) simple-markup conduct models.

Implements three closely related simple-markup conduct models from
Dearing, Magnolfi, Quint, Sullivan, and Waldfogel (2026), "Learning
Firm Conduct: Pass-Through as a Foundation for Instrument Relevance."

Models
------

``RuleOfThumb(phi)`` — Example 1, pp. 7-8. Price is a fixed multiple
    :math:`\\varphi \\geq 1` of marginal cost: :math:`p = \\varphi \\cdot
    mc`, which rearranges to markup :math:`\\Delta = (\\varphi - 1) / \\varphi
    \\cdot p`. Implemented as an ergonomic wrapper over
    :class:`~pyRVtest.models.base.ConductModel.cost_scaling`. Setting
    ``phi=2.0`` yields 50%-of-price markups (the traditional "keystone"
    rule); ``phi=1.0`` recovers marginal-cost pricing (perfect
    competition).

``Keystone()`` — Example 1, p. 8. The ``phi=2`` special case named after
    Escudero (2018). A one-line subclass of :class:`RuleOfThumb`.

``ConstantMarkup(markup)`` — Example 7, pp. 23-24. A fixed per-product
    dollar markup :math:`\\Delta_{jt} = \\zeta_j` that does not vary
    across markets. Unlike the rule-of-thumb family, the markup does
    NOT come from the first-order condition — it is a model primitive
    supplied by the researcher as either a scalar (same dollar markup
    for every product) or a column name in ``product_data``
    (product-specific dollar markups).

Mechanism
---------

``RuleOfThumb`` and ``Keystone`` reuse the existing ``cost_scaling``
machinery. ``cost_scaling`` was previously a column name only; v0.4
step 12a extends it to accept a numeric scalar. The implementation
sets ``cost_scaling = phi - 1`` so that
``problem.py``'s effective-price / effective-markup post-processing
(``prices_effective = tax_adj * p / (1 + cost_scaling) - unit_tax``,
``markups_effective = tax_adj / (1 + cost_scaling) * markups``) yields
``mc = p / phi`` when the underlying markup is zero. Hence the
implementation inherits from :class:`~pyRVtest.PerfectCompetition` via
the ``_model_name = 'perfect_competition'`` dispatch — the
zero-underlying-markup plus post-processing cost division delivers the
Dearing math.

``ConstantMarkup`` introduces a new additive-markup mechanism. The
existing ``cost_scaling`` factor is multiplicative (:math:`mc = p /
(1 + \\lambda)`) and cannot express an additive markup. A new branch
in :func:`~pyRVtest.markups.evaluate_first_order_conditions` reads the
per-row markup vector from a new ``constant_markup`` field threaded
through the ``Models`` recarray, treating the model as if it had a
user-supplied markup without the column-name requirement.

References
----------

Dearing, Magnolfi, Quint, Sullivan, and Waldfogel (2026), "Learning
Firm Conduct: Pass-Through as a Foundation for Instrument Relevance."

Escudero (2018), cited in Example 1 for the term "Keystone".
"""

from __future__ import annotations

from typing import Any, Union

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypeAlias

from ..exceptions import ValidationError
from .base import ConductModel


__all__ = ['ConstantMarkup', 'Keystone', 'RuleOfThumb']


_NDArray: TypeAlias = NDArray[Any]


class RuleOfThumb(ConductModel):
    r"""Rule-of-thumb markup: :math:`p = \varphi \cdot mc`.

    Dearing, Magnolfi, Quint, Sullivan, and Waldfogel (2026), Example 1
    (pp. 7-8). Firms set price as a fixed multiple :math:`\varphi \geq 1`
    of marginal cost, which rearranges to the implied markup
    :math:`\Delta = (\varphi - 1) / \varphi \cdot p`. Equivalently, the
    implied marginal cost is :math:`mc = p / \varphi`.

    Implemented as an ergonomic wrapper over the existing ``cost_scaling``
    mechanism: ``RuleOfThumb(phi=2.0)`` internally sets
    ``cost_scaling = phi - 1 = 1.0`` and ``_model_name =
    'perfect_competition'``. The post-processing in ``problem.py``
    then delivers ``mc = p / (1 + 1) = p / 2``, which is the Dearing
    Example 1 formula for :math:`\varphi = 2`.

    Parameters
    ----------
    phi : float
        Markup multiplier, :math:`\varphi \geq 1`. ``phi = 1`` degenerates to
        marginal-cost pricing (perfect competition); ``phi = 2`` is the
        Keystone rule (50% markup over cost, equivalently 50% of price).

    Other Parameters
    ----------------
    Accepts all other :class:`~pyRVtest.models.base.ConductModel`
    keyword arguments EXCEPT ``cost_scaling``, which is set internally
    from ``phi``. Passing both ``phi`` and ``cost_scaling`` raises
    :class:`~pyRVtest.exceptions.ValidationError`.

    Examples
    --------
    >>> import numpy as np
    >>> from pyRVtest import RuleOfThumb
    >>> model = RuleOfThumb(phi=2.0)
    >>> model.phi
    2.0
    >>> model.cost_scaling
    1.0
    >>> model._model_name
    'perfect_competition'
    >>> # _compute_markup returns zero (the cost_scaling post-processing
    >>> # delivers the Dearing math downstream).
    >>> model._compute_markup(np.eye(2), np.eye(2), np.array([0.3, 0.3]))
    array([[0.],
           [0.]])
    """

    _model_name = 'perfect_competition'

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
                f"pricing; phi=2 is the Keystone rule)."
            )
        if 'cost_scaling' in kwargs:
            raise ValidationError(
                f"Expected RuleOfThumb to own the cost_scaling factor "
                f"(it is set internally from phi via cost_scaling = phi - 1). "
                f"Received an explicit cost_scaling={kwargs['cost_scaling']!r} "
                f"together with phi={phi!r}. "
                f"Fix: drop the cost_scaling argument, or switch to the "
                f"lower-level PerfectCompetition(cost_scaling=...) API."
            )
        super().__init__(cost_scaling=float(phi) - 1.0, **kwargs)
        self.phi: float = float(phi)

    def _compute_markup(self, O: _NDArray, D: _NDArray, s: _NDArray) -> _NDArray:
        """Return a zero markup vector (shape ``(J, 1)``).

        The Dearing math is delivered by the ``cost_scaling`` post-processing
        in ``problem.py``: the zero raw markup plus the ``1 / (1 + (phi - 1))
        = 1 / phi`` scaling produces ``mc = p / phi``. This override exists
        to satisfy the ``ConductModel._compute_markup`` contract (Liskov);
        it is also the value consumed by the live string-dispatch path
        through ``_model_name = 'perfect_competition'``.
        """
        J = int(np.asarray(s).shape[0])
        return np.zeros((J, 1))

    def _markup_derivative(
            self, O: _NDArray, D: _NDArray, dD: _NDArray,
            s: _NDArray, mu: _NDArray,
    ) -> _NDArray:
        """Zero gradient (see :meth:`_compute_markup`)."""
        J = int(np.asarray(s).shape[0])
        return np.zeros(J)

    def __repr__(self) -> str:
        return f'RuleOfThumb(phi={self.phi!r})'


class Keystone(RuleOfThumb):
    r"""Keystone rule: 50%-of-price markup.

    Dearing et al. (2026), Example 1, p. 8. The ``phi = 2`` special case
    of :class:`RuleOfThumb`, named after Escudero (2018). Price equals
    twice marginal cost, i.e. the markup equals half the price.

    Examples
    --------
    >>> from pyRVtest import Keystone
    >>> k = Keystone()
    >>> k.phi
    2.0
    >>> k.cost_scaling
    1.0
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(phi=2.0, **kwargs)

    def __repr__(self) -> str:
        return 'Keystone()'


class ConstantMarkup(ConductModel):
    r"""Fixed per-product dollar markup: :math:`\Delta_{jt} = \zeta_j`.

    Dearing et al. (2026), Example 7, pp. 23-24. Each product carries a
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
