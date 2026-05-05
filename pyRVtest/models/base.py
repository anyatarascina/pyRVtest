"""Base class for all conduct models.

`ConductModel` holds the shared configuration fields (ownership column
name, kappa specification, user-supplied markups, taxes, vertical
integration, cost scaling, mix flag) and provides two abstract math
hooks that subclasses implement:

  - ``_compute_markup(O, D, s)``: markup vector from ownership matrix,
    response matrix (dshares/dprices), and shares, all for one market.

  - ``_markup_derivative(O, D, dD, s, mu)``: gradient of the markup w.r.t.
    one theta parameter, via implicit differentiation of the model FOC.
    ``dD`` is ``d(D)/d(theta_k)`` for one particular k; the caller loops
    over k.

Split of concerns:

  - Math is in the subclasses (``Bertrand``, ``Cournot``, etc.).
  - Config is shared via this base class (inherited by all subclasses).
  - Non-vertical case: config attributes live on the bare subclass.
  - Vertical case: ``Vertical`` wrapper carries the shared config
    (vertical_integration, taxes); inner conduct instances carry only
    their own ``ownership`` / ``kappa_specification`` / ``mix_flag``.
    Validation that config is in the right place happens at
    ``Problem.__init__`` in step 5b.

Backward compat with ``ModelFormulation`` is preserved via the
deprecation alias in step 5c; users can migrate at their own pace.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Optional, Set, Union

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypeAlias


__all__ = ['ConductModel']


_NDArray: TypeAlias = NDArray[Any]


# once-per-session flags for the model-level tax deprecations.
# Keyed by the warning kind ('unit_tax', 'advalorem_tax') so a session with
# many Bertrand instances carrying a legacy unit_tax emits one warning
# rather than one per instance. Tests that exercise the warning reset this
# set before the call. Shared with pyRVtest/problem.py's conflict-warning
# path so a user who triggers BOTH the per-model deprecation (emitted here
# at construction) and the Problem-level-vs-model-level conflict
# (emitted at Problem.__init__) only sees the generic deprecation once.
_legacy_tax_deprecation_warned: Set[str] = set()


_MODEL_UNIT_TAX_DEPRECATION_MSG = (
    "Specifying unit_tax on an individual ConductModel / ModelFormulation / "
    "Vertical is deprecated; pass unit_tax='col' at the Problem level "
    "instead (the DGP defines the tax; models opt out for salience tests via "
    "unit_tax_salient=False). Model-level unit_tax continues to work through "
    "v0.6 and will be removed in v0.7. See docs/migrating_to_v0.4.rst for "
    "the migration recipe."
)

_MODEL_ADVALOREM_TAX_DEPRECATION_MSG = (
    "Specifying advalorem_tax / advalorem_payer on an individual ConductModel "
    "/ ModelFormulation / Vertical is deprecated; pass advalorem_tax='col' "
    "and advalorem_payer='firm'|'consumer' at the Problem level instead. "
    "Model-level advalorem_tax / advalorem_payer continue to work through "
    "v0.6 and will be removed in v0.7. See docs/migrating_to_v0.4.rst."
)


def _maybe_warn_per_model_tax_deprecation(field: str, stacklevel: int = 3) -> None:
    """Emit the v0.7-removal DeprecationWarning for per-model tax kwargs.

    Fires once per Python session per ``field`` value (``'unit_tax'`` or
    ``'advalorem_tax'``), using ``_legacy_tax_deprecation_warned`` as
    the guard. Called from ``ConductModel.__init__`` and
    ``Vertical.__init__`` at construction time so users see the warning
    immediately when they write ``pyRVtest.Bertrand(unit_tax='col')``.
    """
    if field in _legacy_tax_deprecation_warned:
        return
    if field == 'unit_tax':
        msg = _MODEL_UNIT_TAX_DEPRECATION_MSG
    elif field == 'advalorem_tax':
        msg = _MODEL_ADVALOREM_TAX_DEPRECATION_MSG
    else:
        return
    warnings.warn(msg, DeprecationWarning, stacklevel=stacklevel)
    _legacy_tax_deprecation_warned.add(field)


class ConductModel:
    """Abstract base for all conduct models.

    Parameters
    ----------
    ownership : str, optional
        Column name in product_data holding firm/ownership identifiers.
        For single-product firms this is typically ``'firm_ids'``.
    kappa_specification : str or callable, optional
        Partial-collusion / profit-weight specification. Passed through
        to ``pyblp.build_ownership`` for the kappa-modified ownership
        matrix.
    user_supplied_markups : str, optional
        Column name holding pre-computed markups; bypasses the conduct
        math. Used when the researcher computes markups externally.
    unit_tax, advalorem_tax, advalorem_payer : str, optional
        Tax column and payer specification (per-unit taxes and/or
        ad-valorem taxes). See Problem.solve for how taxes enter the
        effective markup.

        .. deprecated:: v0.4
            Per-model ``unit_tax`` / ``advalorem_tax`` /
            ``advalorem_payer`` are kept for backward compatibility but
            the preferred home for tax columns is at the Problem level
            (``Problem(..., unit_tax='col', advalorem_tax='col',
            advalorem_payer='firm')``). Use the salience flags
            ``unit_tax_salient`` and ``advalorem_tax_salient`` on
            individual models to opt out of a Problem-level tax for a
            salience / non-salience test.
    unit_tax_salient, advalorem_tax_salient : bool, optional
        v0.4: opt-out flags for Problem-level taxes on a per-model
        basis. Default is ``True`` (the model sees the Problem-level
        tax). Setting to ``False`` makes this model ignore the
        Problem-level tax while peers see it — the mechanism for
        salience tests (``Bertrand()`` vs.
        ``Bertrand(unit_tax_salient=False)`` under
        ``Problem(..., unit_tax='col')``). Ignored if no Problem-level
        tax is set (no-op).
    cost_scaling : str, float, int, optional
        Per-product cost-scaling factor :math:`\\lambda`. The effective
        markup becomes ``tax_adj / (1 + lambda) * markup`` and the
        effective price becomes ``tax_adj * price / (1 + lambda) - unit_tax``.

        Accepts either a column name (``str``) pointing to a per-product
        column in ``product_data``, or a numeric scalar (``float``/``int``)
        applied uniformly to every row.
    vertical_integration : str, optional
        Column name indicating which products are vertically integrated
        (store brands, etc.). Only meaningful on the outer (combined)
        model; for the ``Vertical`` wrapper it lives on the wrapper.
    mix_flag : str, optional
        Column name for the boolean Bertrand/Cournot mix indicator.
        Required by ``MixCournotBertrand``.

    Examples
    --------
    ``ConductModel`` is abstract; subclasses (``Bertrand``, ``Cournot``,
    etc.) implement the math. Instances of subclasses carry the shared
    config:

    >>> from pyRVtest import Bertrand, ConductModel
    >>> m = Bertrand(ownership='firm_ids')
    >>> isinstance(m, ConductModel)
    True
    >>> m.ownership
    'firm_ids'
    """

    def __init__(
            self,
            ownership: Optional[str] = None,
            kappa_specification: Optional[Union[str, Callable[[Any, Any], float]]] = None,
            user_supplied_markups: Optional[str] = None,
            unit_tax: Optional[str] = None,
            advalorem_tax: Optional[str] = None,
            advalorem_payer: Optional[str] = None,
            cost_scaling: Optional[Union[str, float, int]] = None,
            vertical_integration: Optional[str] = None,
            mix_flag: Optional[str] = None,
            unit_tax_salient: bool = True,
            advalorem_tax_salient: bool = True,
    ) -> None:
        self.ownership = ownership
        self.kappa_specification = kappa_specification
        self.user_supplied_markups = user_supplied_markups
        self.unit_tax = unit_tax
        self.advalorem_tax = advalorem_tax
        self.advalorem_payer = advalorem_payer
        self.cost_scaling: Optional[Union[str, float, int]] = cost_scaling
        self.vertical_integration = vertical_integration
        self.mix_flag = mix_flag
        # per-model salience flags for Problem-level taxes.
        # Default True means the model sees the Problem-level tax; False
        # makes this model opt out (used for salience tests).
        self.unit_tax_salient = unit_tax_salient
        self.advalorem_tax_salient = advalorem_tax_salient
        self._validate_shared_config()
        # emit the
        # per-model tax deprecation at construction time so users who
        # write e.g. ``pyRVtest.Bertrand(unit_tax='col')`` see the
        # v0.7-removal warning during migration, not only when the
        # Problem is built later.
        if unit_tax is not None:
            _maybe_warn_per_model_tax_deprecation('unit_tax')
        if advalorem_tax is not None:
            _maybe_warn_per_model_tax_deprecation('advalorem_tax')

    def _validate_shared_config(self) -> None:
        """Shared validation applicable to all subclasses."""
        if self.cost_scaling is not None and not isinstance(
                self.cost_scaling, (str, float, int)):
            raise TypeError(
                f"Expected cost_scaling to be a column-name string, a "
                f"numeric scalar, or None. "
                f"Received {type(self.cost_scaling).__name__}. "
                f"Fix: pass a column name like cost_scaling='lambda_col', a "
                f"scalar like cost_scaling=0.5, or drop the argument."
            )
        # Reject boolean explicitly: isinstance(True, int) is True.
        if isinstance(self.cost_scaling, bool):
            raise TypeError(
                f"Expected cost_scaling to be a column-name string or a "
                f"numeric scalar. "
                f"Received bool ({self.cost_scaling!r}). "
                f"Fix: pass a numeric scalar (e.g. 0.5) or a column name."
            )
        if self.advalorem_tax is not None and self.advalorem_payer is None:
            raise TypeError(
                "Expected advalorem_payer to be 'firm' or 'consumer' when "
                "advalorem_tax is supplied. "
                "Received advalorem_payer=None. "
                "Fix: set advalorem_payer='firm' or 'consumer' to indicate who "
                "remits the ad-valorem tax."
            )
        if self.advalorem_payer is not None and self.advalorem_payer not in {
                'firm', 'consumer', 'firms', 'consumers'}:
            raise TypeError(
                f"Expected advalorem_payer to be 'firm' or 'consumer' (or None). "
                f"Received {self.advalorem_payer!r}. "
                f"Fix: pass advalorem_payer='firm' or 'consumer'."
            )
        # per-model salience flags must be booleans. Reject
        # truthy-but-non-bool values (e.g. strings) with an actionable
        # error so users don't accidentally pass column names here.
        if not isinstance(self.unit_tax_salient, bool):
            raise TypeError(
                f"Expected unit_tax_salient to be True or False. "
                f"Received {type(self.unit_tax_salient).__name__} "
                f"({self.unit_tax_salient!r}). "
                f"Fix: pass unit_tax_salient=True (the default, model "
                f"sees Problem-level unit_tax) or False (opt out)."
            )
        if not isinstance(self.advalorem_tax_salient, bool):
            raise TypeError(
                f"Expected advalorem_tax_salient to be True or False. "
                f"Received {type(self.advalorem_tax_salient).__name__} "
                f"({self.advalorem_tax_salient!r}). "
                f"Fix: pass advalorem_tax_salient=True (default, model "
                f"sees Problem-level advalorem_tax) or False (opt out)."
            )

    # -----------------------------------------------------------------
    # Math hooks (abstract).
    # -----------------------------------------------------------------

    def _compute_markup(self, O: _NDArray, D: _NDArray, s: _NDArray) -> _NDArray:
        """Compute the markup vector for one market.

        Parameters
        ----------
        O
            Per-market ownership matrix, shape ``(J_t, J_t)``. For
            partial collusion this is the kappa-modified ownership;
            construction happens at ``Models`` / ``Problem`` setup.
        D
            Per-market response matrix ``ds/dp``, shape ``(J_t, J_t)``.
        s
            Per-market share vector, shape ``(J_t,)`` or ``(J_t, 1)``.

        Returns
        -------
        ndarray
            Markup vector for this market, shape ``(J_t, 1)``.
        """
        raise NotImplementedError(
            f"pyRVtest internal error: expected {type(self).__name__} to "
            f"override ConductModel._compute_markup(O, D, s). "
            f"Received a call on the abstract base method."
        )

    def _markup_derivative(
            self, O: _NDArray, D: _NDArray, dD: _NDArray,
            s: _NDArray, mu: _NDArray,
    ) -> _NDArray:
        """Derivative of the markup w.r.t. one theta parameter, per market.

        Parameters
        ----------
        O
            Per-market ownership matrix, shape ``(J_t, J_t)``.
        D
            Per-market response matrix, shape ``(J_t, J_t)``.
        dD
            Per-market ``d(D)/d(theta_k)`` for one specific k,
            shape ``(J_t, J_t)``. Callers loop over k.
        s
            Per-market shares, shape ``(J_t,)``.
        mu
            Per-market markups evaluated at the current theta, shape
            ``(J_t,)``.

        Returns
        -------
        ndarray
            ``d(markup)/d(theta_k)`` vector, shape ``(J_t,)``.
        """
        raise NotImplementedError(
            f"pyRVtest internal error: expected {type(self).__name__} to "
            f"override ConductModel._markup_derivative(O, D, dD, s, mu). "
            f"Received a call on the abstract base method."
        )

    # -----------------------------------------------------------------
    # Legacy string dispatch key.
    #
    # Step 5c's ModelFormulation alias reads this to construct the right
    # subclass. Also used internally by Models / _compute_markups while
    # the string-dispatch path still exists; step 5b wires the class
    # instance directly so this becomes alias-only.
    # -----------------------------------------------------------------

    _model_name: str = ''  # subclasses override

    def __repr__(self) -> str:
        parts = [f'ownership={self.ownership!r}']
        if self.kappa_specification is not None:
            parts.append(f'kappa_specification={self.kappa_specification!r}')
        if self.user_supplied_markups is not None:
            parts.append(f'user_supplied_markups={self.user_supplied_markups!r}')
        return f'{type(self).__name__}({", ".join(parts)})'
