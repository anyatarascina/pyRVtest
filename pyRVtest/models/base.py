"""Base class for all conduct models (v0.4 step 5a).

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

from typing import Any, Callable, Optional, Union

import numpy as np
from numpy.typing import NDArray


__all__ = ['ConductModel']


_NDArray = NDArray[Any]


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
    cost_scaling : str, optional
        Column name for the per-product cost-scaling factor.
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
            cost_scaling: Optional[str] = None,
            vertical_integration: Optional[str] = None,
            mix_flag: Optional[str] = None,
    ) -> None:
        self.ownership = ownership
        self.kappa_specification = kappa_specification
        self.user_supplied_markups = user_supplied_markups
        self.unit_tax = unit_tax
        self.advalorem_tax = advalorem_tax
        self.advalorem_payer = advalorem_payer
        self.cost_scaling = cost_scaling
        self.vertical_integration = vertical_integration
        self.mix_flag = mix_flag
        self._validate_shared_config()

    def _validate_shared_config(self) -> None:
        """Shared validation applicable to all subclasses."""
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
