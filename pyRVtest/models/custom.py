"""CustomConductModel: user-supplied markup formula.

Supersedes the ``model_downstream='other'`` +
``custom_model_specification`` callable path with a cleaner class API.
The deprecation alias (step 5c) keeps the old string+dict usage
working.

The callable has the signature ``(ownership, response_matrix, shares)
-> markups`` (shape ``(J, 1)``). Derivatives w.r.t. demand parameters
fall back to finite-diff in the unified ``compute_demand_adjustment``
function.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypeAlias

from ..exceptions import ValidationError
from .base import ConductModel


__all__ = ['CustomConductModel']


_NDArray: TypeAlias = NDArray[Any]


class CustomConductModel(ConductModel):
    r"""User-supplied markup formula.

    Parameters
    ----------
    markup_fn : callable, required
        Function with signature ``(ownership, response_matrix, shares)
        -> ndarray (J, 1)``. Called once per market.
    ownership : str, optional
        Column name for the ownership identifiers; passed through the
        usual pyblp ownership construction pipeline.
    name : str, optional
        Label used in results output / repr.
    side : {'product', 'labor', None}, optional
        Sign-convention opt-in for cross-side validation. Unlike
        :class:`~pyRVtest.PerfectCompetition` (zero markup == zero
        markdown, genuinely side-neutral), a user-supplied
        ``markup_fn`` implicitly picks a sign convention. This flag
        makes that choice explicit so that a product-side formula
        cannot silently be used on a labor problem (or vice versa).

        * ``'product'`` (or ``None``, the default) — the formula is
          written for the classic downstream-markup convention
          (``p - MC``). Accepted on ``market_side='product'`` only.
        * ``'labor'`` — the formula is written for the upward-sloping
          labor-supply convention (``w - MRP``, markdown). Accepted on
          ``market_side='labor'`` only.

        Mismatches raise :class:`~pyRVtest.exceptions.ValidationError`
        at ``Problem.__init__``.

    Examples
    --------
    >>> import numpy as np
    >>> from pyRVtest import CustomConductModel
    >>> # A trivial markup function: constant 10% markup per product.
    >>> def fixed_markup(O, D, s):
    ...     return 0.1 * np.ones((len(s), 1))
    >>> model = CustomConductModel(markup_fn=fixed_markup, name='flat')
    >>> model.name
    'flat'
    >>> model._compute_markup(np.eye(2), np.eye(2), np.array([0.3, 0.4]))
    array([[0.1],
           [0.1]])
    """

    _model_name = 'other'

    def __init__(
            self,
            markup_fn: Callable[[_NDArray, _NDArray, _NDArray], _NDArray],
            ownership: Optional[str] = None,
            name: Optional[str] = None,
            side: Optional[str] = None,
            **kwargs: Any,
    ) -> None:
        if not callable(markup_fn):
            raise TypeError(
                f"Expected markup_fn to be callable with signature "
                f"(ownership, response_matrix, shares) -> ndarray; "
                f"markup_fn must be callable. "
                f"Received {type(markup_fn).__name__}. "
                f"Fix: pass a Python function or lambda that takes three "
                f"per-market arguments and returns the markup vector."
            )
        if side is not None and side not in ('product', 'labor'):
            raise ValidationError(
                f"Expected side to be one of 'product', 'labor', or None. "
                f"Received {side!r}. "
                f"Fix: pass side='product' (the default) for a classic "
                f"downstream-markup formula, side='labor' for an upward-"
                f"sloping labor-supply markdown formula, or omit the "
                f"argument to default to the product-side convention."
            )
        super().__init__(ownership=ownership, **kwargs)
        self.markup_fn = markup_fn
        self.name = name or 'custom'
        self.side: Optional[str] = side

    def __repr__(self) -> str:
        parts = [f'ownership={self.ownership!r}']
        if self.kappa_specification is not None:
            parts.append(f'kappa_specification={self.kappa_specification!r}')
        if self.user_supplied_markups is not None:
            parts.append(f'user_supplied_markups={self.user_supplied_markups!r}')
        if self.side is not None:
            parts.append(f'side={self.side!r}')
        return f'{type(self).__name__}({", ".join(parts)})'

    def _compute_markup(self, O: _NDArray, D: _NDArray, s: _NDArray) -> _NDArray:
        markup = self.markup_fn(O, D, np.asarray(s))
        markup = np.asarray(markup)
        if markup.ndim == 1:
            markup = markup.reshape(-1, 1)
        return markup

    def _markup_derivative(
            self, O: _NDArray, D: _NDArray, dD: _NDArray,
            s: _NDArray, mu: _NDArray,
    ) -> _NDArray:
        # Custom markup functions have no closed-form derivative by
        # assumption; compute_demand_adjustment handles the finite-diff
        # fallback via backend.perturbed at the pipeline level. This
        # method signals that to the caller.
        raise NotImplementedError(
            "Expected CustomConductModel to route markup derivatives through "
            "the backend's finite-diff perturbation path (compute_demand_adjustment "
            "handles this for custom and vertical models via backend.perturbed). "
            "Received a direct _markup_derivative call on CustomConductModel, "
            "which has no closed-form derivative by construction. "
            "Fix: this method should not be called directly; check the caller "
            "and route through the finite-diff fallback instead."
        )
