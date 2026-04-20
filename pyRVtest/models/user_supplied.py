"""UserSuppliedMarkups: pre-computed markup column wrapper.

v0.4.0rc1 step 1. Supersedes the legacy pattern
``ModelFormulation(user_supplied_markups='col', ownership_downstream='firm_ids')``
(which asserted in the reverse adapter after ``ModelFormulation`` was
deprecated) with a first-class class. Use::

    pyRVtest.UserSuppliedMarkups(markups='mkup_col', ownership='firm_ids')

The named column in ``product_data`` is used as the markup directly;
``_compute_markups`` at ``pyRVtest/markups.py`` bypasses the FOC
dispatch when ``user_supplied_markups`` is set, so ``_compute_markup``
and ``_markup_derivative`` on this class are never called.

Side-neutral like :class:`~pyRVtest.PerfectCompetition`: the user's
column encodes the sign convention, so the class accepts
``market_side='product'`` and ``market_side='labor'`` without an
explicit flag.
"""

from __future__ import annotations

from typing import Any, Optional

from numpy.typing import NDArray
from typing_extensions import TypeAlias

from .base import ConductModel


__all__ = ['UserSuppliedMarkups']


_NDArray: TypeAlias = NDArray[Any]


class UserSuppliedMarkups(ConductModel):
    r"""Pre-computed markup column wrapper.

    Parameters
    ----------
    markups : str, required
        Column name in ``product_data`` holding the pre-computed
        markups (or markdowns on a labor-side problem). The sign
        convention is encoded by the user in the column values; the
        class does not transform them.
    ownership : str, optional
        Column name for the ownership identifiers. Stored for
        consistency with other conduct models but unused when markups
        are supplied (no FOC math runs).
    **kwargs
        Passed through to :class:`~pyRVtest.ConductModel`: taxes
        (``unit_tax``, ``advalorem_tax``, ``advalorem_payer``),
        ``cost_scaling``, ``vertical_integration``, and the per-model
        salience flags (``unit_tax_salient``, ``advalorem_tax_salient``).

    Examples
    --------
    >>> from pyRVtest import UserSuppliedMarkups
    >>> model = UserSuppliedMarkups(markups='mkup_B', ownership='firm_ids')
    >>> model._model_name
    'user_supplied'
    >>> model.user_supplied_markups
    'mkup_B'
    """

    _model_name = 'user_supplied'

    def __init__(
            self,
            markups: str,
            ownership: Optional[str] = None,
            **kwargs: Any,
    ) -> None:
        if not isinstance(markups, str) or not markups:
            raise TypeError(
                f"Expected markups to be a non-empty string naming a column "
                f"in product_data that holds the pre-computed markups. "
                f"Received {type(markups).__name__} ({markups!r}). "
                f"Fix: pass markups='my_markup_column'."
            )
        if 'user_supplied_markups' in kwargs:
            raise TypeError(
                "Expected the markup column to be supplied via the "
                "markups= parameter only. "
                "Received both markups= and user_supplied_markups=. "
                "Fix: drop the user_supplied_markups= kwarg; pass the "
                "column name as markups=."
            )
        super().__init__(
            ownership=ownership,
            user_supplied_markups=markups,
            **kwargs,
        )

    def __repr__(self) -> str:
        parts = [f'markups={self.user_supplied_markups!r}']
        if self.ownership is not None:
            parts.append(f'ownership={self.ownership!r}')
        return f'{type(self).__name__}({", ".join(parts)})'

    def _compute_markup(self, O: _NDArray, D: _NDArray, s: _NDArray) -> _NDArray:
        # Unreachable: ``_compute_markups`` at pyRVtest/markups.py:216
        # bypasses the FOC dispatch when ``user_supplied_markups`` is
        # set on the model. This method exists only to satisfy the
        # abstract interface on :class:`~pyRVtest.ConductModel`.
        raise NotImplementedError(
            "Expected UserSuppliedMarkups to read markups directly from the "
            "named column in product_data (bypassing FOC dispatch in "
            "_compute_markups). "
            "Received a direct _compute_markup call, which should never "
            "happen for this class. "
            "Fix: check the caller; markups are supplied, not computed."
        )

    def _markup_derivative(
            self, O: _NDArray, D: _NDArray, dD: _NDArray,
            s: _NDArray, mu: _NDArray,
    ) -> _NDArray:
        # User-supplied markups have no closed-form derivative w.r.t.
        # demand parameters (they are an external column, not a
        # functional of the demand system). The demand-adjustment
        # pipeline does not invoke this path when the model has
        # ``user_supplied_markups`` set.
        raise NotImplementedError(
            "Expected UserSuppliedMarkups to have no demand-parameter "
            "derivative (markups are an external column, not a functional "
            "of demand). "
            "Received a direct _markup_derivative call, which should never "
            "happen for this class. "
            "Fix: check the caller; supplied markups do not participate in "
            "analytical demand adjustment."
        )
