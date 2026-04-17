"""CustomConductModel: user-supplied markup formula.

v0.4 step 5a. Supersedes the ``model_downstream='other'`` +
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

from .base import ConductModel


__all__ = ['CustomConductModel']


_NDArray = NDArray[Any]


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
            **kwargs: Any,
    ) -> None:
        if not callable(markup_fn):
            raise TypeError(
                f"markup_fn must be callable (ownership, D, shares) -> "
                f"markups, got {type(markup_fn).__name__}."
            )
        super().__init__(ownership=ownership, **kwargs)
        self.markup_fn = markup_fn
        self.name = name or 'custom'

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
            "CustomConductModel has no analytical markup derivative; "
            "compute_demand_adjustment falls back to finite-diff via "
            "backend.perturbed for custom and vertical models."
        )
