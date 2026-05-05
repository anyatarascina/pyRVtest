"""PartialCollusion: price-setting with a kappa-modified ownership matrix.

``PartialCollusion`` shares the Bertrand FOC; the difference lives
entirely in how the ownership matrix is constructed (kappa modifies
off-diagonal elements) at ``Models`` / ``Problem`` setup.
``_compute_markup`` and ``_markup_derivative`` therefore inherit
unchanged from ``Bertrand``. The class signals intent at the call site
and enforces that ``kappa_specification`` is supplied.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Union

from .standard import Bertrand


__all__ = ['PartialCollusion']


class PartialCollusion(Bertrand):
    r"""Price-setting with a kappa-profit-weighted ownership matrix.

    Parameters
    ----------
    ownership : str
        Column name for firm identifiers (required, same as Bertrand).
    kappa_specification : str or callable, required
        Specification for the kappa profit-weight matrix. Passed to
        ``pyblp.build_ownership``. Common values:

        - ``'collusion_row'`` — full row-based collusion (Miller-Weinberg).
        - ``'collusion_column'`` — column-based.
        - Custom callable ``(firm_id_i, firm_id_j) -> float`` for
          flexible specifications.

    Examples
    --------
    >>> from pyRVtest import PartialCollusion
    >>> model = PartialCollusion(ownership='firm_ids', kappa_specification='collusion_row')
    >>> model.ownership
    'firm_ids'
    >>> model.kappa_specification
    'collusion_row'
    >>> PartialCollusion(ownership='firm_ids')  # doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    TypeError: Expected kappa_specification to define the profit-weight matrix ...
    """

    _model_name = 'bertrand'  # same math; only ownership construction differs

    def __init__(
            self,
            ownership: Optional[str] = None,
            kappa_specification: Optional[Union[str, Callable[[Any, Any], float]]] = None,
            **kwargs: Any,
    ) -> None:
        if kappa_specification is None:
            raise TypeError(
                "Expected kappa_specification to define the profit-weight matrix "
                "for the partial-collusion model; kappa_specification is "
                "required for PartialCollusion. "
                "Received kappa_specification=None. "
                "Fix: pass kappa_specification='collusion_row' (or "
                "'collusion_column', or a custom callable). For standard "
                "Bertrand (kappa = identity) use Bertrand(...) directly."
            )
        super().__init__(
            ownership=ownership,
            kappa_specification=kappa_specification,
            **kwargs,
        )
