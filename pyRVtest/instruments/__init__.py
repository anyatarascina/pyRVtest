"""Testing-IV constructors for product-side and labor-side specifications.

v0.4 step 13. Mechanical, vectorized constructors for building columns that
users pass to :class:`pyRVtest.Problem` via ``instrument_formulation``.

These are thin convenience helpers; they do not encode methodology opinions
about which bundles of instruments to use together.

Product-side (:mod:`pyRVtest.instruments.product`):

  - :func:`rival_sums`
  - :func:`differentiation_ivs`
  - :func:`blp_instruments`

Labor-side (:mod:`pyRVtest.instruments.labor`):

  - :func:`hausman`
  - :func:`bartik`

A ``concentration_hhi`` helper is *not* provided on the labor side: labor-
market HHI is a function of the endogenous employment shares and is not a
valid instrument for wages. See :mod:`pyRVtest.instruments.labor` for the
full rationale and references.

Examples
--------
>>> from pyRVtest import instruments
>>> sorted(instruments.__all__)
['bartik', 'blp_instruments', 'differentiation_ivs', 'hausman', 'rival_sums']
"""

from .labor import bartik, hausman
from .product import blp_instruments, differentiation_ivs, rival_sums


__all__ = [
    # product-side
    'rival_sums', 'differentiation_ivs', 'blp_instruments',
    # labor-side
    'hausman', 'bartik',
]
