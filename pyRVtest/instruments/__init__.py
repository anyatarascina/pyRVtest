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
  - :func:`concentration_hhi`

Examples
--------
>>> from pyRVtest import instruments
>>> sorted(instruments.__all__)
['bartik', 'blp_instruments', 'concentration_hhi', 'differentiation_ivs', 'hausman', 'rival_sums']
"""

from .labor import bartik, concentration_hhi, hausman
from .product import blp_instruments, differentiation_ivs, rival_sums


__all__ = [
    # product-side
    'rival_sums', 'differentiation_ivs', 'blp_instruments',
    # labor-side
    'hausman', 'bartik', 'concentration_hhi',
]
