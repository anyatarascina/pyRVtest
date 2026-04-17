"""Labor-side demand/supply backends.

v0.4 step 14b re-exports the :class:`LaborSupplyBackend` skeleton.
The full Almagro-Sood-style labor supply backend is deferred to v0.5+
pending real labor-project data; this module ships the class shape and
``DemandBackend`` protocol surface so that ``Problem(market_side='labor')``
wiring, public API pin tests, and migration guides all exercise the
expected name.

Examples
--------
>>> from pyRVtest.backends import labor as labor_backends
>>> 'LaborSupplyBackend' in labor_backends.__all__
True
"""

from .nested_logit_labor import LaborSupplyBackend


__all__ = ['LaborSupplyBackend']
