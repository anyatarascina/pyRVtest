"""Demand and supply backend protocols and implementations.

The :class:`DemandBackend` Protocol (defined in :mod:`.base`) is the
core interface every demand backend implements: ``compute_jacobian``,
``compute_hessian``, parameter enumeration, and the ``perturbed``
context manager. :class:`SupportsDemandAdjustment` is an optional
mixin for backends that can supply first-stage correction inputs
(``xi``, ``Z_D``, ``W_D``, gradients) for the DMSS Appendix C
demand-adjustment correction.

Concrete implementations: :class:`PyBLPBackend`, :class:`LogitBackend`,
:class:`NestedLogitBackend`, :class:`UserSuppliedBackend`. A bare
labor-supply skeleton (:class:`LaborSupplyBackend`) is exposed for the
v0.5 labor-side workflow; calling its math methods raises
``NotImplementedError`` in v0.4.

Examples
--------
>>> from pyRVtest import backends
>>> 'LogitBackend' in backends.__all__
True
>>> 'PyBLPBackend' in backends.__all__
True
"""

from .base import DemandBackend, SupportsDemandAdjustment
from .labor.nested_logit_labor import LaborSupplyBackend
from .logit import LogitBackend
from .nested_logit import NestedLogitBackend
from .pyblp import PyBLPBackend
from .user import UserSuppliedBackend

__all__ = [
    'DemandBackend', 'SupportsDemandAdjustment',
    'PyBLPBackend', 'LogitBackend', 'NestedLogitBackend',
    'UserSuppliedBackend',
    'LaborSupplyBackend',
]
