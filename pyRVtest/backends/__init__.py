"""Demand / supply backend protocols and implementations.

As of v0.4 step 3a, the `DemandBackend` and `SupportsDemandAdjustment`
protocols are defined in `base.py` and re-exported here. Concrete
implementations (PyBLPBackend, LogitBackend, NestedLogitBackend,
UserSuppliedBackend) land in sub-steps 3b/3c/3d. The LaborSupplyBackend
skeleton lands in step 14.

See `.claude/plans/v0.4-refactor.md` §4.1 for the module layout and §3
for the PyBLPBackend prototype design.

Examples
--------
>>> from pyRVtest import backends
>>> 'LogitBackend' in backends.__all__
True
>>> 'PyBLPBackend' in backends.__all__
True
"""

from .base import DemandBackend, SupportsDemandAdjustment
from .logit import LogitBackend
from .nested_logit import NestedLogitBackend
from .pyblp import PyBLPBackend
from .user import UserSuppliedBackend

__all__ = [
    'DemandBackend', 'SupportsDemandAdjustment',
    'PyBLPBackend', 'LogitBackend', 'NestedLogitBackend',
    'UserSuppliedBackend',
]
