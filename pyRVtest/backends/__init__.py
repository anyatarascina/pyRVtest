"""Demand / supply backend protocols and implementations.

As of v0.4 step 3a, the `DemandBackend` and `SupportsDemandAdjustment`
protocols are defined in `base.py` and re-exported here. Concrete
implementations (PyBLPBackend, LogitBackend, NestedLogitBackend,
UserSuppliedBackend) land in sub-steps 3b/3c/3d. The LaborSupplyBackend
skeleton lands in step 14.

See `.claude/plans/v0.4-refactor.md` §4.1 for the module layout and §3
for the PyBLPBackend prototype design.
"""

from .base import DemandBackend, SupportsDemandAdjustment
from .logit import LogitBackend, NestedLogitBackend
from .pyblp import PyBLPBackend

__all__ = [
    'DemandBackend', 'SupportsDemandAdjustment',
    'PyBLPBackend', 'LogitBackend', 'NestedLogitBackend',
]
