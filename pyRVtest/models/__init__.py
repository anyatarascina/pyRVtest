"""Conduct model library: class-based API (v0.4 step 5).

As of v0.4 step 5a the mechanical conduct classes land:

    from pyRVtest.models import (
        ConductModel,
        Bertrand, Cournot, Monopoly, PerfectCompetition,
        MixCournotBertrand, PartialCollusion, CustomConductModel,
        Vertical,
    )

Usage::

    pyRVtest.Problem(
        ...,
        models=[
            Bertrand(ownership='firm_ids'),
            PerfectCompetition(),
            Vertical(
                downstream=Bertrand(ownership='firm_ids'),
                upstream=Monopoly(ownership='manufacturer_ids'),
                vertical_integration='vi_col',
            ),
        ],
    )

Step 5b wires ``Problem(models=[...])`` into the pipeline; step 5c
preserves ``ModelFormulation(model_downstream='bertrand', ...)`` as a
deprecation alias that constructs the right class internally.

Step 12 will add ``ConstantMarkup``, ``RuleOfThumb``, and ``CostPlus``
once the Dearing notation question is resolved. Step 14 will add the
labor-side models (``Monopsony``, ``BertrandWages``,
``CournotEmployment``, ``NashBargaining``).

See ``.claude/plans/v0.4-refactor.md`` §4.2 for the full API design.

Examples
--------
>>> from pyRVtest import models
>>> 'Bertrand' in models.__all__
True
>>> 'Vertical' in models.__all__
True
"""

from .base import ConductModel
from .collusion import PartialCollusion
from .custom import CustomConductModel
from .mixed import MixCournotBertrand
from .standard import Bertrand, Cournot, Monopoly, PerfectCompetition
from .vertical import Vertical


__all__ = [
    'ConductModel',
    'Bertrand',
    'Cournot',
    'Monopoly',
    'PerfectCompetition',
    'MixCournotBertrand',
    'PartialCollusion',
    'CustomConductModel',
    'Vertical',
]
