"""Conduct model library: class-based API.

As of the mechanical conduct classes land:

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

Step 12 adds Dearing et al. (2024) simple-markup models:
``RuleOfThumb(phi)`` and ``ConstantMarkup(markup)`` in
``pyRVtest.models.constant``. Step 14
adds the labor-side models (``Monopsony``, ``BertrandWages``,
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
from .constant import ConstantMarkup, RuleOfThumb
from .custom import CustomConductModel
from .labor import BertrandWages, CournotEmployment, Monopsony, NashBargaining
from .mixed import MixCournotBertrand
from .standard import Bertrand, Cournot, Monopoly, PerfectCompetition
from .user_supplied import UserSuppliedMarkups
from .vertical import Vertical


# Frozen sets of model names used by Problem(market_side=...) to reject
# cross-side specifications at init time. Defined here (next to the class
# imports they name) so additions stay in lockstep with model registrations.
#
# ``perfect_competition`` is side-neutral: it yields zero markup /
# markdown either way and is the natural null on both sides. It lives in
# neither set so the cross-side check accepts it under both
# market_side='product' and market_side='labor'.
_PRODUCT_SIDE_MODEL_NAMES = frozenset({
    'bertrand', 'cournot', 'monopoly',
    'mix_cournot_bertrand', 'partial_collusion',
})
_LABOR_SIDE_MODEL_NAMES = frozenset({
    'monopsony', 'bertrand_wages', 'cournot_employment', 'nash_bargaining',
})


__all__ = [
    'ConductModel',
    'Bertrand',
    'Cournot',
    'Monopoly',
    'PerfectCompetition',
    'MixCournotBertrand',
    'PartialCollusion',
    'CustomConductModel',
    'UserSuppliedMarkups',
    'Vertical',
    # Dearing et al. (2024) simple-markup models.
    'RuleOfThumb',
    'ConstantMarkup',
    # labor-side conduct models.
    'Monopsony',
    'BertrandWages',
    'CournotEmployment',
    'NashBargaining',
]
