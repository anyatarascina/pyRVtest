"""Public-facing objects for pyRVtest.

Exports the main user API: ``Problem``, ``Formulation``,
``ModelFormulation``, ``ProblemResults``, the ``ConductModel`` class
hierarchy, the in-package logit / nested-logit estimators, and the
helper functions ``build_markups``, ``build_ownership``,
``construct_passthrough_matrix``, ``evaluate_first_order_conditions``,
and ``read_pickle``.
"""

from . import data, options
from . import backends, estimators, instruments, models, solve
from .estimators import LogitEstimator, NestedLogitEstimator
from .exceptions import (
    BackendError,
    DemandBackendError,
    HessianUnavailableError,
    InstrumentDataError,
    PyRVTestError,
    ValidationError,
)
from .formulation import Formulation, ModelFormulation
from .markups import (
    build_ownership, build_markups, construct_passthrough_matrix,
    evaluate_first_order_conditions, read_pickle
)
from .models import (
    Bertrand, BertrandWages, ConductModel, ConstantMarkup, Cournot, CournotEmployment,
    CustomConductModel, MixCournotBertrand, Monopoly, Monopsony, NashBargaining,
    PartialCollusion, PerfectCompetition, RuleOfThumb, UserSuppliedMarkups, Vertical,
)
from .problem import Problem, Models
from .products import Products
from .results import PanelResults, ProblemResults
from .solve.passthrough import build_passthrough
from ._agent_guide import show_agent_guide
from .version import __version__

__all__ = [
    'data', 'options', 'build_ownership', 'build_markups', 'construct_passthrough_matrix',
    'evaluate_first_order_conditions', 'read_pickle', 'Formulation', 'ModelFormulation', 'Problem', 'Models',
    'Products', 'ProblemResults', 'PanelResults', '__version__',
    'backends', 'estimators', 'instruments', 'models', 'solve',
    'LogitEstimator', 'NestedLogitEstimator',
    'ConductModel', 'Bertrand', 'Cournot', 'Monopoly', 'PerfectCompetition',
    'MixCournotBertrand', 'PartialCollusion', 'CustomConductModel',
    'UserSuppliedMarkups',
    'Vertical',
    'RuleOfThumb', 'ConstantMarkup',
    'Monopsony', 'BertrandWages', 'CournotEmployment', 'NashBargaining',
    'build_passthrough',
    'show_agent_guide',
    'PyRVTestError', 'ValidationError', 'InstrumentDataError',
    'BackendError', 'DemandBackendError', 'HessianUnavailableError',
]
