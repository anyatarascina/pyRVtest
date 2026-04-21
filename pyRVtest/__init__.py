"""Public-facing objects.

v0.4 step 1: new subpackages (`backends`, `models`, `instruments`,
`solve`) are exposed here as empty namespaces to pin the future layout
from `.claude/plans/v0.4-refactor.md` §4.1. Each subpackage ships with
`__all__ = []` until the step that populates it (3, 4, 5, 8, etc.).

The existing public API (Problem, Formulation, ModelFormulation,
ProblemResults, build_markups, build_ownership, etc.) is unchanged.
"""

from . import data, options
# New v0.4 subpackages (empty skeletons until their populating steps).
from . import backends, instruments, models, solve
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
# v0.4 step 5a: class-based ConductModel API re-exported at the package level.
# v0.4 step 14a: labor-side models added next to their product-side siblings.
from .models import (
    Bertrand, BertrandWages, ConductModel, ConstantMarkup, Cournot, CournotEmployment,
    CustomConductModel, MixCournotBertrand, Monopoly, Monopsony, NashBargaining,
    PartialCollusion, PerfectCompetition, RuleOfThumb, UserSuppliedMarkups, Vertical,
)
from .problem import Problem, Models
from .products import Products
from .results import PanelResults, ProblemResults
# v0.4 step 11: public build_passthrough helper re-exported at the package level.
from .solve.passthrough import build_passthrough
# v0.4 step 23: agent-guide exporter for AI assistants and new contributors.
from ._agent_guide import show_agent_guide
from .version import __version__

__all__ = [
    # v0.3 public API (unchanged)
    'data', 'options', 'build_ownership', 'build_markups', 'construct_passthrough_matrix',
    'evaluate_first_order_conditions', 'read_pickle', 'Formulation', 'ModelFormulation', 'Problem', 'Models',
    'Products', 'ProblemResults', 'PanelResults', '__version__',
    # v0.4 subpackage namespaces (populated in later migration steps)
    'backends', 'instruments', 'models', 'solve',
    # v0.4 step 5a: class-based ConductModel API.
    'ConductModel', 'Bertrand', 'Cournot', 'Monopoly', 'PerfectCompetition',
    'MixCournotBertrand', 'PartialCollusion', 'CustomConductModel',
    # v0.4.0rc1: first-class wrapper for pre-computed markup columns.
    'UserSuppliedMarkups',
    'Vertical',
    # v0.4 step 12: Dearing et al. (2026) simple-markup models.
    'RuleOfThumb', 'ConstantMarkup',
    # v0.4 step 14a: labor-side conduct models.
    'Monopsony', 'BertrandWages', 'CournotEmployment', 'NashBargaining',
    # v0.4 step 11: public build_passthrough helper.
    'build_passthrough',
    # v0.4 step 23: agent-guide exporter.
    'show_agent_guide',
    # v0.4 step 19: custom exception hierarchy.
    'PyRVTestError', 'ValidationError', 'InstrumentDataError',
    'BackendError', 'DemandBackendError', 'HessianUnavailableError',
]
