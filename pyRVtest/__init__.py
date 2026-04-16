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
from .formulation import Formulation, ModelFormulation
from .markups import (
    build_ownership, build_markups, construct_passthrough_matrix,
    evaluate_first_order_conditions, read_pickle
)
from .problem import Problem, Models
from .products import Products
from .results import ProblemResults
from .version import __version__

__all__ = [
    # v0.3 public API (unchanged)
    'data', 'options', 'build_ownership', 'build_markups', 'construct_passthrough_matrix',
    'evaluate_first_order_conditions', 'read_pickle', 'Formulation', 'ModelFormulation', 'Problem', 'Models',
    'Products', 'ProblemResults', '__version__',
    # v0.4 subpackage namespaces (populated in later migration steps)
    'backends', 'instruments', 'models', 'solve',
]
