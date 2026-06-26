"""Public-facing objects for pyRVtest.

Exports the main user API: ``Problem``, ``Formulation``,
``ModelFormulation``, ``ProblemResults``, the ``ConductModel`` class
hierarchy, the in-package logit / nested-logit estimators, and the
helper functions ``build_markups``, ``build_ownership``,
``build_phi_matrix``, ``build_markup_derivative``,
``construct_passthrough_matrix``, ``evaluate_first_order_conditions``,
and ``read_pickle``.
"""

# v0.4.0rc5: PEP 508 cannot express "numpy 2 requires pyblp >= 1.2"
# directly in requirements.txt because environment markers cannot
# reference one dependency's version from another. CI exercises only
# the two consistent matrix variants (numpy<2 + pyblp<1.2 and
# numpy>=2 + pyblp>=1.2). A fresh install under Python 3.12+ can
# still pip-resolve to numpy 2 + pyblp 1.1.x, where pyblp's older
# LAPACK paths produce silent NaN / wrong-sign diagnostics. Detect
# the bad combination here and raise a clear ImportError with the
# fix the user actually needs.
import numpy as _np
import pyblp as _pyblp


def _parse_major_minor(version_str: str) -> "tuple[int, int]":
    """Best-effort (major, minor) tuple from a PEP 440 version string.

    Tolerates suffixes like ``1.2.0rc1`` / ``2.4.4.dev0``; only the
    leading numeric major.minor is used for the gate.
    """
    parts = version_str.split('+', 1)[0].split('-', 1)[0].split('.')

    def _digits_or_zero(s: str) -> int:
        digits = ''.join(ch for ch in s if ch.isdigit())
        return int(digits) if digits else 0

    major = _digits_or_zero(parts[0]) if len(parts) >= 1 else 0
    minor = _digits_or_zero(parts[1]) if len(parts) >= 2 else 0
    return (major, minor)


if _parse_major_minor(_np.__version__) >= (2, 0) and _parse_major_minor(_pyblp.__version__) < (1, 2):
    raise ImportError(
        f"pyRVtest detected an unsupported dependency combination: "
        f"numpy=={_np.__version__} with pyblp=={_pyblp.__version__}. "
        f"numpy >= 2.0 requires pyblp >= 1.2 (the first pyblp release "
        f"with numpy-2 LAPACK compatibility). "
        f"Fix: either downgrade numpy (pip install 'numpy<2'), or "
        f"upgrade pyblp (pip install 'pyblp>=1.2'). "
        f"See requirements.txt for the supported matrix."
    )
del _np, _pyblp, _parse_major_minor

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
    build_ownership, build_markups, build_phi_matrix, build_markup_derivative,
    PhiMatrixData, MarkupDerivativeData,
    construct_passthrough_matrix, evaluate_first_order_conditions, read_pickle
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
    'data', 'options', 'build_ownership', 'build_markups', 'build_phi_matrix',
    'build_markup_derivative', 'PhiMatrixData', 'MarkupDerivativeData',
    'construct_passthrough_matrix',
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
