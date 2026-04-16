"""Backward-compat shim: re-exports analytical demand functions from backends.logit.

v0.4 step 3c. The canonical implementations now live in
`pyRVtest/backends/logit.py`. This module remains so that existing
internal callers (pyRVtest/problem.py, pyRVtest/markups.py) and
external test files (tests/test_demand_params.py,
tests/test_demand_params_integration.py) keep working without import
path changes.

Step 4 (unify demand adjustment) may remove this shim once the internal
callers no longer reference `pyRVtest.demand_jacobian`. External users
who import from here will get a deprecation warning at that point.
"""

from .backends.logit import (  # noqa: F401
    _infer_nesting_columns,
    _logit_jacobian,
    _nested_logit_jacobian,
    _nested_logit_jacobian_derivative,
    compute_analytical_hessian,
    compute_analytical_jacobian,
)

__all__ = [
    'compute_analytical_jacobian',
    'compute_analytical_hessian',
    '_logit_jacobian',
    '_nested_logit_jacobian',
    '_nested_logit_jacobian_derivative',
    '_infer_nesting_columns',
]
