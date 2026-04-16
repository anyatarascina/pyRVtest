"""v0.4 step 1: import-roundtrip test for the new module skeleton.

Acceptance criterion from `.claude/plans/v0.4-refactor.md` §5 step 1:

> Create module skeleton (empty files, `__init__.py` re-exports mirror
> current public API) + `__all__` declarations everywhere

> Tests must pass: All step-0 tests + import-roundtrip test

This file verifies two invariants:

1. Every name in `pyRVtest.__all__` is importable from the package.
2. Every new v0.4 subpackage listed in the plan §4.1 is importable,
   has `__all__`, and (for step 1) is empty (`__all__ == []`) because
   no behavior has been moved yet. When a later step populates a
   subpackage, it will update that subpackage's `__all__` and the
   corresponding test here should follow.

Future migrations that add names to a subpackage's `__all__` should add
a positive test to this file (e.g., step 3 adds backends/pyblp.py with
`PyBLPBackend` in `__all__`; a new test here should assert it's
importable).
"""

from __future__ import annotations

import importlib
import pytest

import pyRVtest


# ---------------------------------------------------------------------------
# 1. Every name in pyRVtest.__all__ is importable.
# ---------------------------------------------------------------------------

def test_public_api_all_importable():
    """Every symbol in pyRVtest.__all__ resolves to a non-None attribute."""
    missing = []
    for name in pyRVtest.__all__:
        if not hasattr(pyRVtest, name):
            missing.append(name)
        elif getattr(pyRVtest, name) is None:
            missing.append(f"{name} (is None)")
    assert not missing, f"Names in __all__ not importable: {missing}"


def test_v03_public_api_preserved():
    """The pre-v0.4 public API symbols are all still present.

    Paranoid explicit check so a future refactor cannot silently remove
    a user-facing symbol without tripping this test. The list below is
    the v0.3 baseline at v0.3.3-stable.
    """
    expected = {
        'data', 'options',
        'build_ownership', 'build_markups', 'construct_passthrough_matrix',
        'evaluate_first_order_conditions', 'read_pickle',
        'Formulation', 'ModelFormulation',
        'Problem', 'Models', 'Products',
        'ProblemResults',
        '__version__',
    }
    missing = expected - set(pyRVtest.__all__)
    assert not missing, (
        f"v0.3 public API symbol(s) dropped from pyRVtest.__all__: {missing}. "
        f"This would break existing user code. See plan §6 backward "
        f"compatibility guarantees."
    )


# ---------------------------------------------------------------------------
# 2. v0.4 subpackage skeletons exist, have __all__, and (step 1) empty.
# ---------------------------------------------------------------------------

# Modules expected to exist as of v0.4 step 1. Each is the (dotted path,
# expected __all__) pair. Step 1 __all__ is always empty; later steps
# update this list as they populate the subpackages.
_STEP_1_SKELETON_MODULES: list[tuple[str, list[str]]] = [
    # Updated in step 3a: protocols populated in backends/base.py.
    # Updated in step 3b: PyBLPBackend added.
    # Updated in step 3c: LogitBackend, NestedLogitBackend added.
    # Updated in step 3d: UserSuppliedBackend added.
    ('pyRVtest.backends', [
        'DemandBackend', 'SupportsDemandAdjustment',
        'PyBLPBackend', 'LogitBackend', 'NestedLogitBackend',
        'UserSuppliedBackend',
    ]),
    ('pyRVtest.backends.base', ['DemandBackend', 'SupportsDemandAdjustment']),
    ('pyRVtest.backends.pyblp', ['PyBLPBackend']),
    ('pyRVtest.backends.logit', [
        'compute_analytical_jacobian', 'compute_analytical_hessian',
        '_logit_jacobian', '_nested_logit_jacobian',
        '_nested_logit_jacobian_derivative', '_infer_nesting_columns',
        'LogitBackend',
    ]),
    # Split out after step 3 for user-facing clarity (tracebacks + API docs).
    ('pyRVtest.backends.nested_logit', ['NestedLogitBackend']),
    ('pyRVtest.backends.user', ['UserSuppliedBackend']),
    ('pyRVtest.backends.labor', []),
    ('pyRVtest.backends.labor.nested_logit_labor', []),
    ('pyRVtest.models', []),
    ('pyRVtest.models.standard', []),
    ('pyRVtest.models.vertical', []),
    ('pyRVtest.models.mixed', []),
    ('pyRVtest.models.collusion', []),
    ('pyRVtest.models.constant', []),
    ('pyRVtest.models.labor', []),
    ('pyRVtest.models.custom', []),
    ('pyRVtest.instruments', []),
    ('pyRVtest.instruments.product', []),
    ('pyRVtest.instruments.labor', []),
    ('pyRVtest.solve', []),
    ('pyRVtest.solve.markups', []),
    ('pyRVtest.solve.passthrough', []),
    ('pyRVtest.solve.orthogonalize', []),
    ('pyRVtest.solve.endogenous_cost', []),
    ('pyRVtest.solve.demand_adjustment', []),
    ('pyRVtest.solve.test_engine', []),
    # Added in v0.4 step 2: Products extracted to its own module.
    ('pyRVtest.products', ['Products']),
]


@pytest.mark.parametrize('dotted_path,expected_all', _STEP_1_SKELETON_MODULES)
def test_v04_skeleton_module_importable_and_all_matches(dotted_path, expected_all):
    """Every v0.4 skeleton module imports and has the expected __all__.

    At step 1 every expected_all is []; later steps update the
    _STEP_1_SKELETON_MODULES list as they populate subpackages.
    """
    mod = importlib.import_module(dotted_path)
    assert hasattr(mod, '__all__'), (
        f"{dotted_path} has no __all__ declaration. Every subpackage or "
        f"module in the v0.4 tree must declare __all__ explicitly."
    )
    assert mod.__all__ == expected_all, (
        f"{dotted_path}.__all__ = {mod.__all__!r}, expected {expected_all!r}. "
        f"If this is a deliberate change (step N populated the subpackage), "
        f"update _STEP_1_SKELETON_MODULES in tests/test_import_roundtrip.py."
    )


# ---------------------------------------------------------------------------
# 3. Core classes still behave correctly (sanity check the move of
#    results.py -> results/__init__.py didn't break anything).
# ---------------------------------------------------------------------------

def test_problem_results_class_location():
    """ProblemResults and Progress are accessible from pyRVtest.results."""
    from pyRVtest.results import ProblemResults, Progress
    # Existence check; not a class-identity check because the class is
    # user-facing and may be re-exported from pyRVtest top-level too.
    assert ProblemResults is not None
    assert Progress is not None


def test_progress_is_dataclass():
    """Progress is still a dataclass with expected fields.

    Sanity check after moving results.py into a subpackage (step 1).
    """
    from pyRVtest.results import Progress
    import dataclasses
    assert dataclasses.is_dataclass(Progress)
    field_names = {f.name for f in dataclasses.fields(Progress)}
    # Sample of the fields that downstream code depends on
    for expected in ('problem', 'markups', 'g', 'Q',
                     'test_statistic_RV', 'F', 'MCS_pvalues'):
        assert expected in field_names, (
            f"Progress dataclass missing expected field {expected!r}"
        )
