"""v0.4 step 22: public API __all__ audit.

Acceptance criterion from `.claude/plans/v0.4-refactor.md` §5 step 22:

> Audit pass: for every pyRVtest (sub)module that declares __all__,
> verify that (a) every non-underscore attribute is declared and (b)
> every declared name resolves to a real attribute.

This locks the incremental invariant used by prior steps: each sub-step
that added a class/function also added the name to the right __all__.
This test walks the entire package tree and flags drift.

Complements `test_import_roundtrip.py`, which hand-enumerates the
expected __all__ contents; this file instead dynamically audits the
two invariants so that adding a new public helper without updating
__all__ fails fast.

Invariants:

1. No orphan public names. For every module that declares __all__,
   every non-underscore attribute that (a) is not a submodule and
   (b) is defined here (object.__module__ == module.__name__) must
   appear in __all__.

2. No dead names in __all__. Every name in __all__ must resolve to
   a real attribute of the module.

Imported-from-elsewhere objects (e.g. typing helpers, pyblp re-exports,
or cross-module dependency imports like `from .logit import
LogitBackend` inside nested_logit.py) are NOT required to be in __all__.
The filter used is `obj.__module__ == module.__name__`.

See the module-level ALLOWLIST comment below for the very short list
of cases where the filter doesn't suffice and we explicitly skip a
name.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Set, Tuple

import pytest

import pyRVtest


# ---------------------------------------------------------------------------
# Allowlist: module-name -> names intentionally NOT in __all__.
#
# Kept as a narrow escape hatch. The invariant filter (object.__module__ ==
# module.__name__) already excludes typing imports, pyblp re-exports, and
# cross-module dependency imports. Empty on the v0.4-refactor branch as of
# step 22; entries here must justify why the symbol is public-by-accident.
# ---------------------------------------------------------------------------
_ORPHAN_ALLOWLIST: dict[str, Set[str]] = {}


def _iter_pyrvtest_modules() -> list:
    """Return every pyRVtest module (top-level + recursive subpackages)."""
    mods = [pyRVtest]
    for info in pkgutil.walk_packages(pyRVtest.__path__, prefix='pyRVtest.'):
        try:
            mods.append(importlib.import_module(info.name))
        except ImportError as e:  # pragma: no cover - surfaced as a hard failure below
            raise AssertionError(f"Could not import {info.name!r}: {e}")
    return mods


def _public_defined_here(mod) -> Set[str]:
    """Return non-underscore attrs that are defined in `mod` (not re-imports).

    Excluded:
      - underscore-prefixed names (not public by PEP 8 convention)
      - submodules (namespace references, not symbols)
      - objects whose __module__ differs from mod.__name__ (imported
        from elsewhere — typing, pyblp, cross-module dep imports)
      - objects with no __module__ attribute (typing.TYPE_CHECKING is
        literally False, a bool; similar constants have no provenance)
    """
    name = mod.__name__
    result: Set[str] = set()
    for attr in dir(mod):
        if attr.startswith('_'):
            continue
        obj = getattr(mod, attr)
        if inspect.ismodule(obj):
            continue
        obj_module = getattr(obj, '__module__', None)
        if obj_module is None:
            continue
        if obj_module != name:
            continue
        result.add(attr)
    return result


# Modules with __all__ are the ones under audit.
_ALL_MODULES = [m for m in _iter_pyrvtest_modules() if '__all__' in m.__dict__]

# Parametrize by dotted module name for readable test output.
_MODULE_PARAMS: list[Tuple[str, object]] = [(m.__name__, m) for m in _ALL_MODULES]


@pytest.mark.parametrize('module_name,module', _MODULE_PARAMS, ids=[n for n, _ in _MODULE_PARAMS])
def test_no_orphan_public_names(module_name, module):
    """Invariant 1: every public symbol defined here is in __all__.

    "Public" = non-underscore attribute whose object.__module__ matches
    this module (i.e. defined in this file, not imported from elsewhere).
    """
    declared = set(module.__all__)
    allowlisted = _ORPHAN_ALLOWLIST.get(module_name, set())
    defined_here = _public_defined_here(module)
    orphans = defined_here - declared - allowlisted
    assert not orphans, (
        f"Public names defined in {module_name!r} are missing from __all__: "
        f"{sorted(orphans)}. Either add them to __all__, rename them with a "
        f"leading underscore, or (if they're intentionally public-by-accident "
        f"for some design reason) add them to _ORPHAN_ALLOWLIST in "
        f"tests/test_public_api_pin.py with a justification."
    )


@pytest.mark.parametrize('module_name,module', _MODULE_PARAMS, ids=[n for n, _ in _MODULE_PARAMS])
def test_no_dead_names_in_all(module_name, module):
    """Invariant 2: every name in __all__ resolves to a real attribute."""
    declared = list(module.__all__)
    actual = set(dir(module))
    dead = [n for n in declared if n not in actual]
    assert not dead, (
        f"Names in {module_name}.__all__ do not resolve to real attributes: "
        f"{dead}. Either remove them from __all__ or add the missing symbols."
    )


def test_audit_covers_expected_modules():
    """Sanity: the audit actually walks the package (not a no-op)."""
    covered = {m.__name__ for m in _ALL_MODULES}
    # Must at minimum cover the top-level package and every subpackage
    # __init__ that declared __all__ in earlier steps.
    required = {
        'pyRVtest',
        'pyRVtest.backends',
        'pyRVtest.backends.base',
        'pyRVtest.backends.labor',
        'pyRVtest.backends.labor.nested_logit_labor',  # v0.4 step 14b
        'pyRVtest.backends.logit',
        'pyRVtest.backends.nested_logit',
        'pyRVtest.backends.pyblp',
        'pyRVtest.backends.user',
        'pyRVtest.formulation',
        'pyRVtest.instruments',
        'pyRVtest.models',
        'pyRVtest.models._adapter',
        'pyRVtest.models.base',
        'pyRVtest.models.collusion',
        'pyRVtest.models.custom',
        'pyRVtest.models.labor',  # v0.4 step 14a
        'pyRVtest.models.mixed',
        'pyRVtest.models.standard',
        'pyRVtest.models.vertical',
        'pyRVtest.products',
        'pyRVtest.results',
        'pyRVtest.results.panel',
        'pyRVtest.results.results',
        'pyRVtest.solve',
        'pyRVtest.solve.demand_adjustment',
        'pyRVtest.solve.passthrough',
        'pyRVtest.exceptions',
    }
    missing = required - covered
    assert not missing, (
        f"Audit is not covering expected modules: {sorted(missing)}. If one "
        f"of these legitimately lost its __all__ declaration, remove it from "
        f"the required set in tests/test_public_api_pin.py."
    )
