"""Helpers for v0.4 Step 0b snapshot regression suite.

A snapshot captures numerical outputs of a pinned DGP's `.solve()` call
(markups, marginal_cost, taus, g, Q, TRV, F, MCS_pvalues) in a JSON file.
Every later migration step must reproduce the snapshot to atol=1e-10, or
document a deliberate change per the decision rule in
.claude/plans/v0.4-refactor.md §5:

  * delta <= 1e-12         => auto-update (numerical noise)
  * 1e-12 < delta <= 1e-7  => update only with explicit commit-message
                              justification identifying the deliberate source
  * delta > 1e-7           => BLOCK MERGE, investigate root cause

Regeneration: set env var REGENERATE_SNAPSHOTS=1 when running pytest to
overwrite the JSON files with current outputs instead of comparing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SNAPSHOT_DIR = Path(__file__).resolve().parent / 'snapshots'
SNAPSHOT_TOLERANCE = 1e-10
REGENERATE_ENV_VAR = 'REGENERATE_SNAPSHOTS'

# Which ProblemResults attributes get snapshotted. See §5 of the v0.4 plan
# for the canonical list. Snapshot shape is either ndarray or list[ndarray].
_SNAPSHOTTED_ATTRIBUTES = (
    'markups',
    'marginal_cost',
    'taus',
    'g',
    'Q',
    'TRV',
    'F',
    'MCS_pvalues',
)


def _should_regenerate() -> bool:
    """True if user set REGENERATE_SNAPSHOTS=1 (overwrite instead of compare)."""
    return os.environ.get(REGENERATE_ENV_VAR, '').strip() in ('1', 'true', 'True', 'yes')


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy arrays / scalars to JSON-compatible types.

    NaN and Inf are preserved as `float('nan')` / `float('inf')` inside lists;
    `json.dumps(..., allow_nan=True)` (the Python default) serializes them as
    the non-standard tokens `NaN` / `Infinity`. Python's `json.loads` reads
    them back as floats. Diff-friendly across editors.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    return obj


def extract_results(results) -> dict:
    """Pull snapshot-worthy fields out of a ProblemResults into a plain dict."""
    out = {}
    for attr in _SNAPSHOTTED_ATTRIBUTES:
        if not hasattr(results, attr):
            continue
        out[attr] = _to_jsonable(getattr(results, attr))
    return out


def _compare_nested(expected: Any, actual: Any, atol: float, path: str = '') -> list[str]:
    """Return list of human-readable mismatch descriptions; empty list means match."""
    errors: list[str] = []

    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            errors.append(f"{path}: length mismatch ({len(expected)} expected, {len(actual)} actual)")
            return errors
        for i, (e, a) in enumerate(zip(expected, actual)):
            errors.extend(_compare_nested(e, a, atol, path=f"{path}[{i}]"))
        return errors

    # Leaf: scalar comparison (NaN treated equal to NaN)
    try:
        e_val = float(expected) if expected is not None else None
        a_val = float(actual) if actual is not None else None
    except (TypeError, ValueError):
        if expected != actual:
            errors.append(f"{path}: non-numeric mismatch (expected={expected!r}, actual={actual!r})")
        return errors

    if e_val is None or a_val is None:
        if e_val != a_val:
            errors.append(f"{path}: None mismatch (expected={expected}, actual={actual})")
        return errors

    if np.isnan(e_val) and np.isnan(a_val):
        return errors
    if np.isinf(e_val) and np.isinf(a_val) and (e_val > 0) == (a_val > 0):
        return errors

    delta = abs(e_val - a_val)
    if not np.isfinite(delta) or delta > atol:
        errors.append(
            f"{path}: delta {delta:.3e} > atol {atol:.1e} "
            f"(expected={e_val:.15g}, actual={a_val:.15g})"
        )
    return errors


def assert_snapshot(name: str, results, atol: float = SNAPSHOT_TOLERANCE) -> None:
    """Compare current results to a pinned snapshot file, or regenerate if asked.

    Parameters
    ----------
    name : str
        Logical snapshot name, e.g. "analytical_base". Written/read at
        `tests/snapshots/{name}.json`.
    results : ProblemResults
        Output of `Problem.solve(...)`.
    atol : float
        Absolute tolerance per element. Default 1e-10 (see plan §5 decision rule).

    Behavior
    --------
    - If REGENERATE_SNAPSHOTS is set: serialize current results to JSON and skip
      the comparison. Use sparingly — every snapshot change requires a commit
      message explaining the source (see decision rule).
    - If the snapshot file does not exist: write it and xfail once so the test
      run is not silently green. Subsequent runs compare.
    - Else: compare element-wise, failing on first delta > atol.
    """
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / f"{name}.json"
    current = extract_results(results)

    if _should_regenerate():
        with open(path, 'w') as fh:
            json.dump(current, fh, indent=2, allow_nan=True)
        # Skipping the assertion: the user asked to regenerate. A separate
        # comparison run (without REGENERATE_SNAPSHOTS) is required for CI.
        return

    if not path.exists():
        # First-time write: serialize and fail with a clear message so the
        # caller knows to re-run normally and commit the new snapshot.
        with open(path, 'w') as fh:
            json.dump(current, fh, indent=2, allow_nan=True)
        raise AssertionError(
            f"Snapshot {path.relative_to(SNAPSHOT_DIR.parent)} did not exist. "
            f"It has been created. Re-run pytest to verify it matches; commit "
            f"the new file with a message documenting the source DGP."
        )

    with open(path) as fh:
        expected = json.load(fh)

    # Check that the structure keys match
    missing = set(expected.keys()) - set(current.keys())
    extra = set(current.keys()) - set(expected.keys())
    if missing or extra:
        raise AssertionError(
            f"Snapshot {name}: key mismatch. missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )

    # Element-wise comparison per attribute
    all_errors: list[str] = []
    for key in sorted(expected.keys()):
        errs = _compare_nested(expected[key], current[key], atol, path=key)
        all_errors.extend(errs)
        if len(all_errors) > 10:
            all_errors.append(f"... (additional mismatches truncated; rerun with more atol to see all)")
            break

    if all_errors:
        raise AssertionError(
            f"Snapshot {name} mismatch ({len(all_errors)} differences):\n  "
            + "\n  ".join(all_errors)
            + f"\n\nIf the change is deliberate, update the snapshot by running:\n"
            f"  REGENERATE_SNAPSHOTS=1 pytest tests/test_snapshots.py::<name>\n"
            f"and commit with a message identifying the source per plan §5 decision rule."
        )
