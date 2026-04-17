"""Pin the mypy --strict invariant for the package.

v0.4 step 17. Runs ``mypy`` as a subprocess over the full ``pyRVtest/``
tree and asserts zero errors. Uses the project's ``mypy.ini``, whose
per-module ``[mypy-pyRVtest.<module>]`` sections govern which files
enforce ``strict = True`` and which legacy files (``problem.py``,
``markups.py``) remain lax with explicit ``disable_error_code`` lists.

Skipped automatically if mypy is not installed in the test environment.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


mypy = pytest.importorskip("mypy")  # noqa: F841 — just a presence check


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_mypy_strict_clean() -> None:
    """``mypy pyRVtest/ --strict`` must pass with zero errors.

    Runs mypy from the project root so the repo's ``mypy.ini`` is
    picked up. ``--strict`` is applied on top of the module-level
    strict / disable-error-code rules. The legacy ``problem.py`` and
    ``markups.py`` sections suppress specific error codes (documented
    in ``mypy.ini``); every other pyRVtest module should be cleanly
    strict-typed.

    Asserts returncode == 0 and "Success" in stdout so that accidental
    warnings (e.g., unused ignores) still surface as test failures.
    """
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "pyRVtest/", "--strict"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"mypy --strict failed with returncode={result.returncode}.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "Success" in result.stdout, (
        f"mypy --strict did not report Success.\nstdout:\n{result.stdout}"
    )
