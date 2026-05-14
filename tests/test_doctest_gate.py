"""Mirror the CI doctest gate inside the local test suite.

Added in rc15 after the rc14 audit caught a doctest-collection failure
in ``pyRVtest/solve/passthrough.py`` that the CI gate raised but the
local ``pytest tests/`` run did not. Running ``pytest --doctest-modules``
locally with the rest of the suite means a malformed docstring shows
up on developer machines, not first in CI.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_doctest_modules_clean() -> None:
    """``pytest --doctest-modules pyRVtest`` must pass.

    Mirrors the corresponding CI step in ``.github/workflows/ci.yml``.
    Common failure modes this catches:

    * A docstring example with an option directive on a comment-only
      line (``>>> # foo  # doctest: +SKIP``) — collection-time
      ``ValueError``.
    * A docstring example that no longer matches its computed output
      after a refactor — assertion failure.
    """
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--doctest-modules", "pyRVtest", "-q"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"pytest --doctest-modules failed with returncode={result.returncode}.\n"
        f"stdout:\n{result.stdout[-2000:]}\n"
        f"stderr:\n{result.stderr[-1000:]}"
    )
