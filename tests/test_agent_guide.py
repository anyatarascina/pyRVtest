"""v0.4 step 23: test the show_agent_guide() exporter.

Acceptance criteria from `.claude/plans/v0.4-refactor.md` §5 step 23:

> Write `AGENTS.md` (root) + `docs/agent_guide.md` + `docs/custom_demand.md`.
> Export `pyRVtest.show_agent_guide()`. Tests: Docs build +
> show_agent_guide() returns expected content.

This file exercises the runtime exporter, not the docs build (which
has its own tox env). The invariants checked:

1. `pyRVtest.show_agent_guide()` runs without error.
2. The output contains the strings "pyRVtest" and
   "class-based ConductModel" (sanity check that content was printed).
3. The output length is > 500 characters (catches a regression where
   the function silently prints nothing).
4. `show_agent_guide` is in `pyRVtest.__all__`.
5. The helper reads the real AGENTS.md when it is present.
6. The fallback path fires cleanly when AGENTS.md is absent.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from unittest.mock import patch

import pytest

import pyRVtest
from pyRVtest import _agent_guide


# ---------------------------------------------------------------------------
# 1. The exporter runs and prints something useful.
# ---------------------------------------------------------------------------

def _capture_show_agent_guide() -> str:
    """Run `pyRVtest.show_agent_guide()` and return whatever went to stdout."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        pyRVtest.show_agent_guide()
    return buf.getvalue()


def test_show_agent_guide_runs_without_error():
    """Smoke test: the function returns cleanly."""
    out = _capture_show_agent_guide()
    # If the fallback and the real file both print nothing we'd hit
    # the length assertion below; this one just confirms no exception.
    assert out is not None


def test_show_agent_guide_contains_expected_strings():
    """Output mentions the package name and the class-based API."""
    out = _capture_show_agent_guide()
    assert 'pyRVtest' in out, (
        "Agent guide output is missing the package name 'pyRVtest'; "
        "something is off with either AGENTS.md or the fallback."
    )
    assert 'class-based ConductModel' in out, (
        "Agent guide output is missing the marker phrase "
        "'class-based ConductModel'. Either AGENTS.md was edited to drop "
        "this reference (update the test) or the fallback text is "
        "stale."
    )


def test_show_agent_guide_output_length_sanity():
    """Output is substantive (> 500 chars), not a truncated stub."""
    out = _capture_show_agent_guide()
    assert len(out) > 500, (
        f"Agent guide output is only {len(out)} characters; expected "
        f">500. Either AGENTS.md is missing and the fallback was "
        f"truncated, or show_agent_guide() silently printed nothing."
    )


# ---------------------------------------------------------------------------
# 2. Public API integration.
# ---------------------------------------------------------------------------

def test_show_agent_guide_in_package_all():
    """`show_agent_guide` is in the top-level `pyRVtest.__all__`."""
    assert 'show_agent_guide' in pyRVtest.__all__, (
        "show_agent_guide must be declared in pyRVtest.__all__ so the "
        "public-API audit in test_public_api_pin.py sees it."
    )


def test_show_agent_guide_in_module_all():
    """`show_agent_guide` is in `pyRVtest._agent_guide.__all__`."""
    assert 'show_agent_guide' in _agent_guide.__all__


def test_show_agent_guide_is_callable():
    """The re-exported symbol is the real function."""
    assert callable(pyRVtest.show_agent_guide)
    assert pyRVtest.show_agent_guide is _agent_guide.show_agent_guide


# ---------------------------------------------------------------------------
# 3. File-resolution logic.
# ---------------------------------------------------------------------------

def test_locate_agents_md_finds_real_file():
    """In a source checkout, `_locate_agents_md()` finds AGENTS.md."""
    path = _agent_guide._locate_agents_md()
    # On the source checkout used for the test run, the file is
    # present. If it is missing, that is itself a failure worth
    # surfacing (step 23 ships AGENTS.md as a required deliverable).
    assert path is not None, (
        "AGENTS.md was not found at the expected repo-root location. "
        "Either the file is missing or the path resolution logic in "
        "pyRVtest/_agent_guide.py is broken."
    )
    assert path.is_file()
    assert path.name == 'AGENTS.md'


def test_fallback_fires_when_agents_md_missing():
    """If `_locate_agents_md` returns None, the fallback text is printed."""
    with patch.object(_agent_guide, '_locate_agents_md', return_value=None):
        out = _capture_show_agent_guide()
    assert 'pyRVtest agent guide' in out
    # Fallback should still mention the class-based API so the
    # content-presence test above passes in both branches.
    assert 'class-based ConductModel' in out
    # And still be > 500 chars.
    assert len(out) > 500


def test_fallback_fires_on_oserror(tmp_path):
    """If AGENTS.md exists but fails to read, we degrade to the fallback."""
    # Point the resolver at a path that exists but is unreadable.
    # On most filesystems, a directory posing as a file triggers
    # read_text -> IsADirectoryError, which is a subclass of OSError.
    fake = tmp_path / "AGENTS.md"
    fake.mkdir()
    with patch.object(_agent_guide, '_locate_agents_md', return_value=fake):
        out = _capture_show_agent_guide()
    assert 'pyRVtest agent guide' in out


# ---------------------------------------------------------------------------
# 4. Real-file content sanity check.
# ---------------------------------------------------------------------------

def test_agents_md_file_has_required_sections():
    """The real AGENTS.md contains the marker sections the guide promises."""
    path = _agent_guide._locate_agents_md()
    if path is None:  # pragma: no cover - covered by the resolver test
        pytest.skip("AGENTS.md not present in this checkout")
    text = path.read_text(encoding='utf-8')
    # Spot-check the top-level section headings documented in the
    # step-23 spec, not the full outline. Changes to AGENTS.md that
    # affect these headings should also update this test.
    for section in (
        'What pyRVtest is',
        'Architecture at a glance',
        'Deprecation policy',
        'Running the test suite',
        'What NOT to change casually',
        'Where to start for common tasks',
    ):
        assert section in text, (
            f"AGENTS.md is missing the section header {section!r}. "
            f"If this header was deliberately renamed, update "
            f"tests/test_agent_guide.py to match."
        )
