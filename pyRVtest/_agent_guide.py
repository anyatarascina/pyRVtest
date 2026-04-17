"""Agent-guide exporter (v0.4 step 23).

Exposes :func:`show_agent_guide`, a small helper that prints the
contents of ``AGENTS.md`` at the repository root to stdout. Target
audience: AI coding assistants (Claude, GPT, etc.) that are about to
extend or modify pyRVtest and want a quick orientation without
reverse-engineering the commit history.

If the ``AGENTS.md`` file cannot be located (for example when pyRVtest
is installed from a wheel that didn't include the root-level markdown
file), the function falls back to a short hard-coded message pointing
at the online documentation.

See also: ``docs/agent_guide.rst`` (longer narrative walkthrough,
available via the rendered Sphinx docs).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


__all__ = ['show_agent_guide']


# Fallback text if AGENTS.md is not found on disk. Keeps the shape of
# the guide visible even in environments where the repo-root markdown
# file wasn't installed alongside the package.
_FALLBACK = """pyRVtest agent guide
====================

The full AGENTS.md file that normally backs this function could not be
located on disk. This typically means pyRVtest was installed from a
wheel that did not include the repository root. The rendered guide is
available online at the project documentation under 'Agent guide'; the
key points are:

- pyRVtest is a Python package for testing firm conduct models using
  the Rivers-Vuong framework. v0.4 introduces a class-based ConductModel
  API (Bertrand, Cournot, Monopoly, PerfectCompetition,
  MixCournotBertrand, PartialCollusion, CustomConductModel, Vertical)
  and a backend-generic demand interface (DemandBackend +
  SupportsDemandAdjustment protocols).
- Legacy ModelFormulation and demand_params['sigma'] are deprecated
  in v0.4 and v0.5, removed in v0.6.
- Tests live under tests/; run with `python -m pytest tests/ -q`.
- The authoritative migration plan is .claude/plans/v0.4-refactor.md
  in the source repository.

See docs/agent_guide.rst for the full walkthrough.
"""


def _locate_agents_md() -> Optional[Path]:
    """Return the path to AGENTS.md if present at the repo root.

    The file lives at ``<repo_root>/AGENTS.md``; this module lives at
    ``<repo_root>/pyRVtest/_agent_guide.py``. So the candidate path is
    ``Path(__file__).parent.parent / 'AGENTS.md'``.

    Returns ``None`` if the file is absent (for example when installed
    from a wheel that didn't ship the root markdown).
    """
    candidate = Path(__file__).resolve().parent.parent / 'AGENTS.md'
    if candidate.is_file():
        return candidate
    return None


def show_agent_guide() -> None:
    """Print the pyRVtest agent guide to stdout.

    The guide is the contents of the ``AGENTS.md`` file at the
    repository root. If that file is not available (for example in a
    stripped-down wheel install), a short fallback pointing at the
    online documentation is printed instead.

    Examples
    --------
    >>> import pyRVtest
    >>> pyRVtest.show_agent_guide()  # doctest: +SKIP
    # AGENTS.md
    ...
    """
    path = _locate_agents_md()
    if path is None:
        print(_FALLBACK)
        return
    try:
        text = path.read_text(encoding='utf-8')
    except OSError:
        print(_FALLBACK)
        return
    print(text)
