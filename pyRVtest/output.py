"""Output formatting and legacy ``output()`` shim.

Status messages from pyRVtest go through per-module
:mod:`logging` loggers (``pyRVtest.problem``,
``pyRVtest.backends.logit``, etc.). :func:`pyRVtest.output.output` is a
compatibility shim that routes its argument through ``logger.info`` on
the ``pyRVtest.output`` logger and emits a :class:`DeprecationWarning`
on first call. It exists solely so that ``from pyRVtest.output import
output`` keeps working through the v0.6 deprecation window; new code
should not call it. :func:`format_table` is the long-term formatting
helper.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Container, List, Optional, Sequence


# Silence the deprecation shim with
# ``logging.getLogger("pyRVtest.output").setLevel(logging.WARNING)``.
logger = logging.getLogger(__name__)


# Fire the DeprecationWarning only once per Python session.
_output_shim_deprecation_warned: bool = False


def output(message: Any) -> None:
    """Legacy progress-message helper. Deprecated in v0.4; removed in v0.6.

    Forwards ``str(message)`` through the ``pyRVtest.output`` logger at
    INFO level. On the first call in any Python session it also emits a
    :class:`DeprecationWarning` pointing at the replacement idiom.

    Parameters
    ----------
    message : Any
        Object whose ``str(...)`` representation will be logged. This
        matches the v0.3 pyblp ``output`` call signature so existing user
        code continues to work.
    """
    global _output_shim_deprecation_warned
    if not _output_shim_deprecation_warned:
        warnings.warn(
            "pyRVtest.output.output() is deprecated and will be removed in "
            "v0.6. pyRVtest now uses the standard logging module. To see "
            "progress messages, call logging.basicConfig(level=logging.INFO) "
            "once at program start. To silence a specific subsystem use "
            "logging.getLogger('pyRVtest.problem').setLevel(logging.WARNING). "
            "See docs/agent_guide.rst for the logger layout.",
            DeprecationWarning,
            stacklevel=2,
        )
        _output_shim_deprecation_warned = True
    logger.info(str(message))


def format_table(
        header: Sequence[Any], subheader: Sequence[Any], *data: Sequence[Any], title: Optional[str] = None,
        include_notes: bool = False, include_border: bool = True, include_header: bool = True,
        include_subheader: bool = True, line_indices: Container[int] = (),
        extra_notes: Optional[Sequence[Sequence[str]]] = None) -> str:
    """Format table information as a string, which has fixed widths, vertical lines after any specified indices, and
    optionally a title, border, header, subheader, and F-stat significance notes.

    Parameters
    ----------
    extra_notes : optional sequence of sequence of str
        Additional note lines appended after the standard significance notes
        when ``include_notes`` is True. Each inner sequence is one row.
        Used by ``ProblemResults._format_results_tables`` to attach the
        F-stat reliability footer; passed empty / None for tables that
        don't need it.
    """

    # construct the header rows
    row_index = -1
    header_rows: List[List[str]] = []
    header = [[c] if isinstance(c, str) else c for c in header]
    while True:
        header_row = ["" if len(c) < -row_index else c[row_index] for c in header]
        if not any(header_row):
            break
        header_rows.insert(0, header_row)
        row_index -= 1

    # construct the sub-header rows
    row_index = -1
    subheader_rows: List[List[str]] = []
    subheader = [[c] if isinstance(c, str) else c for c in subheader]
    while True:
        subheader_row = ["" if len(c) < -row_index else c[row_index] for c in subheader]
        if not any(subheader_row):
            break
        subheader_rows.insert(0, subheader_row)
        row_index -= 1

    # construct the data rows
    data_rows = [[str(c) for c in r] + [""] * (len(header) - len(r)) for r in data]

    # compute column widths
    widths = []
    for column_index in range(len(header)):
        widths.append(max(len(r[column_index]) for r in header_rows + subheader_rows + data_rows))

    # build the template
    template = "  " .join("{{:^{}}}{}".format(w, "  |" if i in line_indices else "") for i, w in enumerate(widths))
    template_notes = "  " .join("{{:^{}}}{}".format(w, "  " if i in line_indices else "") for i, w in enumerate(widths))

    # build the table
    lines = []
    if title is not None:
        lines.append(f"{title}:")
    if include_border:
        lines.append("=" * len(template.format(*[""] * len(widths))))
    if include_header:
        lines.extend([template.format(*r) for r in header_rows])
        lines.append(template.format(*("-" * w for w in widths)))
    if include_subheader:
        lines.extend([template.format(*r) for r in subheader_rows])
        lines.append(template.format(*("-" * w for w in widths)))
    lines.extend([template.format(*r) for r in data_rows])
    if include_border:
        lines.append("=" * len(template.format(*[""] * len(widths))))
    if include_notes:
        notes: List[List[str]] = []
        # TRV significance markers (Phase 3 of feat/f-reliability).
        notes.append(
            ['TRV significance (two-sided, asymptotic):']
        )
        notes.append(
            ['  *, **, or *** indicate |TRV| > 1.64, 1.96, 2.58 (10%, 5%, 1% two-sided)']
        )
        # F-stat significance markers (size symbols moved from `*` to `†`
        # in Phase 3 to free `*` for TRV; power symbols unchanged).
        notes.append(
            ['F-stat significance (DMSS):']
        )
        notes.append(
            ['  †, ††, or ††† indicate F > cv for a worst-case size of 0.125, 0.10, and 0.075 given d_z and rho']
        )
        notes.append(
            ['  ^, ^^, or ^^^ indicate F > cv for a best-case power of 0.50, 0.75, and 0.95 given d_z and rho']
        )
        if extra_notes:
            for row in extra_notes:
                notes.append(list(row))
        notes_rows = [[str(c) for c in r] + [""] * (len(header) - len(r)) for r in notes]
        lines.extend([template_notes.format(*r) for r in notes_rows])
        lines.append("=" * len(template_notes.format(*[""] * len(widths))))
    return "\n".join(lines)
