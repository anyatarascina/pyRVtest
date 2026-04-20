"""Shared rendering helpers for the :mod:`pyRVtest.results` subpackage.

Internal module — not part of the public API. Currently hosts
``_dataframe_to_github_markdown`` used by both
:meth:`pyRVtest.results.results.ProblemResults.to_markdown` and
:meth:`pyRVtest.results.panel.PanelResults.to_markdown`.

Resolves Open Question 12 from ``.claude/plans/v0.4-refactor.md``:
before this module, the helper lived under ``results.results`` with an
underscore prefix and was cross-imported by ``results.panel``. Giving
the helper its own module keeps both callers above the
"sibling-reaches-into-sibling" line without expanding the public API.
"""

from __future__ import annotations

from typing import List, TYPE_CHECKING

import numpy as np

from .. import options

if TYPE_CHECKING:
    import pandas as pd


__all__: List[str] = ['_dataframe_to_github_markdown']


def _dataframe_to_github_markdown(frame: 'pd.DataFrame') -> str:
    """Render a DataFrame as a GitHub-flavored markdown pipe table.

    Kept out of :meth:`pandas.DataFrame.to_markdown` to avoid the
    optional ``tabulate`` dependency that pandas pulls in; GitHub-flavored
    markdown only needs pipes and hyphens with an empty-row guard.

    Parameters
    ----------
    frame : pd.DataFrame
        The frame to render. Floats are formatted via
        ``f"{val:.{pyRVtest.options.digits}g}"`` (default ``6`` significant
        digits); NaNs render as ``"NaN"``; booleans render as
        ``"True"`` / ``"False"``; everything else goes through ``str``.

    Returns
    -------
    str
        A multi-line markdown string. Trailing newline included.
    """
    columns = [str(c) for c in frame.columns]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body_lines: List[str] = []
    if len(frame) == 0:
        # GitHub renders zero-row tables correctly with just header + sep.
        return "\n".join([header, sep]) + "\n"
    digits = int(options.digits)
    float_fmt = f"{{val:.{digits}g}}"
    for _, row in frame.iterrows():
        cells: List[str] = []
        for col in columns:
            val = row[col]
            if isinstance(val, float):
                if np.isnan(val):
                    cells.append("NaN")
                else:
                    cells.append(float_fmt.format(val=val))
            elif isinstance(val, (bool, np.bool_)):
                cells.append("True" if bool(val) else "False")
            else:
                cells.append(str(val))
        body_lines.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *body_lines]) + "\n"
