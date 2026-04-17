"""Conduct testing results subpackage.

v0.4 step 9: ``Progress`` and ``ProblemResults`` now live in
``pyRVtest/results/results.py``. This ``__init__`` re-exports them so
existing imports (``from pyRVtest.results import ProblemResults``)
continue to work unchanged.

v0.4 step 10: ``PanelResults`` lives in ``pyRVtest/results/panel.py``
and aggregates a mapping of :class:`ProblemResults` (keyed by e.g.
``(market_id, year)``) into a single panel-level view.
"""

from .panel import PanelResults
from .results import Progress, ProblemResults

__all__ = ['Progress', 'ProblemResults', 'PanelResults']
