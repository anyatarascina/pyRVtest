"""Conduct testing results subpackage.

v0.4 step 9: ``Progress`` and ``ProblemResults`` now live in
``pyRVtest/results/results.py``. This ``__init__`` re-exports them so
existing imports (``from pyRVtest.results import ProblemResults``)
continue to work unchanged. Downstream step 10 will add a sibling
``panel.py`` for multi-problem aggregation.
"""

from .results import Progress, ProblemResults

__all__ = ['Progress', 'ProblemResults']
