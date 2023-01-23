"""Economy-level structuring of abstract BLP problem results."""

import abc
from typing import Any, Optional, TYPE_CHECKING

import numpy as np

from pyblp.utilities.basics import Array, StringRepresentation


# only import objects that create import cycles when checking types
if TYPE_CHECKING:
    from ..economies.problem import ProblemEconomy


class Results(abc.ABC, StringRepresentation):
    """Abstract results of a solved BLP problem."""

    problem: 'ProblemEconomy'

    def __init__(
            self, problem: 'ProblemEconomy') -> None:
        """Store the underlying problem and parameter information."""
        self.problem = problem

    def _select_market_ids(self, market_id: Optional[Any] = None) -> Array:
        """Select either a single market ID or all unique IDs."""
        if market_id is None:
            return self.problem.unique_market_ids
        if market_id in self.problem.unique_market_ids:
            return np.array(market_id, np.object)
        raise ValueError(f"market_id must be None or one of {list(sorted(self.problem.unique_market_ids))}.")
