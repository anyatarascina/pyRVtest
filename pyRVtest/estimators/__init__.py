"""In-package demand estimators (2SLS plain logit and one-level nested logit).

Users with simpler demand systems can run estimation here instead of
through PyBLP. The estimator returns a populated ``demand_params`` dict
that ``Problem(demand_params=...)`` consumes via the existing analytical-
backend path in ``Problem._construct_demand_backend``.

For BLP / random-coefficient demand, continue to use PyBLP and pass
``demand_results=pyblp_results``.
"""

from .logit import LogitEstimator
from .nested_logit import NestedLogitEstimator

__all__ = ['LogitEstimator', 'NestedLogitEstimator']
