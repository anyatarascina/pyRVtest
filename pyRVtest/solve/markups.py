"""Markups-assembly stage.

v0.4 step 8b extraction. Hosts :func:`compute`, the stage that builds
per-model markups by dispatching to
:func:`pyRVtest.markups._compute_markups` with the ``Problem``'s
constructed demand backend. The returned tuple
``(markups, markups_downstream, markups_upstream)`` matches what
``_compute_markups`` has always returned — no math change relative to
the pre-step-8 call site.

The module-level ``pyRVtest.markups`` module (which still owns
``build_markups``, ``_compute_markups``, ``build_ownership`` etc.) is
the math; this stage is the thin per-:class:`Problem` orchestration
hook. Renaming the math module was deferred — it still carries the
published public API (``pyRVtest.build_markups``).
"""

from __future__ import annotations

import logging
from typing import Any, List, Tuple

from numpy.typing import NDArray

from ..markups import _compute_markups


__all__ = ['compute']


_NDArray = NDArray[Any]


logger = logging.getLogger(__name__)


def compute(problem: Any) -> Tuple[List[_NDArray], List[_NDArray], List[_NDArray]]:
    """Build per-model markups using the Problem's demand backend.

    Moved from ``Problem._perturb_and_build_markups`` in v0.4 step 8b.
    No behavioral change.

    Parameters
    ----------
    problem
        The :class:`pyRVtest.Problem` instance. Accessed read-only for
        ``products``, ``demand_results``, ``models``, and
        ``_demand_backend``.

    Returns
    -------
    tuple of list of ndarray
        ``(markups, markups_downstream, markups_upstream)`` — each a
        length-``M`` list of ``(N, 1)`` arrays. ``markups[m]`` is the
        total implied markup for model ``m`` (downstream + upstream
        where applicable).

    Notes
    -----
    v0.4 step 4g: after step 4f deleted the inline demand-adjustment
    methods that mutated ``self.demand_results._sigma`` / ``._pi`` /
    ``._beta`` / ``._rho`` directly, nothing mutates demand state
    behind the backend's cache. Routing through
    ``problem._demand_backend`` is therefore safe: the cache is
    invalidated correctly by ``backend.perturbed(...)`` everywhere
    it's used. Legacy ``demand_jacobian`` / ``demand_alpha`` /
    ``demand_sigma`` kwargs on ``_compute_markups`` are gone;
    ``compute_jacobian`` / ``compute_hessian`` on the backend subsume
    them.
    """
    return _compute_markups(
        problem.products, problem.demand_results, problem.models["models_downstream"],
        problem.models["ownership_downstream"], problem.models["models_upstream"],
        problem.models["ownership_upstream"], problem.models["vertical_integration"],
        problem.models["custom_model_specification"], problem.models["user_supplied_markups"],
        problem.models["mix_flag"], demand_backend=problem._demand_backend,
    )
