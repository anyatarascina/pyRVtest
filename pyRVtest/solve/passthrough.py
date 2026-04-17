"""Stand-alone ``build_passthrough`` public helper.

v0.4 step 11. Surfaces the Villas-Boas (2007) passthrough-matrix
construction (currently :func:`pyRVtest.markups.construct_passthrough_matrix`
and :func:`pyRVtest.markups._construct_passthrough_from_hessian`) as a
user-callable diagnostic decoupled from :meth:`pyRVtest.Problem.solve`.

The public export ``build_passthrough`` wraps the internal helpers used
by :func:`pyRVtest.markups._compute_markups` during upstream-markup
construction for vertical models. It does NOT replace those helpers;
the solve pipeline still calls them directly. This module is the
user-facing entry point.
"""

from __future__ import annotations

from typing import Any, Dict, Hashable, Optional, Union

import numpy as np
from numpy.typing import NDArray

from ..markups import _construct_passthrough_from_hessian
from ..models.vertical import Vertical


__all__ = ['build_passthrough']


_NDArray = NDArray[Any]


def build_passthrough(
        problem: Any,
        model_index: int,
        market_id: Optional[Hashable] = None,
) -> Union[_NDArray, Dict[Hashable, _NDArray]]:
    """Compute the Villas-Boas (2007) passthrough matrix for a vertical model.

    This is the public entry point for the passthrough-matrix math used
    internally by :meth:`pyRVtest.Problem.solve` when a vertical model is
    present. It wraps :func:`pyRVtest.markups._construct_passthrough_from_hessian`
    so users can inspect the matrix without running the full testing
    pipeline.

    Parameters
    ----------
    problem : pyRVtest.Problem
        An initialized :class:`pyRVtest.Problem` whose model at
        ``model_index`` is a :class:`pyRVtest.Vertical` instance (i.e.,
        has a non-``None`` upstream conduct). Otherwise passthrough is
        undefined.
    model_index : int
        Which entry in ``problem._models`` to use. Must be in
        ``[0, len(problem._models))``.
    market_id : hashable, optional
        If ``None`` (default), return ``{t: passthrough_matrix_t}`` for
        every market ``t`` in ``problem.unique_market_ids``. Otherwise
        return the single ``(J_t, J_t)`` matrix for that market.

    Returns
    -------
    ndarray or dict
        Passthrough matrix per Villas-Boas (2007). Shape ``(J_t, J_t)``
        for a specific ``market_id``, or ``{market_id: (J_t, J_t)}``
        across all markets.

    Raises
    ------
    ValueError
        If the model at ``model_index`` has no upstream conduct (not
        vertical); if ``market_id`` is not in the Problem's
        ``unique_market_ids``; if ``model_index`` is out of range; or
        if the Problem's demand backend does not provide a Hessian
        (e.g., ``UserSuppliedBackend`` without ``hessian_fn``).
    """
    # --- validate model_index ---
    n_models = len(problem._models)
    if not (0 <= model_index < n_models):
        raise ValueError(
            f"model_index={model_index} out of range; problem has "
            f"{n_models} model(s). Valid indices are 0..{n_models - 1}."
        )

    # --- validate model is vertical ---
    candidate = problem._models[model_index]
    models_upstream = problem.models['models_upstream']
    is_vertical = (
        isinstance(candidate, Vertical)
        or models_upstream[model_index] is not None
    )
    if not is_vertical:
        raise ValueError(
            f"model_index={model_index} refers to a non-vertical model "
            f"({type(candidate).__name__}); build_passthrough requires a "
            f"Vertical model (one with a non-None upstream conduct)."
        )

    # --- validate market_id if provided ---
    unique_market_ids = problem.unique_market_ids
    if market_id is not None:
        if not np.any(unique_market_ids == market_id):
            raise ValueError(
                f"market_id={market_id!r} is not in problem.unique_market_ids. "
                f"Valid market ids: {list(unique_market_ids)}."
            )
        markets_to_compute = [market_id]
    else:
        markets_to_compute = list(unique_market_ids)

    # --- compute downstream markups once (backend-routed) ---
    _, markups_downstream, _ = problem._perturb_and_build_markups()
    markups_down = markups_downstream[model_index]

    # --- guard against backends without a Hessian (e.g., bare UserSuppliedBackend) ---
    backend = problem._demand_backend
    if backend is None:
        raise ValueError(
            "problem._demand_backend is None; build_passthrough requires "
            "a demand backend (supply demand_params or demand_results to Problem)."
        )

    ownership_downstream = problem.models['ownership_downstream'][model_index]
    product_market_ids = problem.products.market_ids.flatten()

    results: Dict[Hashable, _NDArray] = {}
    for t in markets_to_compute:
        index_t = np.where(product_market_ids == t)[0]

        # Jacobian: backend strips NaN-padded columns on the per-market call.
        D_t = backend.compute_jacobian(market_id=t)

        # Hessian: required for vertical passthrough.
        d2s_dp2_t = backend.compute_hessian(market_id=t)
        if d2s_dp2_t is None:
            raise ValueError(
                f"problem._demand_backend.compute_hessian(market_id={t!r}) "
                f"returned None; vertical-integration passthrough requires a "
                f"demand backend that provides a Hessian. Supply `hessian_fn` "
                f"to UserSuppliedBackend or use one of the built-in backends."
            )

        # Ownership: slice to this market and drop NaN-padded columns.
        ownership_t = ownership_downstream[index_t]
        ownership_t = ownership_t[:, ~np.isnan(ownership_t).all(axis=0)]

        # Downstream markups for this market.
        markups_t = markups_down[index_t]

        passthrough_t = _construct_passthrough_from_hessian(
            d2s_dp2_t, D_t, ownership_t, markups_t
        )
        results[t] = passthrough_t

    if market_id is not None:
        return results[market_id]
    return results
