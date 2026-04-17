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

from ..exceptions import HessianUnavailableError
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

    Examples
    --------
    >>> from pyRVtest import build_passthrough  # doctest: +SKIP
    >>> # Requires a constructed ``pyRVtest.Problem`` whose ``models=[...]`` list
    >>> # contains a ``Vertical(...)`` entry. See docs/tutorial.rst for the
    >>> # end-to-end setup. A representative call is:
    >>> G = build_passthrough(problem, model_index=0, market_id=5)  # doctest: +SKIP
    """
    # --- validate model_index ---
    n_models = len(problem._models)
    if not (0 <= model_index < n_models):
        raise ValueError(
            f"Expected model_index to be in [0, {n_models}) for this Problem "
            f"(has {n_models} model(s)). "
            f"Received model_index={model_index} (out of range). "
            f"Fix: pass an integer in 0..{n_models - 1}."
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
            f"Expected the model at model_index={model_index} to be a "
            f"Vertical model (i.e., with a non-None upstream conduct) because "
            f"passthrough is only defined for vertical structures. "
            f"Received a non-vertical model ({type(candidate).__name__}). "
            f"Fix: point model_index at a Vertical(...) entry, or skip "
            f"build_passthrough for this model."
        )

    # --- validate market_id if provided ---
    unique_market_ids = problem.unique_market_ids
    if market_id is not None:
        if not np.any(unique_market_ids == market_id):
            raise ValueError(
                f"Expected market_id to appear in problem.unique_market_ids. "
                f"Received market_id={market_id!r}, which is not in "
                f"problem.unique_market_ids={list(unique_market_ids)}. "
                f"Fix: pass a market id from problem.unique_market_ids, or "
                f"omit market_id to compute passthrough for all markets."
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
            "Expected Problem to carry a constructed demand backend so that "
            "compute_jacobian / compute_hessian can run. "
            "Received problem._demand_backend=None (neither demand_params nor "
            "demand_results was supplied at Problem construction). "
            "Fix: pass demand_params={...} or demand_results=<pyblp.ProblemResults> "
            "to Problem(...)."
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
            raise HessianUnavailableError(
                f"Expected the demand backend to provide a Hessian for "
                f"vertical-integration passthrough. "
                f"Received problem._demand_backend.compute_hessian(market_id="
                f"{t!r}) returned None. "
                f"Fix: supply `hessian_fn` to UserSuppliedBackend, or use a "
                f"built-in backend (PyBLPBackend, LogitBackend, NestedLogitBackend)."
            )

        # Ownership: slice to this market and drop NaN-padded columns.
        ownership_t = ownership_downstream[index_t]
        ownership_t = ownership_t[:, ~np.isnan(ownership_t).all(axis=0)]

        # Downstream markups for this market.
        markups_t = markups_down[index_t]

        # ``_construct_passthrough_from_hessian`` lives in legacy ``markups.py``
        # which is not strict-typed (see mypy.ini); treat as returning _NDArray.
        passthrough_t: _NDArray = _construct_passthrough_from_hessian(  # type: ignore[no-untyped-call]
            d2s_dp2_t, D_t, ownership_t, markups_t
        )
        results[t] = passthrough_t

    if market_id is not None:
        return results[market_id]
    return results
