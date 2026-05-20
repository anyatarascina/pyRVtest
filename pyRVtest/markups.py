"""Markup computation."""

import contextlib
import os
from pathlib import Path
import pickle
from typing import Any, Callable, Dict, Mapping, Optional, Union

import numpy as np
from numpy.linalg import inv
import pyblp
from pyblp.utilities.basics import Array, RecArray

from . import options


def build_ownership(
        product_data: Mapping, firm_ids_column_name: str,
        kappa_specification: Optional[Union[str, Callable[[Any, Any], float]]] = None) -> Array:
    r"""Build ownership matrices, :math:`O`.

    Thin wrapper around :func:`pyblp.build_ownership` that allows specifying which column contains firm IDs, rather
    than requiring a column named ``firm_ids``. This supports separate downstream and upstream ownership structures.

    Parameters
    ----------
    product_data : `structured array-like`
        Each row corresponds to a product. Must have a ``market_ids`` field.
    firm_ids_column_name : `str`
        Column in ``product_data`` with firm IDs. Ignored for special-case ``kappa_specification`` values.
    kappa_specification : `str or callable, optional`
        Passed directly to :func:`pyblp.build_ownership`. See that function for full documentation.

    Returns
    -------
    `ndarray`
        Stacked :math:`J_t \times J_t` ownership matrices for each market :math:`t`.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import pyRVtest
    >>> product_data = pd.DataFrame({
    ...     'market_ids': [0, 0, 0, 1, 1, 1],
    ...     'firm_ids':   [1, 1, 2, 1, 2, 2],
    ... })
    >>> O = pyRVtest.build_ownership(product_data, 'firm_ids')
    >>> O.shape
    (6, 3)
    """
    names = (
        product_data.dtype.names if hasattr(product_data, 'dtype') and product_data.dtype.names
        else product_data.columns
    )
    modified_data = {name: product_data[name] for name in names}
    if firm_ids_column_name is not None:
        modified_data['firm_ids'] = product_data[firm_ids_column_name]
    return pyblp.build_ownership(modified_data, kappa_specification)


def build_markups(
        model_formulations: Any, product_data: RecArray, pyblp_results: Mapping) -> Array:
    r"""Compute markups for one or more candidate conduct models.

    This is the main user-facing markup function. It accepts :class:`ModelFormulation` objects together with product
    data and a PyBLP demand results object, and returns implied markups for each model.

    Supported models include standard Bertrand, Cournot, monopoly, perfect competition, mixed Cournot/Bertrand,
    bilateral oligopoly (with optional vertical integration), and custom user-specified markup formulas.

    Parameters
    ----------
    model_formulations : `sequence of ModelFormulation`
        One or more :class:`ModelFormulation` instances that specify the conduct model(s) for which markups should be
        computed. At least one formulation must be provided.
    product_data : `structured array-like`
        Product-level data used for demand estimation. Must contain ``market_ids``, ``shares``, and ``prices``
        fields.
    pyblp_results : `structured array-like`
        Results object returned by ``pyblp.Problem.solve``. Used to compute demand Jacobians and Hessians.

    Returns
    -------
    `tuple[list, list, list]`
        Computed total markups, downstream markups, and upstream markups for each model. Each element of the
        returned lists is an ``(N, 1)`` array of markups stacked across markets.

    Examples
    --------
    >>> import pyRVtest  # doctest: +SKIP
    >>> # Requires a fitted pyblp.ProblemResults; see docs/tutorial.rst
    >>> # for the end-to-end setup. A representative call is:
    >>> model = pyRVtest.ModelFormulation(  # doctest: +SKIP
    ...     model_downstream='bertrand', ownership_downstream='firm_ids',
    ... )
    >>> markups, markups_down, markups_up = pyRVtest.build_markups(  # doctest: +SKIP
    ...     [model], product_data, pyblp_results,
    ... )
    """
    from .problem import Models  # local import avoids circular dependency
    if not hasattr(model_formulations, '__len__'):
        model_formulations = [model_formulations]
    models = Models(model_formulations, product_data)
    return _compute_markups(
        product_data, pyblp_results, models["models_downstream"], models["ownership_downstream"],
        models["models_upstream"], models["ownership_upstream"], models["vertical_integration"],
        models["custom_model_specification"], models["user_supplied_markups"], models["mix_flag"],
        constant_markup=models["constant_markup"],
    )


def _construct_passthrough_from_hessian(d2s_dp2_t, retailer_response_matrix, retailer_ownership_matrix, markups_t):
    """Construct passthrough matrix from a pre-computed Hessian (same formula as construct_passthrough_matrix)."""
    J = len(markups_t)
    g = np.zeros((J, J))
    for j in range(J):
        g[:, [j]] = (retailer_ownership_matrix * d2s_dp2_t[:, :, j]) @ markups_t
    H = np.transpose(retailer_ownership_matrix * retailer_response_matrix)
    G = retailer_response_matrix + H + g
    return inv(G) @ H


# ---------------------------------------------------------------------------
# rc10 perf: batched downstream-markup path.
#
# For the "simple" conducts whose markup formulas are a single per-market
# closed form with no upstream / mix / custom hook (bertrand, cournot,
# monopoly, perfect_competition, constant_markup), all markets in the
# same J-bucket can be processed in a constant number of LAPACK calls
# instead of one per market. Mirrors the rc8 batched central-difference
# work in solve/passthrough.py. Falls back to the per-market scalar
# loop for everything else.
# ---------------------------------------------------------------------------


_BATCHABLE_DOWNSTREAM = frozenset({
    'bertrand', 'cournot', 'monopoly', 'perfect_competition', 'constant_markup',
})


def _is_batchable_downstream_model(
        model_downstream: Optional[str],
        model_upstream: Optional[str],
        mix_flag: Optional[Array],
        custom_model_specification: Optional[Any],
) -> bool:
    """True iff the (downstream, upstream, mix, custom) tuple uses only
    pure per-market closed-form math and so can go down the batched path.

    The decision tree must match the conduct types listed in
    ``_BATCHABLE_DOWNSTREAM``. Anything else (vertical / mix /
    custom) falls back to the per-market scalar loop in
    ``_compute_markups`` so the existing math is preserved verbatim.
    """
    if model_downstream not in _BATCHABLE_DOWNSTREAM:
        return False
    if model_upstream is not None:
        return False
    if mix_flag is not None:
        return False
    if custom_model_specification is not None:
        return False
    return True


def _compute_batchable_downstream_markups(
        model_type: str,
        ownership_downstream_i: Optional[Array],
        constant_markup_i: Optional[Array],
        markets: Array,
        market_index_map: Dict[Any, Array],
        market_shares_map: Dict[Any, Array],
        market_response_map: Dict[Any, Array],
        N: int,
) -> Array:
    """Batched downstream-markups for one batchable conduct.

    Groups markets by J_t, stacks per-market (ownership, response,
    shares) into ``(M_g, J, J)`` / ``(M_g, J)`` arrays, dispatches one
    batched LAPACK call per J-group, then scatters results into a
    single ``(N, 1)`` output array.

    For ``perfect_competition`` / ``constant_markup`` the result is
    independent of (D, s) and is produced as a single one-shot copy
    without any per-J grouping.

    Numerically identical to the per-market scalar loop in
    :func:`evaluate_first_order_conditions` for the same model_type
    (same LAPACK routines, same input slices). Pinned with
    :class:`tests.test_compute_markups_direct` and verified via
    :func:`tests.test_compute_markups_direct._expected_*_markups`
    against hand-derived formulas to atol=1e-12.
    """
    out = np.zeros((N, 1), dtype=options.dtype)

    if model_type == 'perfect_competition':
        # Markup is identically zero across all rows.
        return out

    if model_type == 'constant_markup':
        # Per-product dollar markup is a model primitive (Dearing et al.
        # 2024 Example 7). The (N, 1) constant_markup column IS the
        # answer — no per-market computation needed.
        if constant_markup_i is None:
            raise ValueError(
                "Expected constant_markup to be supplied when "
                "model_type='constant_markup'. Received None — internal "
                "wiring bug."
            )
        cm = np.asarray(constant_markup_i, dtype=options.dtype)
        if cm.ndim == 1:
            cm = cm.reshape(-1, 1)
        out[:] = cm
        return out

    # bertrand / cournot / monopoly: group markets by J_t, batch.
    if ownership_downstream_i is None and model_type != 'monopoly':
        # Bertrand and Cournot need ownership. Monopoly is technically
        # ownership-free (every product claims every other), but the
        # per-row ownership padding is passed in anyway by Models, so
        # this branch is defensive.
        raise ValueError(
            f"Expected ownership_downstream to be supplied for "
            f"model_type={model_type!r}. Received None."
        )

    # Bucket per-market arrays by J_t (slicing + NaN-column trimming
    # already produces the (J_t, J_t) ownership matrix; same for the
    # response matrix). Use Python lists then np.stack to avoid the
    # per-iteration overhead of pre-allocating differently-shaped
    # buckets.
    by_J: Dict[int, Dict[str, list]] = {}
    by_J_indices: Dict[int, list] = {}
    for t in markets:
        idx_t = market_index_map[t]
        J_t = len(idx_t)
        if ownership_downstream_i is not None:
            own_t = ownership_downstream_i[idx_t]
            own_t = own_t[:, ~np.isnan(own_t).all(axis=0)]
        else:
            own_t = np.ones((J_t, J_t))  # monopoly fallback
        D_t = market_response_map[t]
        s_t = market_shares_map[t]
        if s_t.ndim == 1:
            s_t = s_t.reshape(-1)
        else:
            s_t = s_t.reshape(-1)
        bucket = by_J.setdefault(J_t, {'O': [], 'D': [], 's': []})
        bucket['O'].append(own_t)
        bucket['D'].append(D_t)
        bucket['s'].append(s_t)
        by_J_indices.setdefault(J_t, []).append(idx_t)

    for J_t, bucket in by_J.items():
        O_batch = np.stack(bucket['O'])               # (M_g, J_t, J_t)
        D_batch = np.stack(bucket['D'])               # (M_g, J_t, J_t)
        s_batch = np.stack(bucket['s'])               # (M_g, J_t)
        s_col = s_batch[..., np.newaxis]              # (M_g, J_t, 1)
        if model_type == 'bertrand':
            markup_batch = -np.linalg.solve(O_batch * D_batch, s_col)
        elif model_type == 'cournot':
            inv_D = np.linalg.inv(D_batch)
            markup_batch = -np.matmul(O_batch * inv_D, s_col)
        elif model_type == 'monopoly':
            markup_batch = -np.linalg.solve(D_batch, s_col)
        else:
            # Defensive: should not reach here given _is_batchable
            # gate above.
            raise ValueError(
                f"Unreachable: batched path entered for "
                f"model_type={model_type!r}."
            )

        # Scatter back into out. markup_batch is (M_g, J_t, 1); strip
        # the trailing axis and assign per-market.
        for m_in_bucket, idx_t in enumerate(by_J_indices[J_t]):
            out[idx_t, 0] = markup_batch[m_in_bucket, :, 0]

    return out


def _compute_markups(
        product_data: RecArray, pyblp_results: Mapping, model_downstream: Optional[Array],
        ownership_downstream: Optional[Array], model_upstream: Optional[Array] = None,
        ownership_upstream: Optional[Array] = None, vertical_integration: Optional[Array] = None,
        custom_model_specification: Optional[dict] = None, user_supplied_markups: Optional[Array] = None,
        mix_flag: Optional[Array] = None, demand_backend: Optional[object] = None,
        constant_markup: Optional[Array] = None) -> Array:
    r"""Compute markups given pre-processed model arrays.

    Internal function called by :func:`build_markups` and :meth:`Problem.solve`. Accepts the raw arrays produced by
    :class:`Models` rather than :class:`ModelFormulation` objects.

    Parameters
    ----------
    product_data : `recarray`
        The `product_data` containing information on markets and product characteristics. This should be the same as
        the data used for demand estimation. To compute markups, this data must include `prices`, `market_ids`, and
        `shares`.
    pyblp_results : `structured array-like`
        The results object obtained from using the pyBLP demand estimation procedure. We use built-in PyBLP
        functions to return the demand Jacobians and Hessians (first and second derivatives of shares with respect
        to prices).
    model_downstream: `ndarray`
        The model of conduct for downstream firms. Can be one of [`bertrand`, `cournot`, `monopoly`,
        `perfect_competition`, `other`]. Only specify option `other` if supplying a custom markup formula.
    ownership_downstream: `ndarray`
        The ownership matrix for price or quantity setting (optional, default is standard ownership).
    model_upstream: `ndarray, optional`
        Upstream firm model of conduct. Only specify option `other` if supplying a custom markup formula. Can be one
        of ['none' (default), `bertrand`, `cournot`, `monopoly`, `perfect_competition`, `other`].
    ownership_upstream: `ndarray, optional`
        Ownership matrix for price or quantity setting of upstream firms (optional, default is None).
    vertical_integration: `ndarray, optional`
        Vector indicating which `product_ids` are vertically integrated (ie store brands) (optional, default is
        None).
    custom_model_specification: `dict, optional`
        Dictionary containing a custom markup formula and the name of the formula (optional, default is None).
    user_supplied_markups: `ndarray, optional`
        Vector containing user-computed markups (optional, default is None). If user supplied own markups, this
        function simply returns them.

    Returns
    -------
    `tuple[list, list, list]`
        . Computed markups, downstream markups, and upstream markups for each model.

    Notes
    _____
    For models without vertical integration, firm_ids must be defined in product_data for vi models, and
    firm_ids_upstream and firm_ids (=firm_ids_downstream) must be defined.

    """

    # initialize market characteristics
    N = product_data.shape[0]
    number_models = len(model_downstream)
    markets = np.unique(product_data.market_ids)

    # initialize markups
    markups = [None] * number_models
    markups_upstream = [None] * number_models
    markups_downstream = [None] * number_models
    for i in range(number_models):
        markups_downstream[i] = np.zeros((N, 1), dtype=options.dtype)
        markups_upstream[i] = np.zeros((N, 1), dtype=options.dtype)

    # Transform absent input into list
    if user_supplied_markups is None:
        user_supplied_markups = [None] * number_models
    if custom_model_specification is None:
        custom_model_specification = [None] * number_models
    if model_upstream is None:
        model_upstream = [None] * number_models
    if vertical_integration is None:
        vertical_integration = [None] * number_models
    if mix_flag is None:
        mix_flag = [None] * number_models
    if constant_markup is None:
        constant_markup = [None] * number_models

    # demand Jacobian comes from the backend when provided,
    # or from pyblp_results for the legacy no-backend path (used only by
    # build_markups() public API when the user passes a pyblp results
    # object directly). May be left undefined when every model uses
    # user_supplied_markups (in which case the per-market loop below is
    # never entered).
    ds_dp = None
    if demand_backend is not None:
        ds_dp = demand_backend.compute_jacobian()
    elif pyblp_results is not None:
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            ds_dp = pyblp_results.compute_demand_jacobians()

    # rc8 perf: pre-extract recarray fields once, and precompute per-market
    # index / shares / response slices once. The previous loop called
    # ``np.where(product_data.market_ids == t)`` for every (model, market)
    # pair — an O(N) scan repeated number_models × len(markets) times.
    # The slices only depend on the market id, so caching is a pure win.
    # recarray field access is also slow (goes through __getattribute__),
    # so we extract the plain ndarray here once. Skip the response cache
    # when ds_dp is None (the all-user-supplied-markups case never enters
    # the per-market loop, so the cache would be unused anyway).
    market_index_map: Dict[Any, Array] = {}
    market_shares_map: Dict[Any, Array] = {}
    market_response_map: Dict[Any, Array] = {}
    need_per_market_loop = any(usm is None for usm in user_supplied_markups)
    if need_per_market_loop:
        market_ids_arr = np.asarray(product_data.market_ids).ravel()
        shares_arr = np.asarray(product_data.shares)
        # rc9 perf: groupby in one O(N log N) pass instead of one
        # O(N) np.where per market. Same trick as
        # _build_market_index_map in solve/passthrough.py — kept local
        # here to avoid the import cycle.
        order = np.argsort(market_ids_arr, kind='stable')
        sorted_ids = market_ids_arr[order]
        # Use element-wise inequality rather than np.diff so this works
        # for any dtype (e.g. string market ids), not just numeric ones.
        change_points = np.concatenate(
            ([0], np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]) + 1, [len(sorted_ids)])
        )
        by_id: Dict[Any, Array] = {}
        for k in range(len(change_points) - 1):
            seg = order[change_points[k]:change_points[k + 1]]
            # Coerce to a Python scalar so the dict key is hashable
            # regardless of the underlying numpy dtype (object arrays
            # from recarrays sometimes produce 0-d ndarrays here).
            key = sorted_ids[change_points[k]]
            if isinstance(key, np.ndarray):
                key = key.item()
            by_id[key] = seg
        for t in markets:
            t_key = t.item() if isinstance(t, np.ndarray) else t
            idx_t = by_id[t_key]
            market_index_map[t] = idx_t
            market_shares_map[t] = shares_arr[idx_t]
            # Response cache is only safe to populate when ds_dp is
            # available. If a model needs the loop but ds_dp is None
            # (no backend, no pyblp_results), the inner loop will
            # crash on market_response_map[t] — same failure mode as
            # the pre-rc8 code crashing on ``ds_dp[index_t]``.
            if ds_dp is not None:
                resp_t = ds_dp[idx_t]
                market_response_map[t] = resp_t[:, ~np.isnan(resp_t).all(axis=0)]

    # compute markups market-by-market
    for i in range(number_models):
        if user_supplied_markups[i] is not None:
            markups[i] = user_supplied_markups[i]
            markups_downstream[i] = user_supplied_markups[i]
            continue

        # rc10 perf: route batchable conduct configurations through a
        # batched LAPACK path. "Batchable" means the FOC math is a
        # pure per-market closed form with no upstream / mix / custom
        # spec — i.e., one of the conducts in _BATCHABLE_DOWNSTREAM
        # below. Fall back to the per-market scalar loop for anything
        # else.
        if _is_batchable_downstream_model(
            model_downstream[i],
            model_upstream[i],
            mix_flag[i],
            custom_model_specification[i],
        ):
            markups_downstream[i] = _compute_batchable_downstream_markups(
                model_type=model_downstream[i],
                ownership_downstream_i=ownership_downstream[i],
                constant_markup_i=constant_markup[i],
                markets=markets,
                market_index_map=market_index_map,
                market_shares_map=market_shares_map,
                market_response_map=market_response_map,
                N=N,
            )
        else:
            for t in markets:
                index_t = market_index_map[t]
                shares_t = market_shares_map[t]
                retailer_response_matrix = market_response_map[t]

                # compute downstream markups for model i market t
                markups_downstream[i], retailer_ownership_matrix = evaluate_first_order_conditions(
                    index_t, model_downstream[i], ownership_downstream[i], retailer_response_matrix, shares_t,
                    markups_downstream[i], custom_model_specification[i], markup_type='downstream',
                    type_mix_flag=mix_flag[i], constant_markup=constant_markup[i])

                # compute upstream markups (if applicable) following formula in Villas-Boas (2007)
                if not (model_upstream[i] is None):

                    # construct the matrix of derivatives with respect to prices for other manufacturers
                    markups_t = markups_downstream[i][index_t]
                    if demand_backend is not None:
                        # Hessian from the backend. PyBLPBackend delegates to
                        # pyblp_results.compute_demand_hessians; Logit/NestedLogit
                        # backends use the analytical Hessian. UserSuppliedBackend
                        # without hessian_fn returns None — raise a clear error
                        # since vertical models require a Hessian.
                        d2s_dp2_t = demand_backend.compute_hessian(market_id=t)
                        if d2s_dp2_t is None:
                            from .exceptions import HessianUnavailableError
                            raise HessianUnavailableError(
                                f"Expected the demand backend to provide a Hessian "
                                f"for vertical-integration passthrough (model_upstream "
                                f"is set). "
                                f"Received demand_backend.compute_hessian(market_id="
                                f"{t!r}) returned None. "
                                f"Fix: supply `hessian_fn` to UserSuppliedBackend, or "
                                f"use a built-in backend (PyBLPBackend, LogitBackend, "
                                f"NestedLogitBackend) that computes the Hessian "
                                f"analytically."
                            )
                        passthrough_matrix = _construct_passthrough_from_hessian(
                            d2s_dp2_t, retailer_response_matrix, retailer_ownership_matrix, markups_t
                        )
                    else:
                        passthrough_matrix = construct_passthrough_matrix(
                            pyblp_results, t, retailer_response_matrix, retailer_ownership_matrix, markups_t
                        )

                    # solve for matrix of cross-price elasticities of derived demand and the effects of cost
                    #   pass-through
                    manufacturer_response_matrix = np.transpose(passthrough_matrix) @ retailer_response_matrix

                    # compute upstream markups
                    markups_upstream[i], manufacturer_ownership_matrix = evaluate_first_order_conditions(
                        index_t, model_upstream[i], ownership_upstream[i], manufacturer_response_matrix, shares_t,
                        markups_upstream[i], custom_model_specification[i], markup_type='upstream'
                    )

    # compute total markups as sum of upstream and downstream markups, taking into account vertical integration
    for i in range(number_models):
        if user_supplied_markups[i] is None:
            if vertical_integration[i] is None:
                vi = np.ones((N, 1))
            else:
                vi = (vertical_integration[i] - 1) ** 2
            markups[i] = markups_downstream[i] + vi * markups_upstream[i]

    return markups, markups_downstream, markups_upstream


def construct_passthrough_matrix(
        pyblp_results, market_id, retailer_response_matrix, retailer_ownership_matrix, markups_t):
    """Construct the passthrough matrix using the formula from Villas-Boas (2007). This matrix contains the derivatives
    of all retail prices with respect to all wholesale prices.

    Examples
    --------
    >>> import pyRVtest  # doctest: +SKIP
    >>> # Requires a fitted pyblp.ProblemResults so Hessians can be computed.
    >>> # See docs/tutorial.rst and pyRVtest.build_passthrough for the
    >>> # backend-routed alternative.
    >>> pyRVtest.construct_passthrough_matrix(  # doctest: +SKIP
    ...     pyblp_results, market_id, D_t, O_t, markups_t,
    ... )
    """

    # compute demand hessians
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        d2s_dp2_t = pyblp_results.compute_demand_hessians(market_id=market_id)

    # compute the product of demand hessians and markups
    J = len(markups_t)
    g = np.zeros((J, J))
    for j in range(J):
        g[:, [j]] = (retailer_ownership_matrix * d2s_dp2_t[:, :, j]) @ markups_t

    # solve for derivatives of all prices with respect to the wholesale prices
    H = np.transpose(retailer_ownership_matrix * retailer_response_matrix)
    G = retailer_response_matrix + H + g
    return inv(G) @ H


def evaluate_first_order_conditions(
        index, model_type, type_ownership_matrix, response_matrix, shares, markups, custom_model_specification,
        markup_type, type_mix_flag=None, constant_markup=None):
    """Compute markups for some standard models including Bertrand, Cournot, monopoly, and perfect competition using
    the first order conditions corresponding to each model. Allow user to pass in their own markup function as well.

    Examples
    --------
    >>> import numpy as np
    >>> from pyRVtest import evaluate_first_order_conditions
    >>> index = np.arange(2)
    >>> O = np.eye(2)
    >>> D = np.array([[-2.0, 0.5], [0.5, -2.0]])
    >>> s = np.array([0.3, 0.3])
    >>> markups = np.zeros((2, 1))
    >>> result, _ = evaluate_first_order_conditions(
    ...     index, 'bertrand', O, D, s, markups, None, 'downstream',
    ... )
    >>> result.flatten().round(4)
    array([0.15, 0.15])
    """
    if len(shares.shape) == 1:
        shares = np.expand_dims(shares, axis=1)
    ownership_matrix = None
    if (markup_type == 'downstream') or (markup_type == 'upstream' and model_type is not None):

        # subset ownership matrix and mix_flag to the current market
        if type_ownership_matrix is not None:
            ownership_matrix = type_ownership_matrix[index]
            ownership_matrix = ownership_matrix[:, ~np.isnan(ownership_matrix).all(axis=0)]
        if type_mix_flag is not None:
            mix_flag = type_mix_flag[index]

        # compute markups based on specified model first order condition
        if model_type == 'bertrand':
            markups[index, :] = -np.linalg.solve(ownership_matrix * response_matrix, shares)
        elif model_type == 'cournot':
            markups[index, :] = -(ownership_matrix * inv(response_matrix)) @ shares
        elif model_type == 'monopoly':
            markups[index, :] = -np.linalg.solve(response_matrix, shares)
        elif model_type == 'perfect_competition':
            markups[index, :] = np.zeros((len(shares), 1))
        elif model_type == 'constant_markup':
            # Dearing et al. (2024) Example 7. The markup
            # is a model primitive (per-product dollar markup), supplied
            # as a scalar or as a column of product_data and broadcast
            # into a per-row (N, 1) array upstream of this call.
            if constant_markup is None:
                raise ValueError(
                    "Expected constant_markup to be supplied when "
                    "model_type='constant_markup'. "
                    "Received constant_markup=None. "
                    "Fix: this is an internal wiring bug — the ConstantMarkup "
                    "instance should have populated the Models recarray with "
                    "a non-None entry."
                )
            markups[index, :] = constant_markup[index]
        elif model_type == 'mix_cournot_bertrand':
            markups[index, :] = _compute_mix_cournot_bertrand_markups(
                ownership_matrix, response_matrix, mix_flag, shares
            )
        else:
            if custom_model_specification is not None:
                custom_model, custom_model_formula = next(iter(custom_model_specification.items()))
                if callable(custom_model_formula):
                    markups[index] = custom_model_formula(ownership_matrix, response_matrix, shares)
                else:
                    raise TypeError(
                        f"Expected custom_model_specification[{custom_model!r}] to "
                        f"be a callable f(ownership, response_matrix, shares) -> ndarray. "
                        f"Received a value of type {type(custom_model_formula).__name__} "
                        f"(string formulas are no longer supported post-v0.3). "
                        f"Fix: replace the value with a callable or use "
                        f"pyRVtest.CustomConductModel(markup_fn=...)."
                    )

    return markups, ownership_matrix


def _compute_mix_cournot_bertrand_markups(ownership_matrix, response_matrix, mix_flag, shares):
    """Compute markups for a mixed Cournot/Bertrand market.

    Cournot products use the standard quantity-setting FOC. Bertrand products use the price-setting FOC
    adjusted by the Schur complement (D_BC @ D_CC^{-1} @ D_CB) to account for the feedback from Cournot
    quantity responses, following the derivation in Morrow and Skerlos (2011).
    """
    b, c = mix_flag, ~mix_flag

    shares_B, shares_C = shares[b], shares[c]
    O_BB = ownership_matrix[np.ix_(b, b)]
    O_CC = ownership_matrix[np.ix_(c, c)]
    D_BB = response_matrix[np.ix_(b, b)]
    D_BC = response_matrix[np.ix_(b, c)]
    D_CB = response_matrix[np.ix_(c, b)]
    D_CC = response_matrix[np.ix_(c, c)]

    D_CC_inv = inv(D_CC)
    mkups_C = -(O_CC * D_CC_inv) @ shares_C
    mkups_B = np.linalg.solve(O_BB * (D_BC @ D_CC_inv @ D_CB + D_BB), -shares_B)

    mkups = np.zeros((len(mix_flag), 1))
    mkups[b] = mkups_B.reshape(-1, 1)
    mkups[c] = mkups_C.reshape(-1, 1)
    return mkups


def read_pickle(path: Union[str, Path]) -> object:
    """Load a pickled object into memory.
    This is a simple wrapper around `pickle.load`.

    Parameters
    ----------
    path : `str or Path`
        File path of a pickled object.
    Returns
    -------
    `object`
        The unpickled object.

    Examples
    --------
    >>> import pickle, tempfile, os
    >>> from pyRVtest import read_pickle
    >>> payload = {'a': 1, 'b': [2, 3]}
    >>> with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as fh:
    ...     _ = pickle.dump(payload, fh)
    ...     path = fh.name
    >>> read_pickle(path)
    {'a': 1, 'b': [2, 3]}
    >>> os.remove(path)
    """
    with open(path, 'rb') as handle:
        return pickle.load(handle)
