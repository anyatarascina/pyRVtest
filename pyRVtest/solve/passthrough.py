"""Pass-through matrix construction.

This module exposes ``build_passthrough`` as a user-callable diagnostic
that returns the per-market, per-candidate pass-through matrix
:math:`P_m = (I - \\partial \\Delta_m / \\partial p)^{-1}`.

History
-------

In v0.3 / v0.4 rc1, ``build_passthrough`` was Vertical-only and wrapped
the Villas-Boas (2007) closed form computed analytically from the
demand Hessian (see :func:`pyRVtest.markups._construct_passthrough_from_hessian`).
That path computed :math:`dp/dw` (retail-price response to a wholesale
shock) for a downstream-Bertrand / upstream-anything bilateral
oligopoly.

Phase 1 of the DMQSW pass-through diagnostic (v0.4 final) generalises
the entry point to every conduct class. Non-Vertical candidates are
routed through :func:`compute_passthrough_numerical`, which computes
:math:`\\partial \\Delta_m / \\partial p` by central-difference
perturbation propagated through the demand Jacobian / Hessian, then
inverts :math:`I - \\partial \\Delta_m / \\partial p`. The Vertical
candidate keeps the existing Villas-Boas analytical fast path: its
:math:`dp/dw` is the same object the upstream-markup pipeline computes
during ``solve()``, and reproducing it numerically would be redundant
work.

The trivial closed forms (PerfectCompetition, ConstantMarkup,
UserSuppliedMarkups all give :math:`P = I`; RuleOfThumb(\\phi) gives
:math:`P = \\phi I`) are handled directly without finite differencing
to avoid round-off noise on what should be exact answers.

Documented analytical formulas live in ``docs/math.rst`` under
"Pass-through by conduct class"; the package computes numerically and
math.rst documents the closed forms the numerics approximate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Hashable, List, Optional, Tuple, Union

import numpy as np
from numpy.linalg import inv
from numpy.typing import NDArray
from typing_extensions import TypeAlias

if TYPE_CHECKING:
    import pandas as pd

from ..exceptions import HessianUnavailableError
from ..markups import _construct_passthrough_from_hessian, evaluate_first_order_conditions
from ..models.constant import ConstantMarkup, RuleOfThumb
from ..models.mixed import MixCournotBertrand
from ..models.standard import PerfectCompetition
from ..models.user_supplied import UserSuppliedMarkups
from ..models.vertical import Vertical


__all__ = [
    'InstrumentChannels',
    'PassthroughSummary',
    'build_passthrough',
    'compute_instrument_channels',
    'compute_passthrough_numerical',
    'compute_passthrough_reliability',
    'compute_passthrough_summary',
]


_NDArray: TypeAlias = NDArray[Any]


# ===========================================================================
# Numerical pass-through core
# ===========================================================================


def _build_market_index_map(
        product_market_ids: _NDArray,
        market_ids: List[Hashable],
) -> Dict[Hashable, _NDArray]:
    """Build ``{market_id: per-market row indices}`` in a single O(N log N) pass.

    The straightforward form ``{t: np.where(product_market_ids == t)[0]
    for t in market_ids}`` does an O(N) scan per market, giving O(M·N)
    work. For typical PT diagnostic inputs (M close to N when products
    per market is small) that quadratic is a real bottleneck. The
    argsort + changepoint form below is O(N log N), with a much smaller
    constant factor.
    """
    pmi = np.asarray(product_market_ids)
    order = np.argsort(pmi, kind='stable')
    sorted_ids = pmi[order]
    change_points = np.concatenate(
        ([0], np.where(np.diff(sorted_ids) != 0)[0] + 1, [len(sorted_ids)])
    )
    by_id: Dict[Hashable, _NDArray] = {}
    for k in range(len(change_points) - 1):
        seg = order[change_points[k]:change_points[k + 1]]
        by_id[sorted_ids[change_points[k]]] = seg
    # Preserve caller-requested ordering: rebuild dict in market_ids
    # order so downstream code that iterates the dict in insertion
    # order (rare but possible) sees the canonical order.
    return {t: by_id[t] for t in market_ids}


# rc8 perf: batched conduct types that have closed-form markup formulas
# expressible as a single numpy linalg call per perturbation direction.
# Adding a class to this set requires writing a batched markup formula
# in :func:`_compute_passthrough_numerical_batched` (just below).
_BATCHABLE_MODEL_TYPES = frozenset({'bertrand', 'cournot', 'monopoly'})


def _compute_passthrough_numerical_batched(
        model_type: str,
        ownership_batch: _NDArray,
        response_batch: _NDArray,
        hessian_batch: _NDArray,
        shares_batch: _NDArray,
        delta: float = 1e-7,
) -> _NDArray:
    r"""Batched central-difference pass-through for same-J markets.

    Computes :math:`P_m = (I - \partial \Delta_m / \partial p)^{-1}` for a
    batch of ``M`` markets that all have the same product count ``J``,
    in a fixed number of vectorized numpy calls. Equivalent to the
    per-market loop in :func:`compute_passthrough_numerical` but a
    constant number of LAPACK dispatches instead of ``O(M * J)``.

    Only the three batchable conduct types in
    :data:`_BATCHABLE_MODEL_TYPES` (``'bertrand'``, ``'cournot'``,
    ``'monopoly'``) are supported. Trivial-closed-form conducts
    (``PerfectCompetition`` / ``ConstantMarkup`` / ``RuleOfThumb`` /
    ``UserSuppliedMarkups``) are short-circuited at the call site in
    :func:`build_passthrough` before reaching here. Mix /
    PartialCollusion / CustomConductModel fall back to the per-market
    :func:`compute_passthrough_numerical`.

    Parameters
    ----------
    model_type : str
        One of ``'bertrand'``, ``'cournot'``, ``'monopoly'``.
    ownership_batch : ndarray, shape ``(M, J, J)``
        Per-market ownership matrices stacked along axis 0.
    response_batch : ndarray, shape ``(M, J, J)``
        Per-market demand Jacobians.
    hessian_batch : ndarray, shape ``(M, J, J, J)``
        Per-market demand Hessians. Indexed
        ``hessian[m, i, j, k] = d^2 s_i / dp_j dp_k`` (market m).
    shares_batch : ndarray, shape ``(M, J)``
        Per-market observed shares.
    delta : float, optional
        Central-difference step.

    Returns
    -------
    ndarray, shape ``(M, J, J)``
        Stacked per-market pass-through matrices, same order as input.
    """
    M, J = shares_batch.shape

    # Shape-check the inputs (debug-time guard; numpy would broadcast
    # silently otherwise).
    assert ownership_batch.shape == (M, J, J), ownership_batch.shape
    assert response_batch.shape == (M, J, J), response_batch.shape
    assert hessian_batch.shape == (M, J, J, J), hessian_batch.shape

    jac_delta = np.zeros((M, J, J))

    for k in range(J):
        # Perturbation in direction k: dp = delta * e_k. Broadcasting:
        # hessian_batch[..., k] is (M, J, J); response_batch[..., k] is (M, J).
        dD_k = hessian_batch[..., k] * delta              # (M, J, J)
        ds_k = response_batch[..., k] * delta             # (M, J)

        D_plus = response_batch + dD_k                    # (M, J, J)
        D_minus = response_batch - dD_k
        s_plus = shares_batch + ds_k                      # (M, J)
        s_minus = shares_batch - ds_k

        # Per-conduct batched markup formula. The s vector is reshaped
        # to (M, J, 1) so numpy's batched solve / matmul see "M batched
        # right-hand sides of length J".
        s_plus_col = s_plus[..., np.newaxis]              # (M, J, 1)
        s_minus_col = s_minus[..., np.newaxis]

        if model_type == 'bertrand':
            # markup = -solve(O * D, s)
            Delta_plus = -np.linalg.solve(
                ownership_batch * D_plus, s_plus_col,
            )[..., 0]
            Delta_minus = -np.linalg.solve(
                ownership_batch * D_minus, s_minus_col,
            )[..., 0]
        elif model_type == 'cournot':
            # markup = -(O * inv(D)) @ s
            inv_D_plus = np.linalg.inv(D_plus)
            inv_D_minus = np.linalg.inv(D_minus)
            Delta_plus = -np.matmul(
                ownership_batch * inv_D_plus, s_plus_col,
            )[..., 0]
            Delta_minus = -np.matmul(
                ownership_batch * inv_D_minus, s_minus_col,
            )[..., 0]
        elif model_type == 'monopoly':
            # markup = -solve(D, s)
            Delta_plus = -np.linalg.solve(D_plus, s_plus_col)[..., 0]
            Delta_minus = -np.linalg.solve(D_minus, s_minus_col)[..., 0]
        else:
            raise ValueError(
                f"Expected model_type in {sorted(_BATCHABLE_MODEL_TYPES)}. "
                f"Received {model_type!r}. "
                f"Fix: call _compute_passthrough_numerical_batched only for "
                f"the batchable conduct types listed in "
                f"_BATCHABLE_MODEL_TYPES, or add a batched markup formula "
                f"for this conduct."
            )

        jac_delta[..., k] = (Delta_plus - Delta_minus) / (2.0 * delta)

    # Batched (I - jac_delta)^{-1}.
    I_batch = np.broadcast_to(np.eye(J), (M, J, J))
    return np.asarray(np.linalg.inv(I_batch - jac_delta))


def compute_passthrough_numerical(
        conduct_model: Any,
        ownership: _NDArray,
        response: _NDArray,
        hessian: _NDArray,
        shares: _NDArray,
        markups: _NDArray,
        delta: float = 1e-7,
        custom_model_specification: Optional[Dict[str, Any]] = None,
        constant_markup: Optional[_NDArray] = None,
        mix_flag: Optional[_NDArray] = None,
) -> _NDArray:
    r"""Compute the pass-through matrix :math:`P_m = (I - \partial \Delta_m / \partial p)^{-1}`.

    The candidate's first-order condition is differentiated numerically
    by central-difference perturbation of prices, propagated through
    the demand system via its Jacobian and Hessian (which give the
    first-order responses :math:`ds/dp_k = D[:, k]` and
    :math:`dD/dp_k = \mathrm{Hess}[:, :, k]`):

    .. math::

       \Delta_m(p \pm \delta e_k) \approx
           f_m(O, D \pm \delta\, \mathrm{Hess}[:, :, k],
               s \pm \delta\, D[:, k])

    so that

    .. math::

       \frac{\partial \Delta_m}{\partial p_k}
           \approx \frac{\Delta_m(p + \delta e_k) - \Delta_m(p - \delta e_k)}{2\,\delta}.

    This converges to the analytical derivative as :math:`\delta \to 0`,
    matching the closed-form per-conduct formulas listed in
    ``docs/math.rst``. The same code path handles every conduct class
    that exposes a markup function, with trivial-closed-form cases
    (PerfectCompetition, ConstantMarkup, UserSuppliedMarkups,
    RuleOfThumb) short-circuited to their exact analytical answer.

    Parameters
    ----------
    conduct_model : pyRVtest.ConductModel
        Candidate conduct instance (e.g. ``Bertrand(...)``,
        ``Cournot(...)``, ``MixCournotBertrand(...)``,
        ``ConstantMarkup(...)``). The Vertical wrapper is NOT accepted
        here; ``build_passthrough`` short-circuits to the Villas-Boas
        closed form for that case.
    ownership : ndarray, shape ``(J_t, J_t)``
        Per-market ownership matrix (kappa-modified for partial
        collusion).
    response : ndarray, shape ``(J_t, J_t)``
        Per-market demand Jacobian :math:`D = \partial s / \partial p`.
    hessian : ndarray, shape ``(J_t, J_t, J_t)``
        Per-market demand Hessian
        :math:`\partial^2 s / \partial p \partial p`. Indexed
        ``hessian[i, j, k] = \partial^2 s_i / \partial p_j \partial p_k``.
    shares : ndarray, shape ``(J_t,)`` or ``(J_t, 1)``
        Per-market observed shares.
    markups : ndarray, shape ``(J_t,)`` or ``(J_t, 1)``
        Per-market candidate markups at the observed state. Used only
        when the candidate's closed form is short-circuited (e.g. for
        the trivial-closed-form classes).
    delta : float, optional
        Central-difference step. Default ``1e-7`` is small enough to be
        well within the analytical-derivative regime for double precision
        and large enough to avoid catastrophic cancellation in the
        markup evaluation.
    custom_model_specification : dict, optional
        Forwarded to :func:`evaluate_first_order_conditions` for
        ``CustomConductModel`` instances. Ignored otherwise.
    constant_markup : ndarray, optional
        Per-row vector :math:`\zeta_j` for ``ConstantMarkup`` /
        ``RuleOfThumb`` instances. Ignored otherwise.
    mix_flag : ndarray, optional
        Per-row boolean mix flag for ``MixCournotBertrand``. Ignored
        otherwise.

    Returns
    -------
    ndarray, shape ``(J_t, J_t)``
        Pass-through matrix :math:`P_m`.

    Raises
    ------
    ValueError
        If ``hessian`` is not a 3D array, if shapes are inconsistent, or
        if the candidate is a Vertical wrapper (which has a separate
        analytical fast path).

    Examples
    --------
    >>> import numpy as np
    >>> from pyRVtest import Bertrand
    >>> from pyRVtest.solve.passthrough import compute_passthrough_numerical
    >>> # Two single-product firms with simple symmetric logit-like demand.
    >>> O = np.eye(2)
    >>> D = np.array([[-2.0, 0.5], [0.5, -2.0]])
    >>> Hess = np.zeros((2, 2, 2))  # zero Hessian -> linear demand
    >>> s = np.array([0.3, 0.3])
    >>> markup = -np.linalg.solve(O * D, s.reshape(2, 1))
    >>> P = compute_passthrough_numerical(
    ...     Bertrand(ownership='firm_ids'), O, D, Hess, s, markup,
    ... )
    >>> P.shape
    (2, 2)
    """
    if isinstance(conduct_model, Vertical):
        raise ValueError(
            "Expected a non-Vertical ConductModel for "
            "compute_passthrough_numerical (Vertical pass-through has a "
            "dedicated analytical fast path via build_passthrough). "
            f"Received {type(conduct_model).__name__}. "
            "Fix: call build_passthrough(problem, model_index=...) for the "
            "vertical case, which routes through the Villas-Boas closed form."
        )

    response = np.asarray(response, dtype=float)
    hessian = np.asarray(hessian, dtype=float)
    ownership = np.asarray(ownership, dtype=float)
    shares = np.asarray(shares, dtype=float).reshape(-1)
    markups = np.asarray(markups, dtype=float).reshape(-1)

    J = response.shape[0]
    if response.shape != (J, J):
        raise ValueError(
            f"Expected response (Jacobian) to be a square (J_t, J_t) array. "
            f"Received shape {response.shape}. "
            f"Fix: pass the per-market ds/dp matrix from the demand backend."
        )
    if hessian.shape != (J, J, J):
        raise ValueError(
            f"Expected hessian to be a (J_t, J_t, J_t) array with "
            f"hessian[i, j, k] = d^2 s_i / dp_j dp_k. "
            f"Received shape {hessian.shape}. "
            f"Fix: pass the per-market Hessian from the demand backend "
            f"(``backend.compute_hessian(market_id=t)``)."
        )
    if ownership.shape != (J, J):
        raise ValueError(
            f"Expected ownership to be a (J_t, J_t) array. "
            f"Received shape {ownership.shape}. "
            f"Fix: pass the per-market ownership matrix."
        )
    if shares.shape[0] != J:
        raise ValueError(
            f"Expected shares to have length J_t = {J}. "
            f"Received shape {shares.shape}. "
            f"Fix: pass the per-market share vector."
        )

    # ---- trivial-closed-form short circuits ----
    if isinstance(conduct_model, PerfectCompetition):
        return np.eye(J)
    if isinstance(conduct_model, ConstantMarkup):
        # constant per-product dollar markup -> dDelta/dp = 0
        return np.eye(J)
    if isinstance(conduct_model, UserSuppliedMarkups):
        # external column treated as primitive -> dDelta/dp = 0
        return np.eye(J)
    if isinstance(conduct_model, RuleOfThumb):
        # Delta = (phi - 1)/phi * p  =>  dDelta/dp = ((phi - 1)/phi) I
        # P = (I - ((phi - 1)/phi) I)^{-1} = phi * I.
        phi = float(conduct_model.phi)
        return phi * np.eye(J)

    # ---- numerical derivative for all other conducts ----
    model_type = conduct_model._model_name

    # Per-call buffers reused for every directional perturbation.
    index = np.arange(J)
    markup_buf_plus = np.zeros((J, 1))
    markup_buf_minus = np.zeros((J, 1))

    def _eval_markup(D_eff: _NDArray, s_eff: _NDArray, buf: _NDArray) -> _NDArray:
        """Evaluate the candidate markup at perturbed (D, s)."""
        s_col = s_eff.reshape(-1)
        if model_type == 'mix_cournot_bertrand':
            # MixCournotBertrand needs the per-market mix_flag slice.
            # The legacy FOC dispatch resolves the slice via ``index``;
            # here we already have the per-market boolean directly.
            if mix_flag is None:
                raise ValueError(
                    "Expected mix_flag to be supplied for MixCournotBertrand. "
                    "Received mix_flag=None."
                )
            from ..markups import _compute_mix_cournot_bertrand_markups
            mix_b = np.asarray(mix_flag, dtype=bool).reshape(-1)
            buf[:, :] = _compute_mix_cournot_bertrand_markups(  # type: ignore[no-untyped-call]
                ownership, D_eff, mix_b, s_col,
            )
            return buf
        # Reuse evaluate_first_order_conditions for every other class so the
        # FOC math stays in one place.
        out, _ = evaluate_first_order_conditions(  # type: ignore[no-untyped-call]
            index, model_type, ownership[None, ...], D_eff, s_col, buf,
            custom_model_specification, markup_type='downstream',
            type_mix_flag=None, constant_markup=constant_markup,
        )
        # evaluate_first_order_conditions writes into ``buf`` indexed by
        # ``index``, but it slices ``type_ownership_matrix`` along axis 0.
        # We sidestepped that by feeding a singleton-axis ownership above;
        # restore the contract here.
        return np.asarray(out)

    jac_delta = np.zeros((J, J))

    # Pre-compute the broadcast ownership we use inside _eval_markup. The
    # FOC dispatch slices ``type_ownership_matrix[index]`` along axis 0,
    # so we pre-stack the ownership matrix to shape (J, J, J) with the
    # first axis a "row index", producing the per-row ownership row when
    # sliced. That matches the contract of evaluate_first_order_conditions
    # in the legacy markups path.
    O_stacked = np.broadcast_to(ownership, (J, J, J)).copy()

    def _eval_markup_direct(D_eff: _NDArray, s_eff: _NDArray) -> _NDArray:
        """Evaluate the candidate markup at perturbed (D, s) directly.

        Bypasses ``evaluate_first_order_conditions`` to avoid the slicing
        ambiguity around per-market ownership formats; calls the
        per-class markup formulas directly.
        """
        s_col = s_eff.reshape(-1, 1)
        if model_type == 'bertrand':
            return -np.linalg.solve(ownership * D_eff, s_col)
        if model_type == 'cournot':
            return np.asarray(-(ownership * inv(D_eff)) @ s_col)
        if model_type == 'monopoly':
            return -np.linalg.solve(D_eff, s_col)
        if model_type == 'perfect_competition':
            return np.zeros_like(s_col)
        if model_type == 'mix_cournot_bertrand':
            if mix_flag is None:
                raise ValueError(
                    "Expected mix_flag to be supplied for MixCournotBertrand. "
                    "Received mix_flag=None."
                )
            from ..markups import _compute_mix_cournot_bertrand_markups
            mix_b = np.asarray(mix_flag, dtype=bool).reshape(-1)
            return np.asarray(_compute_mix_cournot_bertrand_markups(  # type: ignore[no-untyped-call]
                ownership, D_eff, mix_b, s_col.reshape(-1),
            ))
        if model_type == 'constant_markup':
            # ConstantMarkup / RuleOfThumb -> handled by short circuits
            # above. If we reach here something has gone wrong.
            raise RuntimeError(
                "compute_passthrough_numerical reached the FOC dispatch "
                "for a constant_markup model. This should have been "
                f"short-circuited above. type={type(conduct_model).__name__}."
            )
        if model_type == 'user_supplied':
            raise RuntimeError(
                "compute_passthrough_numerical reached the FOC dispatch "
                "for user_supplied. This should have been short-circuited "
                f"above. type={type(conduct_model).__name__}."
            )
        # Custom model
        if custom_model_specification is not None:
            _, formula = next(iter(custom_model_specification.items()))
            if callable(formula):
                out = formula(ownership, D_eff, s_col.reshape(-1))
                return np.asarray(out, dtype=float).reshape(-1, 1)
        raise ValueError(
            f"compute_passthrough_numerical does not know how to evaluate "
            f"markup for model_type={model_type!r} on conduct "
            f"{type(conduct_model).__name__}. "
            f"Fix: extend the dispatch in compute_passthrough_numerical "
            f"or provide a custom_model_specification."
        )

    for k in range(J):
        # Perturbation in direction k: dp = delta * e_k.
        dD_k = hessian[:, :, k] * delta
        ds_k = response[:, k] * delta

        D_plus = response + dD_k
        D_minus = response - dD_k
        s_plus = shares + ds_k
        s_minus = shares - ds_k

        Delta_plus = _eval_markup_direct(D_plus, s_plus).reshape(-1)
        Delta_minus = _eval_markup_direct(D_minus, s_minus).reshape(-1)

        jac_delta[:, k] = (Delta_plus - Delta_minus) / (2.0 * delta)

    return inv(np.eye(J) - jac_delta)


# ===========================================================================
# Public entry point
# ===========================================================================


def build_passthrough(
        problem: Any,
        model_index: int,
        market_id: Optional[Hashable] = None,
        delta: float = 1e-7,
        _precomputed_markups: Optional[Tuple[List[_NDArray], List[Optional[_NDArray]]]] = None,
        _precomputed_demand_derivatives: Optional[Dict[Hashable, Tuple[_NDArray, Optional[_NDArray]]]] = None,
        _precomputed_market_indices: Optional[Dict[Hashable, _NDArray]] = None,
) -> Union[_NDArray, Dict[Hashable, _NDArray]]:
    r"""Compute the pass-through matrix for one candidate conduct model.

    For the Vertical wrapper, returns the Villas-Boas (2007) :math:`dp/dw`
    matrix (retail-price response to a wholesale shock) computed
    analytically from the demand Hessian. For every other conduct class,
    returns :math:`P_m = (I - \partial \Delta_m / \partial p)^{-1}`
    computed numerically via :func:`compute_passthrough_numerical`.

    Parameters
    ----------
    problem : pyRVtest.Problem
        An initialized :class:`pyRVtest.Problem`.
    model_index : int
        Which entry in ``problem._models`` to use. Must be in
        ``[0, len(problem._models))``.
    market_id : hashable, optional
        If ``None`` (default), return ``{t: passthrough_matrix_t}`` for
        every market ``t`` in ``problem.unique_market_ids``. Otherwise
        return the single ``(J_t, J_t)`` matrix for that market.
    delta : float, optional
        Central-difference step for the numerical path. Default
        ``1e-7``. Ignored for the Vertical analytical fast path and for
        the trivial-closed-form classes.
    _precomputed_markups : tuple of (list, list), optional
        Internal optimization hook. ``(markups_full, markups_downstream)``
        as returned by ``problem._perturb_and_build_markups()``. When
        supplied, ``build_passthrough`` skips its own
        ``_perturb_and_build_markups`` call and uses the lists provided.
        The high-level diagnostics
        (:func:`compute_passthrough_summary`,
        :func:`compute_instrument_channels`) compute markups once and
        thread them through to every per-model ``build_passthrough`` call;
        on the 3000-market synthetic this is the dominant cost. Public
        callers should leave this at ``None``; the underscore prefix
        signals that the parameter is not part of the stable API.
    _precomputed_demand_derivatives : dict, optional
        Internal optimization hook. ``{market_id: (D_t, H_t)}`` where
        ``D_t`` is the per-market demand Jacobian and ``H_t`` is the
        Hessian (or ``None`` for trivial-conduct short-circuits that
        don't need it). When supplied, ``build_passthrough`` skips per-
        market ``backend.compute_jacobian`` / ``backend.compute_hessian``
        calls and reads from the dict. Demand derivatives only depend on
        demand state, not on the candidate conduct, so hoisting them out
        of the per-model loop eliminates ``n_models``-fold recomputation
        — particularly important for the Hessian, which isn't cached at
        the backend level. Public callers should leave this at ``None``.

    Returns
    -------
    ndarray or dict
        Pass-through matrix. Shape ``(J_t, J_t)`` for a specific
        ``market_id``, or ``{market_id: (J_t, J_t)}`` across all markets.

    Raises
    ------
    ValueError
        If ``market_id`` is not in the Problem's
        ``unique_market_ids``; if ``model_index`` is out of range; or
        if the Problem's demand backend does not provide a Hessian
        (e.g., ``UserSuppliedBackend`` without ``hessian_fn``) and the
        Hessian is required (Vertical or non-trivial conduct).

    Examples
    --------
    >>> from pyRVtest import build_passthrough  # doctest: +SKIP
    >>> # Phase 1 generalization: any candidate works.
    >>> P_t = build_passthrough(problem, model_index=0, market_id=5)  # doctest: +SKIP
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

    candidate = problem._models[model_index]
    models_upstream = problem.models['models_upstream']
    is_vertical = (
        isinstance(candidate, Vertical)
        or models_upstream[model_index] is not None
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
    # When the caller is a high-level diagnostic that has already paid for
    # the markups assembly, skip the redundant rebuild.
    if _precomputed_markups is not None:
        markups_full, markups_downstream = _precomputed_markups
    else:
        markups_full, markups_downstream, _ = problem._perturb_and_build_markups()
    markups_down = markups_downstream[model_index]
    if markups_down is None:
        # Defensive fallback; markups_full has the combined value for
        # vertical models. The downstream-only slot is needed for
        # Villas-Boas; for non-vertical conducts it equals markups_full.
        markups_down = markups_full[model_index]

    # --- demand backend ---
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
    # rc8 perf: extract recarray fields once (pandas/recarray attribute
    # access goes through __getattribute__ and is slow when called
    # 3000+ times inside the per-market loop below).
    all_shares = np.asarray(problem.products.shares).flatten()
    def _model_field(name: str) -> Any:
        # ``problem.models`` is a structured recarray in production but
        # some test fixtures pass a plain dict. Both support ``key in obj``
        # for membership; recarrays expose ``dtype.names`` while dicts use
        # ``__contains__`` directly.
        names = getattr(getattr(problem.models, 'dtype', None), 'names', None)
        present = (name in names) if names is not None else (name in problem.models)
        return problem.models[name][model_index] if present else None

    constant_markup_full = _model_field('constant_markup')
    mix_flag_full = _model_field('mix_flag')
    custom_model_full = _model_field('custom_model')

    # Trivial-closed-form short circuits don't even need a Hessian. Detect
    # before we reach into the backend.
    needs_hessian = is_vertical or not isinstance(
        candidate, (PerfectCompetition, ConstantMarkup, RuleOfThumb, UserSuppliedMarkups)
    )

    # rc8 perf: when the conduct has a batchable closed-form markup
    # (Bertrand / Cournot / Monopoly — PartialCollusion piggybacks on
    # Bertrand via its kappa-modified ownership and gets included for
    # free), collect per-market inputs and dispatch one batched LAPACK
    # call per perturbation direction instead of one scalar call per
    # (market, direction). Saves the Python-overhead × 9000 cost that
    # dominates the per-market scalar path for small J_t.
    candidate_model_type = getattr(candidate, '_model_name', '')
    use_batched_path = (
        not is_vertical
        and needs_hessian  # excludes the trivial-closed-form short circuits
        and candidate_model_type in _BATCHABLE_MODEL_TYPES
        and custom_model_full is None
        and mix_flag_full is None
    )
    _batch_inputs: Dict[int, List[Dict[str, Any]]] = {}  # J_t -> per-market input dicts

    results: Dict[Hashable, _NDArray] = {}
    for t in markets_to_compute:
        # rc9 perf: pre-built per-market indices when the high-level
        # diagnostic computed them once. Otherwise fall back to an
        # O(N) scan per market.
        if _precomputed_market_indices is not None and t in _precomputed_market_indices:
            index_t = _precomputed_market_indices[t]
        else:
            index_t = np.where(product_market_ids == t)[0]
        j_t = len(index_t)

        # Trivial-closed-form short circuits for the non-vertical case.
        # These conducts don't depend on ownership or the demand Hessian,
        # so we can answer without touching the backend.
        if not is_vertical:
            if isinstance(candidate, (PerfectCompetition, ConstantMarkup, UserSuppliedMarkups)):
                results[t] = np.eye(j_t)
                continue
            if isinstance(candidate, RuleOfThumb):
                results[t] = float(candidate.phi) * np.eye(j_t)
                continue

        # Jacobian / Hessian: if the high-level diagnostic already paid
        # for them, read from the precomputed dict. Otherwise hit the
        # backend. Note that demand derivatives depend only on demand
        # state, not on the candidate conduct, so hoisting them out of
        # the per-model loop eliminates n_models-fold recomputation.
        if _precomputed_demand_derivatives is not None and t in _precomputed_demand_derivatives:
            D_t, _precomputed_H_t = _precomputed_demand_derivatives[t]
        else:
            D_t = backend.compute_jacobian(market_id=t)
            _precomputed_H_t = None  # signal "compute fresh below if needed"

        # Ownership: slice to this market and drop NaN-padded columns.
        ownership_t = ownership_downstream[index_t]
        ownership_t = ownership_t[:, ~np.isnan(ownership_t).all(axis=0)]

        # Markups for this market.
        markups_t = markups_down[index_t]

        if is_vertical:
            # Existing Villas-Boas closed form: dp/dw, retail prices vs
            # wholesale prices. Kept as the analytical fast path; the
            # numerical core would have to perturb the upstream FOC as
            # well, which is redundant work.
            d2s_dp2_t = (
                _precomputed_H_t if _precomputed_H_t is not None
                else backend.compute_hessian(market_id=t)
            )
            if d2s_dp2_t is None:
                raise HessianUnavailableError(
                    f"Expected the demand backend to provide a Hessian for "
                    f"vertical-integration passthrough. "
                    f"Received problem._demand_backend.compute_hessian(market_id="
                    f"{t!r}) returned None. "
                    f"Fix: supply `hessian_fn` to UserSuppliedBackend, or use a "
                    f"built-in backend (PyBLPBackend, LogitBackend, NestedLogitBackend)."
                )
            passthrough_t: _NDArray = _construct_passthrough_from_hessian(  # type: ignore[no-untyped-call]
                d2s_dp2_t, D_t, ownership_t, markups_t,
            )
            results[t] = passthrough_t
            continue

        # Non-Vertical: numerical core.
        if needs_hessian:
            d2s_dp2_t = (
                _precomputed_H_t if _precomputed_H_t is not None
                else backend.compute_hessian(market_id=t)
            )
            if d2s_dp2_t is None:
                raise HessianUnavailableError(
                    f"Expected the demand backend to provide a Hessian for "
                    f"numerical pass-through (non-trivial conduct). "
                    f"Received problem._demand_backend.compute_hessian(market_id="
                    f"{t!r}) returned None. "
                    f"Fix: supply `hessian_fn` to UserSuppliedBackend, or use a "
                    f"built-in backend (PyBLPBackend, LogitBackend, NestedLogitBackend)."
                )
        else:
            # PC / ConstantMarkup / RuleOfThumb / UserSupplied: trivial
            # closed forms don't need the Hessian. Pass a dummy so the
            # signature shape check inside compute_passthrough_numerical
            # passes; it's not read for these classes.
            J_t = D_t.shape[0]
            d2s_dp2_t = np.zeros((J_t, J_t, J_t))

        shares_t = all_shares[index_t]

        # Per-row constant_markup slice (RuleOfThumb / ConstantMarkup
        # path). The Models recarray holds an (N, 1) array per model when
        # populated; index into it by market index here.
        if constant_markup_full is not None:
            cm_t = constant_markup_full[index_t]
        else:
            cm_t = None

        if mix_flag_full is not None:
            mf_t = mix_flag_full[index_t]
        else:
            mf_t = None

        if custom_model_full is not None:
            custom_spec = custom_model_full
        else:
            custom_spec = None

        if use_batched_path:
            # Defer the actual numerical core; bucket by J_t so we can
            # batch same-shape markets in a single LAPACK call below.
            _batch_inputs.setdefault(int(D_t.shape[0]), []).append({
                't': t,
                'ownership': ownership_t,
                'response': D_t,
                'hessian': d2s_dp2_t,
                'shares': shares_t,
            })
        else:
            passthrough_t = compute_passthrough_numerical(
                candidate,
                ownership=ownership_t,
                response=D_t,
                hessian=d2s_dp2_t,
                shares=shares_t,
                markups=markups_t,
                delta=delta,
                custom_model_specification=custom_spec,
                constant_markup=cm_t,
                mix_flag=mf_t,
            )
            results[t] = passthrough_t

    # rc8 perf: discharge any batched buckets we accumulated above.
    # Each bucket is one batched LAPACK dispatch.
    if use_batched_path and _batch_inputs:
        for J_size, bucket in _batch_inputs.items():
            n_in_bucket = len(bucket)
            ownership_batch = np.stack([m['ownership'] for m in bucket])
            response_batch = np.stack([m['response'] for m in bucket])
            hessian_batch = np.stack([m['hessian'] for m in bucket])
            shares_batch = np.stack([np.asarray(m['shares']).reshape(-1) for m in bucket])
            P_batch = _compute_passthrough_numerical_batched(
                candidate_model_type,
                ownership_batch=ownership_batch,
                response_batch=response_batch,
                hessian_batch=hessian_batch,
                shares_batch=shares_batch,
                delta=delta,
            )
            for k_in_bucket, m_in in enumerate(bucket):
                results[m_in['t']] = P_batch[k_in_bucket]
            del ownership_batch, response_batch, hessian_batch, shares_batch
            del n_in_bucket

    if market_id is not None:
        return results[market_id]
    return results


# ===========================================================================
# Phase 2: passthrough_summary — pair × pass-through-feature distances
# ===========================================================================


def _metric_full_pass(P_i: _NDArray, P_j: _NDArray, p_t: _NDArray,
                     D_i: _NDArray, D_j: _NDArray) -> float:
    """Frobenius norm of full pass-through difference (Remark 2 target)."""
    return float(np.linalg.norm(P_i - P_j, 'fro'))


def _metric_offdiag_ratio(P_i: _NDArray, P_j: _NDArray, p_t: _NDArray,
                         D_i: _NDArray, D_j: _NDArray,
                         eps_diag: float = 1e-12) -> float:
    """Frobenius norm of off-diagonal column-ratio differences (Remark 1 target).

    For each off-diagonal position (j, ℓ) with j ≠ ℓ, the Remark 1 target
    is the column-normalized off-diagonal entry (P)_{jℓ}/(P)_{ℓℓ}. The
    aggregated distance over off-diagonals captures whether rival cost
    shifters can distinguish the pair regardless of γ.

    Denominator degeneracy. If any column ℓ has ``|diag(P_i)[ℓ]| <= eps_diag``
    or ``|diag(P_j)[ℓ]| <= eps_diag``, the column ratio is mathematically
    undefined for that column. Returning a numeric value in that regime
    is misleading (pre-rc15 the function returned 0 via the
    ``inf``-substitution, which the user could not distinguish from
    "true structural degeneracy"). rc15 instead returns ``np.nan`` for
    the whole market when any column is degenerate. The caller
    (:func:`compute_passthrough_summary`) aggregates with ``nanmedian``
    and reports the count of degenerate markets separately. Audit 2
    rc14 re-audit Finding 3.
    """
    J = P_i.shape[0]
    if J <= 1:
        return 0.0
    diag_i = np.diag(P_i).astype(float)
    diag_j = np.diag(P_j).astype(float)
    # Denominator-degeneracy check: NaN out when any column has |diag| <= eps.
    if (np.any(np.abs(diag_i) <= eps_diag)
            or np.any(np.abs(diag_j) <= eps_diag)):
        return float('nan')
    ratio_i = P_i / diag_i[np.newaxis, :]
    ratio_j = P_j / diag_j[np.newaxis, :]
    diff = ratio_i - ratio_j
    np.fill_diagonal(diff, 0.0)
    return float(np.linalg.norm(diff, 'fro'))


def _metric_row_sum(P_i: _NDArray, P_j: _NDArray, p_t: _NDArray,
                   D_i: _NDArray, D_j: _NDArray) -> float:
    """L2 norm of row-sum difference (Remark 5 unit-tax target)."""
    iota = np.ones(P_i.shape[0])
    return float(np.linalg.norm((P_i - P_j) @ iota))


def _metric_level_adj(P_i: _NDArray, P_j: _NDArray, p_t: _NDArray,
                     D_i: _NDArray, D_j: _NDArray) -> float:
    """L2 norm of (P · (p − Δ)) difference (Remark 5 ad-valorem-tax target)."""
    diff = P_i @ (p_t - D_i) - P_j @ (p_t - D_j)
    return float(np.linalg.norm(diff))


_PASSTHROUGH_FEATURE_METRICS: Dict[str, Any] = {
    'offdiag_ratio': _metric_offdiag_ratio,
    'full_pass': _metric_full_pass,
    'row_sum': _metric_row_sum,
    'level_adj': _metric_level_adj,
}


# rc8 perf: batched feature metrics.
#
# The scalar variants above each take per-market (P_i, P_j, p_t, D_i, D_j)
# and return a scalar. compute_passthrough_summary used to loop over
# (n_pairs × n_markets) and call each metric one market at a time —
# 6 × 3000 × 4 = 72000 small numpy calls on the shipped synthetic.
#
# The batched variants below accept stacked inputs:
#   P_i_batch, P_j_batch : (M, J, J)
#   p_batch              : (M, J)
#   D_i_batch, D_j_batch : (M, J)
# and return an (M,) vector of per-market metric values. Same math,
# same numerical result; just dispatched once per (pair, metric) group
# instead of once per (pair, market, metric).


def _metric_full_pass_batched(
        P_i_batch: _NDArray, P_j_batch: _NDArray, p_batch: _NDArray,
        D_i_batch: _NDArray, D_j_batch: _NDArray,
) -> _NDArray:
    """Frobenius norm of P_i - P_j, per market."""
    diff = P_i_batch - P_j_batch                                    # (M, J, J)
    return np.asarray(np.sqrt(np.sum(diff * diff, axis=(1, 2))))


def _metric_offdiag_ratio_batched(
        P_i_batch: _NDArray, P_j_batch: _NDArray, p_batch: _NDArray,
        D_i_batch: _NDArray, D_j_batch: _NDArray,
        eps_diag: float = 1e-12,
) -> _NDArray:
    """Off-diagonal column-ratio Frobenius distance, per market.

    Mirrors :func:`_metric_offdiag_ratio` (Remark 1 target) but
    vectorized across the market axis. Returns NaN per market where
    any column of P_i or P_j has |diag| <= eps_diag (denominator
    degeneracy, undefined ratio); see the scalar variant's docstring
    for the rationale.
    """
    M, J, _ = P_i_batch.shape
    if J <= 1:
        return np.zeros(M)
    diag_i = np.diagonal(P_i_batch, axis1=1, axis2=2).astype(float)  # (M, J)
    diag_j = np.diagonal(P_j_batch, axis1=1, axis2=2).astype(float)
    # Per-market degeneracy mask: any column has |diag| <= eps in either matrix.
    deg_i = (np.abs(diag_i) <= eps_diag).any(axis=1)                 # (M,)
    deg_j = (np.abs(diag_j) <= eps_diag).any(axis=1)
    degenerate = deg_i | deg_j                                       # (M,)
    # Use safe denominators for the bulk computation; the deg mask
    # overrides the result for degenerate markets below.
    safe_diag_i = np.where(np.abs(diag_i) > eps_diag, diag_i, 1.0)
    safe_diag_j = np.where(np.abs(diag_j) > eps_diag, diag_j, 1.0)
    ratio_i = P_i_batch / safe_diag_i[:, np.newaxis, :]              # (M, J, J)
    ratio_j = P_j_batch / safe_diag_j[:, np.newaxis, :]
    diff = ratio_i - ratio_j
    diag_idx = np.arange(J)
    diff[:, diag_idx, diag_idx] = 0.0
    out = np.sqrt(np.sum(diff * diff, axis=(1, 2)))
    out = np.where(degenerate, np.nan, out)
    return np.asarray(out)


def _metric_row_sum_batched(
        P_i_batch: _NDArray, P_j_batch: _NDArray, p_batch: _NDArray,
        D_i_batch: _NDArray, D_j_batch: _NDArray,
) -> _NDArray:
    """L2 norm of row-sum difference, per market."""
    diff = P_i_batch - P_j_batch                                    # (M, J, J)
    row_diff = diff.sum(axis=2)                                     # (M, J)
    return np.asarray(np.sqrt(np.sum(row_diff * row_diff, axis=1)))


def _metric_level_adj_batched(
        P_i_batch: _NDArray, P_j_batch: _NDArray, p_batch: _NDArray,
        D_i_batch: _NDArray, D_j_batch: _NDArray,
) -> _NDArray:
    """L2 norm of (P · (p − Δ)) difference, per market."""
    # (p - D_i) and (p - D_j) are (M, J); right-multiply by P gives (M, J).
    vi = (p_batch - D_i_batch)[..., np.newaxis]                     # (M, J, 1)
    vj = (p_batch - D_j_batch)[..., np.newaxis]
    Pi_v = np.matmul(P_i_batch, vi)[..., 0]                          # (M, J)
    Pj_v = np.matmul(P_j_batch, vj)[..., 0]
    diff = Pi_v - Pj_v
    return np.asarray(np.sqrt(np.sum(diff * diff, axis=1)))


_PASSTHROUGH_FEATURE_METRICS_BATCHED: Dict[str, Any] = {
    'offdiag_ratio': _metric_offdiag_ratio_batched,
    'full_pass': _metric_full_pass_batched,
    'row_sum': _metric_row_sum_batched,
    'level_adj': _metric_level_adj_batched,
}


_FEATURE_NOTES: Dict[str, str] = {
    'offdiag_ratio': (
        "rival cost shifters: γ-free; column ratios. Zero ⇒ structural "
        "degeneracy. Magnitude doesn't predict power."
    ),
    'full_pass': (
        "own+rival cost; product chars under linear-index demand: γ_known "
        "scaled or γ=0 by rival exclusion; full pass-through difference."
    ),
    'row_sum': (
        "unit tax: row sums of pass-through. ν observed; fully computable."
    ),
    'level_adj': (
        "ad valorem tax: ‖P_m·(p−Δ_m) − P_m'·(p−Δ_m')‖. ν, p observed."
    ),
}


class PassthroughSummary:
    """Output of :meth:`pyRVtest.Problem.passthrough_summary`.

    Holds the per-pair pass-through-feature distance frame, optional
    per-model structural block, candidate labels, methodology line, and
    a ``__repr__`` that produces the formatted printable view.

    The DataFrame is exposed via :attr:`pair_distances` (always populated)
    and :attr:`per_model` (populated only when ``with_models=True``);
    :meth:`to_dataframe` returns a copy of ``pair_distances``.
    """

    def __init__(
        self,
        pair_distances: 'pd.DataFrame',
        per_model: Optional['pd.DataFrame'],
        detail_mode: str,
        n_markets: int,
        candidate_labels: List[str],
        methodology_line: str,
    ) -> None:
        self.pair_distances = pair_distances
        self.per_model = per_model
        self.detail_mode = detail_mode
        self.n_markets = n_markets
        self.candidate_labels = list(candidate_labels)
        self.methodology_line = methodology_line

    def to_dataframe(self) -> 'pd.DataFrame':
        return self.pair_distances.copy()

    def __repr__(self) -> str:
        lines: List[str] = []

        # Per-model block.
        if self.per_model is not None:
            lines.append(
                f"Per-model pass-through structure "
                f"(median across {self.n_markets} markets):"
            )
            lines.append("")
            lines.append(self.per_model.to_string(index=False))
            lines.append("")

        # Pair distance table.
        agg_label = (
            f"median across {self.n_markets} markets"
            if self.detail_mode == 'median'
            else f"per-market over {self.n_markets} markets"
        )
        lines.append(f"Per-pair pass-through-feature distances ({agg_label}):")
        lines.append("")

        # Hide internal model_i / model_j columns from the printed view.
        display_cols = [
            c for c in self.pair_distances.columns
            if c not in ('model_i', 'model_j')
        ]
        display_df = self.pair_distances[display_cols]
        lines.append(display_df.to_string(index=False))
        lines.append("")

        # Per-feature notes.
        lines.append("Per-feature notes:")
        feature_cols = [c for c in display_cols if c in _FEATURE_NOTES]
        for feature in feature_cols:
            note = _FEATURE_NOTES[feature]
            lines.append(f"  {feature} ({note.split(':', 1)[0]}):")
            lines.append(f"    {note.split(':', 1)[1].strip()}")
        lines.append("")

        # Methodology line.
        lines.append(self.methodology_line)

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.__repr__()


def _build_methodology_line(problem: Any) -> str:
    """One-line description of how P_m was computed, conditional on the
    candidate-set composition."""
    candidates = problem._models

    has_vertical = any(isinstance(c, Vertical) for c in candidates)
    trivial_classes = (
        PerfectCompetition, ConstantMarkup, UserSuppliedMarkups, RuleOfThumb,
    )
    has_trivial = any(isinstance(c, trivial_classes) for c in candidates)
    has_numerical = any(
        not isinstance(c, Vertical) and not isinstance(c, trivial_classes)
        for c in candidates
    )

    parts = []
    if has_vertical:
        parts.append("Villas-Boas analytical (Vertical candidates)")
    if has_numerical:
        parts.append("central-difference numerical, delta=1e-7 (Bertrand / Cournot / Monopoly / PartialCollusion / MixCournotBertrand / CustomConductModel)")
    if has_trivial:
        parts.append(
            "exact via short-circuit (PerfectCompetition / ConstantMarkup / "
            "UserSuppliedMarkups / RuleOfThumb)"
        )

    body = "; ".join(parts)

    return (
        f"Methodology — pass-through: {body}. "
        f"See passthrough_summary() docstring and docs/math.rst."
    )


# ===========================================================================
# rc14: Pass-through reliability vocabulary (Audit 2 Findings C2 + C3).
# ===========================================================================
#
# Three concerns the audit flagged on the PT layer:
#
#   1. cond(I - dDelta/dp) can be large for some (model, market) cells,
#      meaning the inverse is numerically unstable. The reported P_m may
#      then be a poor estimate of the analytical pass-through matrix.
#   2. A reported PT distance of zero is ambiguous between true
#      structural degeneracy (the diagnostic's intended signal),
#      near-zero diagnostic denominator, and ill-conditioned numerical
#      derivative.
#   3. Vertical / analytical short-circuits and numerical-central-
#      difference paths have different reliability profiles, and the
#      user has no way to tell which one fired.
#
# This module adds a passive diagnostic — it does not change any
# computed value. ``compute_passthrough_reliability`` walks
# (model, market) cells, calls the existing ``build_passthrough`` for
# each model (so the per-market P matrices match the live diagnostic
# outputs exactly), and reports condition number / rank / status /
# method per cell. Threshold policy (cond_warn, cond_severe,
# cond_undefined) is documented on the user-facing method.
#
# cond identity: cond(P) = cond((I - dDelta/dp)^{-1}) = cond(I - dDelta/dp)
# for the non-vertical paths. For the Vertical Villas-Boas form
# P = inv(G) * H, cond(P) is bounded by cond(G) * cond(H); we report
# cond(P) as the user-visible quantity and document the interpretation.

_PT_RELIABILITY_COLUMNS: List[str] = [
    'model_index', 'model', 'market_id',
    'pt_method', 'pt_condition_number', 'pt_rank',
    'pt_status', 'pt_warning',
]


def _classify_condition_number(
        cond_val: float,
        rank: int,
        J: int,
        cond_warn: float,
        cond_severe: float,
        cond_undefined: float,
) -> Tuple[str, str]:
    """Map (cond, rank, J) to a (status, warning_text) pair.

    Returns one of four status strings:
    - 'robust'          : cond < cond_warn, rank == J.
    - 'ill-conditioned' : cond_warn <= cond < cond_severe.
    - 'near-degenerate' : cond_severe <= cond < cond_undefined.
    - 'undefined'       : cond >= cond_undefined, rank < J, or non-finite cond.
    """
    if not np.isfinite(cond_val) or cond_val >= cond_undefined:
        return 'undefined', (
            f"pass-through matrix is numerically singular "
            f"(cond={cond_val:.2e}); reported P_m is unreliable."
        )
    if rank < J:
        return 'undefined', (
            f"pass-through matrix has rank {rank} < J={J}; "
            f"reported P_m is unreliable."
        )
    if cond_val >= cond_severe:
        return 'near-degenerate', (
            f"pass-through matrix is near-singular "
            f"(cond={cond_val:.2e}); P_m entries lose ~"
            f"{int(np.log10(cond_val))} digits of precision."
        )
    if cond_val >= cond_warn:
        return 'ill-conditioned', (
            f"pass-through matrix is ill-conditioned "
            f"(cond={cond_val:.2e}); ~"
            f"{int(np.log10(cond_val))} digits of precision lost."
        )
    return 'robust', ""


def _classify_pt_method(candidate: Any, is_vertical: bool) -> str:
    """Return the method name reported in the ``pt_method`` column."""
    if is_vertical:
        return 'analytical_vertical'
    if isinstance(
            candidate,
            (PerfectCompetition, ConstantMarkup, RuleOfThumb, UserSuppliedMarkups),
    ):
        return 'analytical_trivial'
    return 'numerical_central_difference'


def compute_passthrough_reliability(
        problem: Any,
        *,
        market_id: Optional[Hashable] = None,
        cond_warn: float = 1e6,
        cond_severe: float = 1e12,
        cond_undefined: float = 1e16,
) -> 'pd.DataFrame':
    r"""Per-(model, market) numerical reliability diagnostic for PT matrices.

    Reports, for each candidate-conduct model and each market, the
    condition number and rank of the pass-through matrix
    :math:`P_m = (I - \partial \Delta_m / \partial p)^{-1}` along
    with a human-readable status. Lets the user distinguish

    1. **structural** pass-through degeneracy (the
       :meth:`Problem.passthrough_summary` distance is zero because
       :math:`P_i` and :math:`P_j` are genuinely identical under the
       chosen instrument type — Audit 2 Finding C3 case "true
       structural degeneracy");
    2. **numerical** instability in the derivative or its inverse
       (an ill-conditioned :math:`I - \partial \Delta_m / \partial p`
       — Audit 2 Finding C3 case "ill-conditioned numerical derivative").

    The diagnostic does NOT change any computed pass-through value; it
    reports condition numbers of the matrices the existing PT methods
    already invert. Bit-identical to pre-rc14 behavior on all PT
    methods.

    Parameters
    ----------
    problem : pyRVtest.Problem
        Initialized Problem.
    market_id : hashable, optional
        Restrict the report to a single market; default returns every
        market.
    cond_warn : float, optional
        Condition-number threshold above which ``pt_status`` is
        ``'ill-conditioned'``. Default ``1e6`` (loses ~6 digits of
        precision relative to double).
    cond_severe : float, optional
        Threshold above which ``pt_status`` is ``'near-degenerate'``.
        Default ``1e12``.
    cond_undefined : float, optional
        Threshold above which ``pt_status`` is ``'undefined'`` (so is
        any rank deficiency or non-finite cond). Default ``1e16``,
        i.e. effective machine-precision singularity.

    Returns
    -------
    pandas.DataFrame
        One row per (candidate model, market). Columns:

        * ``model_index`` (int): row index in ``problem._models``.
        * ``model`` (str): class name of the candidate.
        * ``market_id``: passes through from
          ``problem.unique_market_ids``.
        * ``pt_method`` (str): which path produced :math:`P_m` —
          ``'analytical_trivial'`` (PC / ConstantMarkup /
          UserSuppliedMarkups / RuleOfThumb give exact closed forms),
          ``'analytical_vertical'`` (Villas-Boas), or
          ``'numerical_central_difference'`` (Bertrand / Cournot /
          Monopoly / PartialCollusion / MixCournotBertrand /
          CustomConductModel).
        * ``pt_condition_number`` (float): :math:`\kappa(P_m)`. For
          the non-vertical paths this equals
          :math:`\kappa(I - \partial \Delta_m / \partial p)`; for the
          Vertical Villas-Boas path it bounds the conditioning of the
          :math:`G` matrix that gets inverted.
        * ``pt_rank`` (int): numerical rank of :math:`P_m`.
        * ``pt_status`` (str): one of
          ``'robust'`` / ``'ill-conditioned'`` /
          ``'near-degenerate'`` / ``'undefined'`` (see threshold
          arguments).
        * ``pt_warning`` (str): empty string when ``pt_status ==
          'robust'``, otherwise a one-line description of the issue.

    Examples
    --------
    >>> df = problem.passthrough_reliability()  # doctest: +SKIP
    >>> df[df.pt_status != 'robust']            # doctest: +SKIP

    See Also
    --------
    Problem.passthrough_summary
    Problem.instrument_channels
    """
    import pandas as pd

    n_models = len(problem._models)
    if market_id is not None:
        if not np.any(problem.unique_market_ids == market_id):
            raise ValueError(
                f"Expected market_id to appear in "
                f"problem.unique_market_ids. Received market_id="
                f"{market_id!r}, which is not in "
                f"problem.unique_market_ids={list(problem.unique_market_ids)}. "
                f"Fix: pass a market id from problem.unique_market_ids, "
                f"or omit market_id to scan all markets."
            )

    # rc15: thread the rc6-rc11 caches through the per-model
    # build_passthrough calls below. Pre-rc15 this method bypassed
    # them — each model's call re-ran the markups assembly, the
    # per-market demand-derivative computation, and the np.where scan
    # over product_market_ids. On the shipped synthetic that drove
    # ``passthrough_reliability()`` to ~5.8s vs ~0.5s for the cached
    # ``passthrough_summary()``. Audit 2 rc14 re-audit Finding 4.
    markups_full, markups_downstream, _ = problem._perturb_and_build_markups()
    _backend = problem._demand_backend
    _all_trivial = all(
        not (isinstance(c, Vertical) or problem.models['models_upstream'][k] is not None)
        and isinstance(c, (PerfectCompetition, ConstantMarkup, RuleOfThumb, UserSuppliedMarkups))
        for k, c in enumerate(problem._models)
    )
    demand_derivs: Dict[Hashable, Tuple[_NDArray, Optional[_NDArray]]] = {}
    if _backend is not None and not _all_trivial:
        for t in problem.unique_market_ids:
            D_t = _backend.compute_jacobian(market_id=t)
            H_t = _backend.compute_hessian(market_id=t)
            demand_derivs[t] = (D_t, H_t)
    product_market_ids = problem.products.market_ids.flatten()
    cached_market_indices = _build_market_index_map(
        product_market_ids, list(problem.unique_market_ids),
    )

    rows: List[Dict[str, Any]] = []
    for m in range(n_models):
        candidate = problem._models[m]
        models_upstream = problem.models['models_upstream']
        is_vertical = (
            isinstance(candidate, Vertical)
            or models_upstream[m] is not None
        )
        method = _classify_pt_method(candidate, is_vertical)
        label = type(candidate).__name__

        pt_dict = build_passthrough(
            problem, model_index=m, market_id=market_id,
            _precomputed_markups=(markups_full, markups_downstream),
            _precomputed_demand_derivatives=(demand_derivs or None),
            _precomputed_market_indices=cached_market_indices,
        )
        if not isinstance(pt_dict, dict):
            # Single-market call returned a bare ndarray.
            pt_dict = {market_id: pt_dict}

        for t, P_t in pt_dict.items():
            P_arr = np.asarray(P_t, dtype=float)
            J_t = P_arr.shape[0]
            # cond + rank can both raise on degenerate inputs; trap and
            # fall through to the 'undefined' status.
            try:
                cond_val = float(np.linalg.cond(P_arr))
            except (np.linalg.LinAlgError, ValueError):
                cond_val = float('inf')
            try:
                rank_val = int(np.linalg.matrix_rank(P_arr))
            except (np.linalg.LinAlgError, ValueError):
                rank_val = 0
            status, warn = _classify_condition_number(
                cond_val, rank_val, J_t,
                cond_warn, cond_severe, cond_undefined,
            )
            rows.append({
                'model_index': m,
                'model': label,
                'market_id': t,
                'pt_method': method,
                'pt_condition_number': cond_val,
                'pt_rank': rank_val,
                'pt_status': status,
                'pt_warning': warn,
            })

    return pd.DataFrame(rows, columns=_PT_RELIABILITY_COLUMNS)


def compute_passthrough_summary(
    problem: Any,
    with_models: bool = False,
    detail: str = 'median',
) -> PassthroughSummary:
    """Compute pair × pass-through-feature distance summary across all
    candidate model pairs.

    For each unordered pair ``(m, m')`` of candidate models in
    ``problem._models`` and each market in ``problem.unique_market_ids``,
    computes four pass-through-feature distances:

    - ``offdiag_ratio`` (Remark 1, rival cost shifters): Frobenius norm of
      column-normalized off-diagonal differences ``(P_m)_{jℓ}/(P_m)_{ℓℓ}
      − (P_m')_{jℓ}/(P_m')_{ℓℓ}``.
    - ``full_pass`` (Remark 2, own+rival cost; product chars under
      linear-index demand): Frobenius norm of ``P_m − P_m'``.
    - ``row_sum`` (Remark 5, unit tax): L2 norm of ``(P_m − P_m') · ι``.
    - ``level_adj`` (Remark 5, ad valorem tax): L2 norm of
      ``P_m·(p − Δ_m) − P_m'·(p − Δ_m')``.

    Per-market values are aggregated across markets via median (default)
    or returned per-market when ``detail='full'``.

    Parameters
    ----------
    problem : pyRVtest.Problem
        Constructed Problem. Must have a demand backend (markups/Hessian
        available).
    with_models : bool, optional
        If True, also compute a per-model summary block (median diagonal,
        max off-diagonal, median row sum across markets).
    detail : {'median', 'full'}, optional
        Aggregation mode for the pair-distance table. Default 'median'.

    Returns
    -------
    PassthroughSummary
        Structured result. ``__repr__`` produces the printable view;
        ``to_dataframe`` returns the per-pair distance DataFrame.
    """
    import pandas as pd

    if detail not in ('median', 'full'):
        raise ValueError(
            f"Expected detail in {{'median', 'full'}}. Received "
            f"detail={detail!r}."
        )

    n_models = len(problem._models)
    if n_models < 2:
        raise ValueError(
            f"passthrough_summary requires at least 2 candidate models for "
            f"per-pair distances. Received n_models={n_models}."
        )

    market_ids = list(problem.unique_market_ids)

    # Compute markups per candidate (per-observation) once and thread the
    # full tuple through every per-model build_passthrough call below.
    # Pre-rc6 the inner build_passthrough call would re-invoke
    # _perturb_and_build_markups, multiplying the markups-assembly cost
    # by n_models. On large markets (audit measured ~50s at T=3000,
    # n_models=4) the redundant rebuilds dominate.
    markups_full, markups_downstream, _ = problem._perturb_and_build_markups()

    # Observed prices per observation; product_market_ids gives row → market mapping.
    prices = np.asarray(problem.products.prices, dtype=float).flatten()
    product_market_ids = problem.products.market_ids.flatten()

    # Hoist per-market demand derivatives out of the model loop. The
    # Jacobian and Hessian depend only on demand, not on the candidate
    # conduct, so caching them across models saves (n_models - 1) ×
    # markets calls. The Jacobian is already cached at the backend
    # level for built-in backends, but the slice-per-market work is
    # noticeable for large T; the Hessian is not cached, so this is
    # the dominant rc6 → rc7 saving. Skip Hessian work for trivial
    # closed-form candidates (PC / ConstantMarkup / etc.).
    _backend = problem._demand_backend
    _all_trivial = all(
        not (isinstance(c, Vertical) or problem.models['models_upstream'][k] is not None)
        and isinstance(c, (PerfectCompetition, ConstantMarkup, RuleOfThumb, UserSuppliedMarkups))
        for k, c in enumerate(problem._models)
    )
    demand_derivs: Dict[Hashable, Tuple[_NDArray, Optional[_NDArray]]] = {}
    if _backend is not None and not _all_trivial:
        for t in market_ids:
            D_t = _backend.compute_jacobian(market_id=t)
            H_t = _backend.compute_hessian(market_id=t)
            demand_derivs[t] = (D_t, H_t)

    # rc9 perf: per-market indices computed once and threaded into every
    # build_passthrough call. Pre-rc9 build_passthrough's per-market
    # loop did its own ``np.where(product_market_ids == t)`` for each
    # (model, market) cell — ``n_models × n_markets`` O(N) scans.
    # Hoisting + the groupby form below converts the build to a single
    # O(N log N) pass instead of ``n_markets × O(N) = O(N²)``.
    market_index_map = _build_market_index_map(product_market_ids, market_ids)

    # Per-(model, market) pass-through matrices, computed once.
    per_model_passthrough: Dict[int, Dict[Hashable, _NDArray]] = {}
    for m in range(n_models):
        # build_passthrough returns dict[market_id, ndarray] when called
        # without market_id — narrowing the union type for mypy.
        full_dict = build_passthrough(
            problem, model_index=m,
            _precomputed_markups=(markups_full, markups_downstream),
            _precomputed_demand_derivatives=(demand_derivs or None),
            _precomputed_market_indices=market_index_map,
        )
        assert isinstance(full_dict, dict)
        per_model_passthrough[m] = full_dict

    # Candidate labels (class name; could expand later).
    labels = [type(c).__name__ for c in problem._models]

    # Per-market data extracted once. Reuse the index map built above.
    market_data: Dict[Hashable, Dict[str, _NDArray]] = {}
    for t in market_ids:
        idx = market_index_map[t]
        market_data[t] = {
            'idx': idx,
            'p': prices[idx],
        }

    # Per-pair distance computation.
    pair_rows: List[Dict[str, Any]] = []
    metric_names = list(_PASSTHROUGH_FEATURE_METRICS.keys())

    # rc8 perf: group markets by J_t so we can dispatch one batched
    # numpy call per (pair, metric, J-group) instead of one call per
    # (pair, market, metric). For uniform-J data (the common case)
    # there is one bucket; for variable-J data the per-J grouping
    # preserves market_ids ordering inside each bucket and re-merges
    # at the end so output rows still come out in market_ids order.
    market_J: Dict[Hashable, int] = {
        t: int(np.asarray(per_model_passthrough[0][t], dtype=float).shape[0])
        for t in market_ids
    }
    markets_by_J: Dict[int, List[Hashable]] = {}
    for t in market_ids:
        markets_by_J.setdefault(market_J[t], []).append(t)

    # Per-model stacked passthrough / markups for each J-bucket, computed once.
    # Keyed (m, J_t) -> (M_g, J_t, J_t) for passthrough, (M_g, J_t) for markups.
    pmt_stacks: Dict[Tuple[int, int], _NDArray] = {}
    mk_stacks: Dict[Tuple[int, int], _NDArray] = {}
    p_stacks: Dict[int, _NDArray] = {}
    for J_t, t_list in markets_by_J.items():
        p_stacks[J_t] = np.stack([market_data[t]['p'].astype(float) for t in t_list])
        for m in range(n_models):
            pmt_stacks[(m, J_t)] = np.stack([
                np.asarray(per_model_passthrough[m][t], dtype=float)
                for t in t_list
            ])
            mk_stacks[(m, J_t)] = np.stack([
                np.asarray(markups_full[m][market_data[t]['idx']], dtype=float).flatten()
                for t in t_list
            ])

    for i in range(n_models):
        for j in range(i + 1, n_models):
            # Per-market value lists in market_ids order. We fill them
            # bucket-by-bucket, then assemble at the end.
            per_market_metrics: Dict[str, List[float]] = {m: [] for m in metric_names}
            # Scratch storage indexed by t for the bucket-merge below.
            per_t_metric: Dict[str, Dict[Hashable, float]] = {m: {} for m in metric_names}
            for J_t, t_list in markets_by_J.items():
                P_i_batch = pmt_stacks[(i, J_t)]
                P_j_batch = pmt_stacks[(j, J_t)]
                p_batch = p_stacks[J_t]
                D_i_batch = mk_stacks[(i, J_t)]
                D_j_batch = mk_stacks[(j, J_t)]
                for name, metric_batch_fn in _PASSTHROUGH_FEATURE_METRICS_BATCHED.items():
                    vals_batch = metric_batch_fn(
                        P_i_batch, P_j_batch, p_batch, D_i_batch, D_j_batch,
                    )
                    for k_in_bucket, t in enumerate(t_list):
                        per_t_metric[name][t] = float(vals_batch[k_in_bucket])
            # Re-emit in market_ids order so downstream median /
            # 'full' detail behavior is order-stable.
            for name in metric_names:
                per_market_metrics[name] = [per_t_metric[name][t] for t in market_ids]

            if detail == 'median':
                row = {
                    'pair': f"({labels[i]}, {labels[j]})",
                    'model_i': i,
                    'model_j': j,
                }
                for name in metric_names:
                    vals = np.asarray(per_market_metrics[name], dtype=float)
                    # rc15: any metric can be NaN-valued for a given
                    # market when its underlying feature is undefined
                    # (currently only ``offdiag_ratio`` has this case —
                    # denominator degeneracy when |diag(P_m)[ℓ]| <=
                    # eps for any column). Use ``nanmedian`` so a
                    # well-defined majority of markets still aggregates
                    # cleanly, and emit a count of degenerate markets
                    # so the user can tell whether the median is
                    # representative. Audit 2 rc14 re-audit Finding 3.
                    n_degen = int(np.isnan(vals).sum())
                    if n_degen == len(vals):
                        row[name] = float('nan')
                    else:
                        row[name] = float(np.nanmedian(vals))
                    row[f'{name}_n_degenerate'] = n_degen
                pair_rows.append(row)
            else:  # detail == 'full'
                for k, t in enumerate(market_ids):
                    row = {
                        'pair': f"({labels[i]}, {labels[j]})",
                        'model_i': i,
                        'model_j': j,
                        'market_id': t,
                    }
                    for name in metric_names:
                        row[name] = per_market_metrics[name][k]
                    pair_rows.append(row)

    pair_distances = pd.DataFrame(pair_rows)

    # Per-model block.
    per_model_df: Optional[pd.DataFrame] = None
    if with_models:
        rows: List[Dict[str, Any]] = []
        for m in range(n_models):
            diag_vals: List[float] = []
            offdiag_max_vals: List[float] = []
            row_sum_vals: List[float] = []
            for t in market_ids:
                P_m = np.asarray(per_model_passthrough[m][t], dtype=float)
                diag_vals.extend(np.diag(P_m).tolist())
                if P_m.shape[0] > 1:
                    offdiag = P_m - np.diag(np.diag(P_m))
                    if np.any(offdiag != 0):
                        # Signed max-magnitude entry for visual cue.
                        flat = offdiag[~np.eye(offdiag.shape[0], dtype=bool)]
                        idx_max = int(np.argmax(np.abs(flat)))
                        offdiag_max_vals.append(float(flat[idx_max]))
                    else:
                        offdiag_max_vals.append(0.0)
                row_sum_vals.extend(
                    (P_m @ np.ones(P_m.shape[0])).tolist()
                )
            rows.append({
                'model': labels[m],
                'diag_avg': float(np.median(diag_vals)),
                'max_offdiag': (
                    float(np.median(offdiag_max_vals))
                    if offdiag_max_vals else 0.0
                ),
                'row_sum_avg': float(np.median(row_sum_vals)),
            })
        per_model_df = pd.DataFrame(rows)

    methodology_line = _build_methodology_line(problem)

    return PassthroughSummary(
        pair_distances=pair_distances,
        per_model=per_model_df,
        detail_mode=detail,
        n_markets=len(market_ids),
        candidate_labels=labels,
        methodology_line=methodology_line,
    )


# ===========================================================================
# Phase 3: instrument_channels — per-pair channel decomposition
# ===========================================================================


def _fwl_residualize(y: _NDArray, X: _NDArray) -> _NDArray:
    """Residualize ``y`` on the columns of ``X`` (with implicit intercept).

    OLS partialling-out. Returns the residual vector. ``X`` may be a
    column matrix or a 2D array; an intercept column is added internally.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    intercept = np.ones((X.shape[0], 1))
    X_full = np.hstack([intercept, X])
    # OLS coefficients via lstsq (robust to near-singular X).
    beta, *_ = np.linalg.lstsq(X_full, y, rcond=None)
    return np.asarray(y - X_full @ beta)


def _ols_slope_partialled(y: _NDArray, z: _NDArray, controls: _NDArray) -> float:
    """OLS coefficient on ``z`` in a regression of ``y`` on ``z`` with
    ``controls`` (intercept added). Computed via FWL: residualize ``y`` and
    ``z`` on ``controls``, then take the simple bivariate slope."""
    y_res = _fwl_residualize(y, controls)
    z_res = _fwl_residualize(z, controls)
    var_z = float(np.var(z_res))
    if var_z < 1e-30:
        return 0.0
    return float(np.cov(y_res, z_res, ddof=0)[0, 1] / var_z)


class InstrumentChannels:
    """Output of :meth:`pyRVtest.Problem.instrument_channels`.

    Holds the per-pair channel decomposition (indirect, direct, total)
    plus the data-side empirical magnitude, the per-candidate direct
    coefficients, the methodology line, and a ``__repr__`` that produces
    the formatted printable view.
    """

    def __init__(
        self,
        column_name: str,
        instrument_type: Optional[str],
        dp0_dz_obs: float,
        sd_z: float,
        z_min: float,
        z_max: float,
        per_candidate: 'pd.DataFrame',
        pair_distances: 'pd.DataFrame',
        n_markets: int,
        candidate_labels: List[str],
        methodology_line: str,
    ) -> None:
        self.column_name = column_name
        self.instrument_type = instrument_type
        self.dp0_dz_obs = dp0_dz_obs
        self.sd_z = sd_z
        self.z_min = z_min
        self.z_max = z_max
        self.per_candidate = per_candidate
        self.pair_distances = pair_distances
        self.n_markets = n_markets
        self.candidate_labels = list(candidate_labels)
        self.methodology_line = methodology_line

    def to_dataframe(self) -> 'pd.DataFrame':
        return self.pair_distances.copy()

    def __repr__(self) -> str:
        lines: List[str] = []

        header = f"Post-solve instrument-channel decomposition: column {self.column_name!r}"
        if self.instrument_type is not None:
            header += f" (declared type: {self.instrument_type!r})"
        header += "."
        lines.append(header)
        lines.append(f"γ_m fitted from solve.")
        lines.append("")

        lines.append(
            "Channel components (combine per instrument type):"
        )
        lines.append(
            "  indirect = P_m · (P_m^{-1} − P_m'^{-1}) · (dp_0/dz)"
        )
        lines.append(
            "                    └── structural ──┘   └── data ──┘"
        )
        lines.append(
            "  direct = β_m − β_m'  (empirical OLS partialling on prices)"
        )
        lines.append("")

        # Data side.
        lines.append("Data-side: empirical effect of z on prices")
        lines.append(
            f"  ‖dp_0/dz‖_obs = {self.dp0_dz_obs:.4f}     "
            f"(sample regression slope of p on {self.column_name})"
        )
        lines.append(f"  SD({self.column_name})  = {self.sd_z:.4f}")
        lines.append(f"  range          = [{self.z_min:.4f}, {self.z_max:.4f}]")
        lines.append("")

        # Per-candidate direct channel.
        lines.append("Direct channel: per-candidate β_m (OLS slope of Δ_m on z | p)")
        lines.append(self.per_candidate.to_string(index=False))
        lines.append("")

        # Per-pair table.
        lines.append(f"Per-pair channel components (median across {self.n_markets} markets):")
        lines.append("")
        display_cols = [
            c for c in self.pair_distances.columns
            if c not in ('model_i', 'model_j')
        ]
        lines.append(self.pair_distances[display_cols].to_string(index=False))
        lines.append("")
        lines.append(
            "  structural: ‖P_m^{-1} − P_m'^{-1}‖_F, γ-free; multiply by the "
            "instrument-specific projection of dp_0/dz to get indirect channel."
        )
        lines.append(
            "  direct: |β_m − β_m'|, empirical magnitude difference."
        )
        lines.append("")

        # Methodology.
        lines.append(self.methodology_line)

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.__repr__()


def _build_instrument_channels_methodology_line(
    problem: Any, instrument_type: Optional[str],
) -> str:
    """Methodology line for instrument_channels output."""
    base = _build_methodology_line(problem)  # passthrough computation method
    if problem.endogenous_cost_component is not None:
        # Non-constant MC: data-side regression and FWL partialling use
        # z residualized on (g(q_tilde), w_exog) — DMQSS Appendix B's
        # z^e — so a single magnitude simultaneously reflects the
        # instrument-relevance condition (DMQSW) and the economic
        # distinctness condition (DMQSS Appendix A.4) under non-constant
        # cost.
        iv_note = (
            "Data-side ‖dp_0/dz‖_obs and direct channel β_m use z residualized "
            "on (g(q̃), w_exog) — DMQSS z^e — which simultaneously reflects the "
            "instrument-relevance condition (DMQSW) and the economic distinctness "
            "condition (DMQSS Appendix A.4). Magnitudes are finite-sample "
            "sample-regression estimates; small-but-nonzero values may reflect "
            "noise rather than population identifying variation. Structural-side "
            "‖P_m^{-1} − P_m'^{-1}‖ is γ-free, computed from the candidate "
            "pass-through matrices as above. See instrument_channels() docstring "
            "+ docs/math.rst."
        )
    else:
        # Constant MC: standard Dearing decomposition with cost-formulation controls.
        iv_note = (
            "Direct channel β_m via conditional regression of Δ_m on z controlling "
            "for p (FWL); data-side ‖dp_0/dz‖_obs via regression of observed prices "
            "on z with cost-formulation controls; structural-side ‖P_m^{-1} − P_m'^{-1}‖ "
            "from the candidate pass-through matrices computed as above. "
            "See instrument_channels() docstring for the channel decomposition formula."
        )
    if instrument_type is not None:
        iv_note += (
            f" Declared instrument type: {instrument_type!r} — see "
            f"docs/math.rst and DMQSW for the targeting interpretation."
        )
    return f"{base}\n{iv_note}"


def compute_instrument_channels(
    problem: Any,
    column: str,
    instrument: Optional[str] = None,
) -> InstrumentChannels:
    """Per-pair channel decomposition components for one instrument column.

    Reports the building blocks of ``dp_m/dz − dp_m'/dz`` separately:

    - **Data-side**: ``‖dp_0/dz‖_obs``, the sample regression slope of
      observed prices on ``z`` controlling for the cost formulation.
      Plus sample ``SD(z)`` and range.
    - **Direct channel**, per candidate: ``β_m`` from OLS regression of
      model-implied ``Δ_m`` on ``z`` with ``p`` as a control (FWL
      partialling). For cost-shifter and tax instruments where the
      analytical direct channel is zero, residual ``β_m`` reflects
      empirical correlation between ``z`` and other markup-function
      inputs (e.g. product characteristics).
    - **Structural-side**, per pair: ``‖P_m^{-1} − P_m'^{-1}‖_F``
      aggregated by median across markets. γ-free.
    - **Direct-side**, per pair: ``|β_m − β_m'|``.

    The full indirect channel ``P_m·(P_m^{-1} − P_m'^{-1})·(dp_0/dz)``
    requires a projection of ``dp_0/dz`` that depends on the instrument's
    targeting (column ℓ of pass-through for rival cost shifter ``z = w_ℓ``;
    row sums for unit tax; ``P·(p−Δ)`` for ad valorem tax). We report
    the structural-side magnitude as a γ-free building block and let the
    user apply the appropriate projection per their instrument type.

    Parameters
    ----------
    problem : pyRVtest.Problem
        Constructed Problem.
    column : str
        Name of the IV column in ``problem.products``. Must be a field of
        the products structured array.
    instrument : str, optional
        Declared instrument type for documentation: 'rival_cost',
        'own_rival_cost', 'unit_tax', 'advalorem_tax', 'rival_product_char',
        'own_product_char', or 'composite'. Used in the methodology line;
        does not change the computation.

    Returns
    -------
    InstrumentChannels
    """
    import pandas as pd

    n_models = len(problem._models)
    if n_models < 2:
        raise ValueError(
            f"instrument_channels requires at least 2 candidate models for "
            f"per-pair decomposition. Received n_models={n_models}."
        )

    # Validate column exists.
    if column not in problem.products.dtype.fields:
        available = [
            f for f in problem.products.dtype.fields
            if not f.startswith('_') and f not in (
                'market_ids', 'cost_ids', 'nesting_ids', 'product_ids',
                'clustering_ids', 'shares', 'prices', 'w', 'Z0',
            )
        ]
        raise ValueError(
            f"Expected column={column!r} to be a field of "
            f"problem.products. Received an unknown column. "
            f"Available columns: {sorted(available)}. "
            f"Fix: pass a column name from the data passed to "
            f"product_data= at Problem construction."
        )

    market_ids = list(problem.unique_market_ids)
    n_markets = len(market_ids)

    # Per-observation z, prices, cost-formulation controls (w).
    z = np.asarray(problem.products[column], dtype=float).flatten()
    prices = np.asarray(problem.products['prices'], dtype=float).flatten()
    w = np.asarray(problem.products['w'], dtype=float)
    if w.ndim == 1:
        w = w.reshape(-1, 1)

    # When endogenous_cost_component is set, the data-side regression
    # should partial z on (g(q_tilde), w_exog) — DMQSS Appendix B's
    # z^e residualization — rather than on raw (q, w_exog). Using raw q
    # over-residualizes z along the endogeneity direction (q is
    # correlated with the cost shock omega) and biases the empirical
    # slope. Using the first-stage prediction q_tilde absorbs only the
    # exogenous component of q in z, so the resulting magnitude is the
    # unified Dearing + DMQSS A.4 distinctness diagnostic (rank-K+1
    # condition collapses to first-stage rank K + nonzero
    # Cov(z^e, model-implied cost difference); see math.rst).
    #
    # The first stage uses the FULL declared instrument set (combined
    # across instrument_formulation entries) plus w_exog, NOT the single
    # diagnostic column z. Using only z would make q_tilde span
    # {z, w_exog}, so z would be perfectly explained by (q_tilde, w_exog)
    # and the residualization would degenerate to zero. The full IV set
    # ensures q_tilde captures the exogenous content of q WITHOUT
    # exhausting z's own variation.
    #
    # Constant-MC case: endogenous_cost_component is None, so this
    # block is a no-op and the prior raw-w residualization remains.
    controls_for_data_side = w
    if problem.endogenous_cost_component is not None:
        # Build (g(q_tilde), w_exog) controls. q_tilde is the first-stage
        # OLS prediction of the K_endog endogenous columns on (Z_full,
        # w_exog). Compute here rather than reusing iv_correct so the
        # diagnostic stays callable pre-solve (no IV correction yet).
        name_to_w_idx = {str(f): i for i, f in enumerate(problem._w_formulation)}
        endog_indices = [name_to_w_idx[name] for name in problem._endogenous_cost_columns]
        endog_set = set(endog_indices)
        exog_indices = [
            i for i in range(problem.products.w.shape[1]) if i not in endog_set
        ]
        endog_cols = problem.products.w[:, endog_indices]  # (N, K_endog)
        w_exog = problem.products.w[:, exog_indices]       # (N, K_w - K_endog)
        # Stack every declared instrument set (across all L bundles) for
        # the first stage. Using the union of test IVs matches DMQSS's
        # construction: q_tilde is the projection of g(q^p) onto the
        # column space of all available exogenous variation.
        z_blocks = []
        for l in range(problem.L):
            z_blocks.append(np.asarray(problem.products["Z{0}".format(l)], dtype=float))
        Z_full = np.hstack(z_blocks)                       # (N, sum_l K_inst_l)
        if w_exog.shape[1] > 0:
            first_stage_X = np.hstack([Z_full, w_exog])
        else:
            first_stage_X = Z_full
        Q_fs, _ = np.linalg.qr(first_stage_X, mode='reduced')
        q_tilde = Q_fs @ (Q_fs.T @ endog_cols)              # (N, K_endog)
        # Controls = (q_tilde, w_exog).
        if w_exog.shape[1] > 0:
            controls_for_data_side = np.hstack([q_tilde, w_exog])
        else:
            controls_for_data_side = q_tilde

    # Markups per candidate (per-observation, may be (M, N, 1) shape).
    # Capture downstream too so the inner build_passthrough calls below
    # don't redo the markups assembly (audit Finding 4 / rc6).
    markups_full, markups_downstream, _ = problem._perturb_and_build_markups()

    # Hoist per-market demand derivatives out of the model loop (rc7).
    # Same rationale as compute_passthrough_summary above.
    _backend = problem._demand_backend
    _all_trivial = all(
        not (isinstance(c, Vertical) or problem.models['models_upstream'][k] is not None)
        and isinstance(c, (PerfectCompetition, ConstantMarkup, RuleOfThumb, UserSuppliedMarkups))
        for k, c in enumerate(problem._models)
    )
    demand_derivs: Dict[Hashable, Tuple[_NDArray, Optional[_NDArray]]] = {}
    if _backend is not None and not _all_trivial:
        for t in problem.unique_market_ids:
            D_t = _backend.compute_jacobian(market_id=t)
            H_t = _backend.compute_hessian(market_id=t)
            demand_derivs[t] = (D_t, H_t)

    # rc9 perf: precompute per-market indices once and thread through
    # build_passthrough. Same rationale as compute_passthrough_summary.
    product_market_ids_arr = problem.products.market_ids.flatten()
    market_index_map = _build_market_index_map(
        product_market_ids_arr, list(problem.unique_market_ids),
    )

    # Pre-compute pass-through per (model, market).
    per_model_passthrough: Dict[int, Dict[Hashable, _NDArray]] = {}
    for m in range(n_models):
        full_dict = build_passthrough(
            problem, model_index=m,
            _precomputed_markups=(markups_full, markups_downstream),
            _precomputed_demand_derivatives=(demand_derivs or None),
            _precomputed_market_indices=market_index_map,
        )
        assert isinstance(full_dict, dict)
        per_model_passthrough[m] = full_dict

    labels = [type(c).__name__ for c in problem._models]

    # Data-side: empirical regression slope of prices on z with the
    # appropriate controls (raw w under constant MC; (q_tilde, w_exog)
    # under non-constant MC — DMQSS z^e residualization).
    dp0_dz_obs = _ols_slope_partialled(prices, z, controls_for_data_side)
    sd_z = float(np.std(z))
    z_min = float(np.min(z))
    z_max = float(np.max(z))

    # Direct channel: per-candidate β_m from regression of Δ_m on z given controls.
    # Same controls as the data-side regression: (p) under constant MC,
    # (p, q_tilde, w_exog) under non-constant MC. The non-constant-MC
    # form prevents endogeneity-direction leakage in β_m exactly like
    # it does for dp_0/dz.
    if problem.endogenous_cost_component is not None:
        # Stack p with the (q_tilde, w_exog) controls computed above.
        controls_for_direct = np.hstack([prices.reshape(-1, 1), controls_for_data_side])
    else:
        controls_for_direct = prices
    per_candidate_rows: List[Dict[str, Any]] = []
    betas: List[float] = []
    for m in range(n_models):
        delta_m = np.asarray(markups_full[m], dtype=float).flatten()
        # FWL: regress y on z | controls.
        beta_m = _ols_slope_partialled(delta_m, z, controls_for_direct)
        betas.append(beta_m)
        per_candidate_rows.append({
            'model': labels[m],
            'beta_m': beta_m,
        })
    per_candidate_df = pd.DataFrame(per_candidate_rows)

    # Per-market index map. Reuse the one built above for the
    # build_passthrough loop instead of running np.where again.
    market_idx: Dict[Hashable, _NDArray] = market_index_map

    # Per-pair structural-side and direct-side magnitudes. We report these
    # as separate components rather than collapse into a single "indirect"
    # channel: indirect = P_m·(P_m^{-1}−P_m'^{-1})·(dp_0/dz) requires a
    # projection of dp_0/dz that depends on the instrument's targeting
    # (column ℓ for rival cost shifter, row sums for unit tax, etc.).
    # The user combines structural and direct magnitudes per their
    # instrument type — see the methodology footer.
    pair_rows: List[Dict[str, Any]] = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            structural_per_market: List[float] = []
            for t in market_ids:
                P_i = np.asarray(per_model_passthrough[i][t], dtype=float)
                P_j = np.asarray(per_model_passthrough[j][t], dtype=float)
                # Inverse-pass-through difference, Frobenius (γ-free).
                Pinv_i = np.linalg.inv(P_i)
                Pinv_j = np.linalg.inv(P_j)
                structural_per_market.append(
                    float(np.linalg.norm(Pinv_i - Pinv_j, 'fro'))
                )

            structural_pair = float(np.median(structural_per_market))
            direct_pair = abs(betas[i] - betas[j])

            pair_rows.append({
                'pair': f"({labels[i]}, {labels[j]})",
                'model_i': i,
                'model_j': j,
                'structural': structural_pair,
                'direct': direct_pair,
            })

    pair_distances = pd.DataFrame(pair_rows)

    methodology_line = _build_instrument_channels_methodology_line(problem, instrument)

    return InstrumentChannels(
        column_name=column,
        instrument_type=instrument,
        dp0_dz_obs=dp0_dz_obs,
        sd_z=sd_z,
        z_min=z_min,
        z_max=z_max,
        per_candidate=per_candidate_df,
        pair_distances=pair_distances,
        n_markets=n_markets,
        candidate_labels=labels,
        methodology_line=methodology_line,
    )
