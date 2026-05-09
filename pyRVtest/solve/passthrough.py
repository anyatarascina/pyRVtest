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

from typing import TYPE_CHECKING, Any, Dict, Hashable, List, Optional, Union

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
    'compute_passthrough_summary',
]


_NDArray: TypeAlias = NDArray[Any]


# ===========================================================================
# Numerical pass-through core
# ===========================================================================


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

    results: Dict[Hashable, _NDArray] = {}
    for t in markets_to_compute:
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

        # Jacobian: backend strips NaN-padded columns on the per-market call.
        D_t = backend.compute_jacobian(market_id=t)

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
            passthrough_t: _NDArray = _construct_passthrough_from_hessian(  # type: ignore[no-untyped-call]
                d2s_dp2_t, D_t, ownership_t, markups_t,
            )
            results[t] = passthrough_t
            continue

        # Non-Vertical: numerical core.
        if needs_hessian:
            d2s_dp2_t = backend.compute_hessian(market_id=t)
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

        shares_t = np.asarray(problem.products.shares[index_t]).flatten()

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
                         D_i: _NDArray, D_j: _NDArray) -> float:
    """Frobenius norm of off-diagonal column-ratio differences (Remark 1 target).

    For each off-diagonal position (j, ℓ) with j ≠ ℓ, the Remark 1 target
    is the column-normalized off-diagonal entry (P)_{jℓ}/(P)_{ℓℓ}. The
    aggregated distance over off-diagonals captures whether rival cost
    shifters can distinguish the pair regardless of γ.
    """
    J = P_i.shape[0]
    if J <= 1:
        return 0.0
    diag_i = np.diag(P_i).astype(float)
    diag_j = np.diag(P_j).astype(float)
    # Avoid division by zero. If a candidate has zero diagonal entry at
    # column ℓ, its column ratio is undefined; we contribute 0 to the
    # distance for that column (caller sees 0 → degenerate via the other
    # candidate's check too).
    safe_diag_i = np.where(np.abs(diag_i) > 1e-12, diag_i, np.inf)
    safe_diag_j = np.where(np.abs(diag_j) > 1e-12, diag_j, np.inf)
    ratio_i = P_i / safe_diag_i[np.newaxis, :]
    ratio_j = P_j / safe_diag_j[np.newaxis, :]
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

    # Compute markups per candidate (per-observation) once.
    markups_full, _, _ = problem._perturb_and_build_markups()

    # Observed prices per observation; product_market_ids gives row → market mapping.
    prices = np.asarray(problem.products.prices, dtype=float).flatten()
    product_market_ids = problem.products.market_ids.flatten()

    # Per-(model, market) pass-through matrices, computed once.
    per_model_passthrough: Dict[int, Dict[Hashable, _NDArray]] = {}
    for m in range(n_models):
        # build_passthrough returns dict[market_id, ndarray] when called
        # without market_id — narrowing the union type for mypy.
        full_dict = build_passthrough(problem, model_index=m)
        assert isinstance(full_dict, dict)
        per_model_passthrough[m] = full_dict

    # Candidate labels (class name; could expand later).
    labels = [type(c).__name__ for c in problem._models]

    # Per-market data extracted once.
    market_data: Dict[Hashable, Dict[str, _NDArray]] = {}
    for t in market_ids:
        idx = np.where(product_market_ids == t)[0]
        market_data[t] = {
            'idx': idx,
            'p': prices[idx],
        }

    # Per-pair distance computation.
    pair_rows: List[Dict[str, Any]] = []
    metric_names = list(_PASSTHROUGH_FEATURE_METRICS.keys())
    for i in range(n_models):
        for j in range(i + 1, n_models):
            per_market_metrics: Dict[str, List[float]] = {m: [] for m in metric_names}
            for t in market_ids:
                P_i = np.asarray(per_model_passthrough[i][t], dtype=float)
                P_j = np.asarray(per_model_passthrough[j][t], dtype=float)
                idx = market_data[t]['idx']
                p_t = market_data[t]['p']
                D_i = np.asarray(markups_full[i][idx], dtype=float).flatten()
                D_j = np.asarray(markups_full[j][idx], dtype=float).flatten()
                for name, metric_fn in _PASSTHROUGH_FEATURE_METRICS.items():
                    per_market_metrics[name].append(
                        metric_fn(P_i, P_j, p_t, D_i, D_j),
                    )

            if detail == 'median':
                row = {
                    'pair': f"({labels[i]}, {labels[j]})",
                    'model_i': i,
                    'model_j': j,
                }
                for name in metric_names:
                    row[name] = float(np.median(per_market_metrics[name]))
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
        # z^e — so the diagnostic unifies the Dearing condition (LHS of
        # DMQSS Eq 10) with the A.4 distinctness check via the rank-K+1
        # decomposition.
        iv_note = (
            "Data-side ‖dp_0/dz‖_obs and direct channel β_m use z residualized "
            "on (g(q̃), w_exog) — DMQSS z^e — so the diagnostic collapses the "
            "Dearing condition and the K+1 distinctness check (Appendix A.4) "
            "into one. Structural-side ‖P_m^{-1} − P_m'^{-1}‖ is γ-free, "
            "computed from the candidate pass-through matrices as above. "
            "See instrument_channels() docstring + docs/math.rst."
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
            f"docs/math.rst and Dearing et al. (2026) for the targeting "
            f"interpretation."
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
    markups_full, _, _ = problem._perturb_and_build_markups()

    # Pre-compute pass-through per (model, market).
    per_model_passthrough: Dict[int, Dict[Hashable, _NDArray]] = {}
    for m in range(n_models):
        full_dict = build_passthrough(problem, model_index=m)
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

    # Per-market index map.
    product_market_ids = problem.products.market_ids.flatten()
    market_idx: Dict[Hashable, _NDArray] = {
        t: np.where(product_market_ids == t)[0] for t in market_ids
    }

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
