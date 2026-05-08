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

from typing import Any, Dict, Hashable, Optional, Union

import numpy as np
from numpy.linalg import inv
from numpy.typing import NDArray
from typing_extensions import TypeAlias

from ..exceptions import HessianUnavailableError
from ..markups import _construct_passthrough_from_hessian, evaluate_first_order_conditions
from ..models.constant import ConstantMarkup, RuleOfThumb
from ..models.mixed import MixCournotBertrand
from ..models.standard import PerfectCompetition
from ..models.user_supplied import UserSuppliedMarkups
from ..models.vertical import Vertical


__all__ = ['build_passthrough', 'compute_passthrough_numerical']


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
