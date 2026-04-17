"""Demand-adjustment first-stage correction (DMSS 2024 Appendix C eq. 77).

v0.4 step 4b lands `_residualize_on_xd`, the shared 2SLS profile-out
helper. v0.4 step 4d lands `compute_demand_adjustment`, the unified
implementation that replaces `Problem._compute_analytical_demand_adjustment`
and `Problem._compute_demand_adjustment_gradient`. Both inline methods
are dead code after step 4e wires the unified function into
`Problem.solve`; step 4f deletes them.

The unified function is generic over any `SupportsDemandAdjustment`
backend. `PyBLPBackend` supplies pyblp-side quantities; `LogitBackend`
and `NestedLogitBackend` supply analytical logit / nested-logit
quantities. `UserSuppliedBackend` does not implement
`SupportsDemandAdjustment` and is rejected with a clear error.

Closes a known capability gap: the analytical inline path previously
returned `gradient_gamma_per_instrument=None`, silently disabling the
endogenous-cost gamma correction for `demand_params` users. After
step 4 lands, `demand_params + endogenous_cost_component +
demand_adjustment=True` computes the gamma gradient the same way the
`demand_results` path always has. See Decisions Log in
`.claude/plans/v0.4-refactor.md` for the capability-parity rationale.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray


__all__ = ['_residualize_on_xd', 'compute_demand_adjustment']


_NDArray = NDArray[Any]


def _residualize_on_xd(
        dxi_dtheta: _NDArray, X_D: _NDArray, Z_D: _NDArray, W_D: _NDArray
) -> _NDArray:
    r"""2SLS-residualize ``dxi_dtheta`` on ``X_D`` using ``Z_D`` as instruments.

    Implements the profiled-moment gradient from DMSS Appendix C eq. (77):

    .. math::

        \frac{\partial \xi_{\text{profiled}}}{\partial \theta}
        = \frac{\partial \xi}{\partial \theta}
        - X_D \big( X_D' Z_D W_D Z_D' X_D \big)^{-1} X_D' Z_D W_D Z_D'
          \frac{\partial \xi}{\partial \theta}

    This concentrates out the linear demand-side coefficients (``beta``,
    including the intercept but excluding ``alpha`` which is already in
    ``dxi_dtheta``) at fixed nonlinear parameters ``(alpha, rho, sigma)``.

    If ``X_D`` has zero columns (no exogenous regressors to concentrate
    out), ``dxi_dtheta`` is returned unchanged.

    Parameters
    ----------
    dxi_dtheta
        Raw partial derivative of :math:`\xi` w.r.t. :math:`\theta`, shape
        ``(N, n_theta)``. For the PyBLP path this is the ``xi_by_theta``
        Jacobian with ``-prices`` appended if ``alpha`` is a linear
        parameter. For the analytical logit path this is
        ``(-prices, -log_within_nest_shares)`` stacked.
    X_D
        Non-price demand-side linear regressors (excluding ``prices``),
        shape ``(N, K_x)``. Demand fixed effects should already be
        absorbed.
    Z_D
        Demand instruments, shape ``(N, K_z)``.
    W_D
        GMM weight matrix used in estimation, shape ``(K_z, K_z)``.
        Per DMSS eq. (77) this is the estimation-time weight
        (``r.W``, not ``r.updated_W``).

    Returns
    -------
    ndarray
        Residualized gradient, shape ``(N, n_theta)``.

    Notes
    -----
    The formula is equivalent to ``dxi_dtheta - X_D @ inv(X_D'
    Z_D W_D Z_D' X_D) @ X_D' Z_D W_D Z_D' @ dxi_dtheta`` but uses a
    factored intermediate (``XtZW = X_D' Z_D W_D``) for clarity. Floating-
    point results differ by at most a few ULP from the one-line form due
    to matrix-product non-associativity; unit tests check this to
    ``atol=1e-11``.
    """
    if X_D.shape[1] == 0:
        return dxi_dtheta
    XtZW = X_D.T @ Z_D @ W_D
    M_xx = XtZW @ Z_D.T @ X_D
    projection_coeffs = np.linalg.inv(M_xx) @ (XtZW @ Z_D.T @ dxi_dtheta)
    residual: _NDArray = dxi_dtheta - X_D @ projection_coeffs
    return residual


def compute_demand_adjustment(
        backend: Any,
        problem: Any,
        M: int,
        N: int,
        markups: List[_NDArray],
        advalorem_tax_adj: List[Any],
        cost_scaling: List[Any],
        marginal_cost_base: Optional[List[_NDArray]] = None,
) -> Tuple[_NDArray, _NDArray, _NDArray, _NDArray, _NDArray, Optional[List[_NDArray]]]:
    """Unified DMSS-eq.-(77) demand-adjustment gradient, generic over backend.

    Replaces ``Problem._compute_analytical_demand_adjustment`` (used for
    ``demand_params``) and ``Problem._compute_demand_adjustment_gradient``
    (used for ``demand_results``). Returns the same 6-tuple either path
    previously produced (the analytical path previously returned 5 values
    with ``gradient_gamma_per_instrument=None`` — that silent gap is
    closed here per the v0.4 capability-parity decision).

    Parameters
    ----------
    backend
        A backend implementing ``SupportsDemandAdjustment``. Raises
        ``TypeError`` for backends that do not (e.g. bare
        ``UserSuppliedBackend`` without adjustment inputs).
    problem
        The ``Problem`` instance. Used as an accessor for
        ``problem.models``, ``problem.products``,
        ``problem.endogenous_cost_component``, ``problem._absorb_cost_ids``,
        ``problem._w_formulation``, ``problem.L`` (instrument-set count),
        and ``problem._compute_iv_correction``. Kept as a single
        parameter for API compactness; step 8 refactors ``solve/`` to
        pass narrower primitives.
    M
        Number of candidate conduct models.
    N
        Total number of observations.
    markups
        Per-model raw markups, length ``M``; each entry is shape
        ``(N, 1)``. These are the RAW (non-tax-adjusted) markups; the
        tax adjustment is applied downstream in ``Problem.solve`` when
        constructing ``markups_effective``.
    advalorem_tax_adj, cost_scaling
        Per-model tax factors (length ``M``). Used only in the gamma
        gradient, matching the inline
        ``_compute_demand_adjustment_gradient`` convention. (The
        analytical inline path did NOT apply tax adjustment to the
        markup gradient, and neither does this function — a known
        limitation under nonzero taxes; test fixtures currently have
        no taxes so byte-level equivalence is preserved.)
    marginal_cost_base
        Per-model marginal cost before the IV correction; required for
        computing the gamma gradient when
        ``problem.endogenous_cost_component is not None``.

    Returns
    -------
    tuple
        ``(gradient_markups, H_prime_wd, H, h_i, h,
           gradient_gamma_per_instrument)`` — same shape as the inline
        PyBLP path's return. ``gradient_gamma_per_instrument`` is a
        list of ``(M, n_theta)`` arrays of length
        ``problem.L`` when ``endogenous_cost_component`` is set, else
        ``None``.
    """
    # Local imports (avoid circular: solve/demand_adjustment is imported by
    # backends/pyblp for _residualize_on_xd; backends is the other direction).
    from ..backends.base import SupportsDemandAdjustment
    from ..markups import _compute_markups
    from .. import options

    if not isinstance(backend, SupportsDemandAdjustment):
        raise TypeError(
            f"compute_demand_adjustment requires a backend implementing "
            f"SupportsDemandAdjustment; got {type(backend).__name__}. "
            f"`UserSuppliedBackend` without adjustment inputs cannot produce "
            f"a demand-side first-stage correction. Either supply the "
            f"adjustment inputs when constructing the backend, or set "
            f"`demand_adjustment=False`."
        )
    # After the isinstance check, backend is a SupportsDemandAdjustment. The
    # Protocol only declares `demand_moments`, `xi_gradient`, and `perturbed`;
    # analytical-path calls (`compute_jacobian`, `jacobian_gradient`) live on
    # the concrete backend classes and are always present when the model path
    # exercises them. Cast to Any for the downstream backend calls.
    backend = cast(Any, backend)

    # -----------------------------------------------------------------
    # 1. Demand-side quantities from the backend.
    # -----------------------------------------------------------------
    xi, Z_D, W_D = backend.demand_moments()
    xi_col = np.asarray(xi).reshape(-1, 1)  # normalize to (N, 1) regardless of backend
    partial_xi_theta = backend.xi_gradient()
    n_theta = partial_xi_theta.shape[1]
    H = (1.0 / N) * Z_D.T @ partial_xi_theta
    H_prime_wd = H.T @ W_D
    h_i = Z_D * xi_col
    h = (1.0 / N) * Z_D.T @ xi_col

    # -----------------------------------------------------------------
    # 2. Markup gradient per model per theta.
    #
    # Standard downstream-only models (bertrand / cournot / monopoly /
    # mix_cournot_bertrand without custom spec, without upstream): use
    # implicit differentiation of the model's FOC. This is the general
    # form that works for ALL theta columns (alpha and sigma together)
    # via `backend.jacobian_gradient(t)` supplying dD/d(theta).
    #
    # Vertical / custom models: finite-diff via `backend.perturbed(k, delta)`.
    # -----------------------------------------------------------------
    gradient_markups = np.zeros((M, N, n_theta), dtype=options.dtype)
    market_ids = np.asarray(problem.products.market_ids).flatten()
    markets = np.unique(market_ids)
    shares_all = np.asarray(problem.products.shares).flatten()

    models_downstream = problem.models["models_downstream"]
    ownership_downstream = problem.models["ownership_downstream"]
    models_upstream = problem.models["models_upstream"]
    custom_spec = problem.models["custom_model_specification"]
    mix_flag_all = problem.models["mix_flag"]

    analytical_models: List[int] = []
    finite_diff_models: List[int] = []
    for m in range(M):
        if models_upstream[m] is not None or custom_spec[m] is not None:
            finite_diff_models.append(m)
        elif models_downstream[m] in ('perfect_competition',):
            # Markup is structurally zero; gradient is zero. Skip.
            pass
        else:
            analytical_models.append(m)

    # --- 2a. Analytical closed-form via implicit differentiation ---
    if analytical_models:
        for t in markets:
            idx = np.where(market_ids == t)[0]
            J_t = idx.shape[0]
            s_t = shares_all[idx]
            D_t = backend.compute_jacobian(market_id=t)   # (J_t, J_t)
            dD_dtheta = backend.jacobian_gradient(market_id=t)  # (J_t, J_t, n_theta)
            for m in analytical_models:
                model_type = models_downstream[m]
                ownership_m = ownership_downstream[m]
                if ownership_m is not None:
                    O_t = ownership_m[idx]
                    O_t = O_t[:, ~np.isnan(O_t).all(axis=0)]
                else:
                    O_t = np.ones((J_t, J_t))
                mu_t = markups[m][idx].flatten()
                for k in range(n_theta):
                    dD_k = dD_dtheta[:, :, k]
                    d_mu = _analytical_markup_derivative(
                        model_type, O_t, D_t, dD_k, s_t, mu_t,
                        mix_flag_all[m], idx, J_t,
                    )
                    gradient_markups[m][idx, k] = d_mu

    # --- 2b. Finite-diff for vertical / custom models ---
    if finite_diff_models:
        eps = options.finite_differences_epsilon
        for k in range(n_theta):
            markups_up, _, _ = _perturb_and_rebuild_markups(
                backend, problem, k, +eps / 2, _compute_markups
            )
            markups_dn, _, _ = _perturb_and_rebuild_markups(
                backend, problem, k, -eps / 2, _compute_markups
            )
            for m in finite_diff_models:
                gradient_markups[m][:, k] = (
                    markups_up[m].flatten() - markups_dn[m].flatten()
                ) / eps

    # -----------------------------------------------------------------
    # 3. Residualize gradient_markups on cost shifters (FWL, matches both
    #    inline paths; linear so order doesn't matter).
    # -----------------------------------------------------------------
    _residualize_grad_on_cost_shifters(gradient_markups, problem, M, n_theta)

    # -----------------------------------------------------------------
    # 3b. Apply tax / cost-scaling factor.
    #
    # ``Problem.solve`` defines ``markups_effective[m] = (advalorem_tax_adj[m] /
    # (1 + cost_scaling[m])) * markups[m]`` and the downstream GMM moment uses
    # ``mc = prices_effective - markups_effective``. So the object that enters
    # ``G_k = -(1/N) Z' @ gradient_markups[m]`` is ``d(markups_effective)/d(theta)``.
    # Since the tax factor does not depend on theta:
    #
    #     d(markups_effective)/d(theta) = tax_factor[m] * d(markups_raw)/d(theta)
    #
    # Pre-v0.4 the inline PyBLP path applied this factor (via
    # ``apply_tax_adjustment`` to markups_u / markups_dn before finite-diff);
    # the inline analytical path did not (a pre-existing bug, silent whenever
    # the fixture had zero taxes). Step 4d initially matched the analytical
    # behavior — which regressed PyBLP-path users with nontrivial taxes.
    # Step 4i fixes it by applying the factor uniformly here.
    for m in range(M):
        # advalorem_tax_adj[m] and cost_scaling[m] are per-observation arrays
        # (shape (N, 1)) — tax rates / scale factors can vary across products
        # within a model. Broadcasting (N, 1) * (N, n_theta) -> (N, n_theta)
        # scales each row of the gradient by the product-specific factor.
        factor_col = np.asarray(advalorem_tax_adj[m]).reshape(-1, 1) / (
            1.0 + np.asarray(cost_scaling[m]).reshape(-1, 1)
        )
        if not np.allclose(factor_col, 1.0):
            gradient_markups[m] = gradient_markups[m] * factor_col

    # -----------------------------------------------------------------
    # 4. Gamma gradient for endogenous cost — finite-diff per demand
    # parameter. Applies tax adjustment per PyBLP convention.
    # -----------------------------------------------------------------
    gradient_gamma_per_instrument: Optional[List[_NDArray]] = None
    if problem.endogenous_cost_component is not None and marginal_cost_base is not None:
        gradient_gamma_per_instrument = _compute_gamma_gradient(
            backend, problem, M, N, n_theta,
            advalorem_tax_adj, cost_scaling, _compute_markups,
        )

    return gradient_markups, H_prime_wd, H, h_i, h, gradient_gamma_per_instrument


# =====================================================================
# Internal helpers
# =====================================================================


def _analytical_markup_derivative(
        model_type: str, O_t: _NDArray, D_t: _NDArray, dD_k: _NDArray,
        s_t: _NDArray, mu_t: _NDArray, mix_flag_m: Any,
        idx: _NDArray, J_t: int,
) -> _NDArray:
    """Implicit-differentiation closed form for d(markup)/d(theta_k) in one market.

    Matches the inline algebra in ``Problem._compute_analytical_demand_adjustment``
    but expresses the alpha column via the same formula as the sigma columns
    (rather than the ``-mu_t / alpha`` shortcut). The two forms are algebraically
    identical; numerically they agree to a few ULP, well within test tolerance.
    """
    if model_type == 'bertrand':
        A = O_t * D_t.T
        dA = O_t * dD_k.T
        bertrand_result: _NDArray = -np.linalg.solve(A, dA @ mu_t)
        return bertrand_result
    if model_type == 'cournot':
        D_inv = np.linalg.inv(D_t)
        dD_inv = -D_inv @ dD_k @ D_inv
        cournot_result: _NDArray = -(O_t * dD_inv) @ s_t
        return cournot_result
    if model_type == 'monopoly':
        dA = dD_k.T
        monopoly_result: _NDArray = -np.linalg.solve(D_t.T, dA @ mu_t)
        return monopoly_result
    if model_type == 'mix_cournot_bertrand':
        b_t = mix_flag_m[idx].flatten().astype(bool)
        c_t = ~b_t
        if not (c_t.any() and b_t.any()):
            return np.zeros(J_t)
        D_BB = D_t[np.ix_(b_t, b_t)]
        D_BC = D_t[np.ix_(b_t, c_t)]
        D_CB = D_t[np.ix_(c_t, b_t)]
        D_CC = D_t[np.ix_(c_t, c_t)]
        D_CC_inv = np.linalg.inv(D_CC)
        O_BB = O_t[np.ix_(b_t, b_t)]
        O_CC = O_t[np.ix_(c_t, c_t)]
        dD_BB = dD_k[np.ix_(b_t, b_t)]
        dD_BC = dD_k[np.ix_(b_t, c_t)]
        dD_CB = dD_k[np.ix_(c_t, b_t)]
        dD_CC = dD_k[np.ix_(c_t, c_t)]
        # Cournot block
        dD_CC_inv = -D_CC_inv @ dD_CC @ D_CC_inv
        d_mu_C = -(O_CC * dD_CC_inv) @ s_t[c_t]
        # Bertrand block via Schur complement
        Schur = D_BC @ D_CC_inv @ D_CB + D_BB
        dSchur = (dD_BC @ D_CC_inv @ D_CB + D_BC @ dD_CC_inv @ D_CB
                  + D_BC @ D_CC_inv @ dD_CB + dD_BB)
        A_B = O_BB * Schur
        dA_B = O_BB * dSchur
        d_mu_B = -np.linalg.solve(A_B, dA_B @ mu_t[b_t])
        d_mu = np.zeros(J_t)
        d_mu[b_t] = d_mu_B.flatten()
        d_mu[c_t] = d_mu_C.flatten()
        return d_mu
    # Perfect competition or unknown model type: zero gradient.
    return np.zeros(J_t)


def _perturb_and_rebuild_markups(
        backend: Any, problem: Any, theta_index: int, delta: float,
        compute_markups_fn: Callable[..., Any],
) -> Tuple[List[_NDArray], List[_NDArray], List[_NDArray]]:
    """Enter ``backend.perturbed(theta_index, delta)`` and call ``_compute_markups``.

    Returns the full ``(markups, markups_down, markups_up)`` tuple.
    """
    with backend.perturbed(theta_index, delta) as perturbed_backend:
        result: Tuple[List[_NDArray], List[_NDArray], List[_NDArray]] = compute_markups_fn(
            problem.products, None,  # pyblp_results unused when demand_backend supplied
            problem.models["models_downstream"],
            problem.models["ownership_downstream"],
            problem.models["models_upstream"],
            problem.models["ownership_upstream"],
            problem.models["vertical_integration"],
            problem.models["custom_model_specification"],
            problem.models["user_supplied_markups"],
            problem.models["mix_flag"],
            demand_backend=perturbed_backend,
        )
        return result


def _residualize_grad_on_cost_shifters(
        gradient_markups: _NDArray, problem: Any, M: int, n_theta: int,
) -> None:
    """In-place residualization of ``gradient_markups[m][:, k]`` on cost shifters.

    Matches the last block of ``Problem._compute_analytical_demand_adjustment``
    (residualization done once at the end on the assembled gradient, same result
    as residualizing per perturbation in the inline PyBLP path).
    """
    if problem.endogenous_cost_component is not None:
        endog_col_idx = next(
            i for i, f in enumerate(problem._w_formulation)
            if str(f) == problem.endogenous_cost_component
        )
        exog_col_indices = [
            i for i in range(problem.products.w.shape[1]) if i != endog_col_idx
        ]
        w_for_ols = problem.products.w[:, exog_col_indices]
    else:
        w_for_ols = problem.products.w

    if problem._absorb_cost_ids is not None:
        w_absorbed, _ = problem._absorb_cost_ids(w_for_ols)
    else:
        w_absorbed = w_for_ols
    Q_w = (
        np.linalg.qr(w_absorbed, mode='reduced')[0]
        if w_absorbed.any() else None
    )

    for m in range(M):
        for k in range(n_theta):
            col = gradient_markups[m][:, k]
            if problem._absorb_cost_ids is not None:
                col, _ = problem._absorb_cost_ids(col.reshape(-1, 1))
                col = col.flatten()
            if Q_w is not None:
                col = col - Q_w @ (Q_w.T @ col)
            gradient_markups[m][:, k] = col


def _compute_gamma_gradient(
        backend: Any, problem: Any, M: int, N: int, n_theta: int,
        advalorem_tax_adj: List[Any], cost_scaling: List[Any],
        compute_markups_fn: Callable[..., Any],
) -> List[_NDArray]:
    """Finite-diff of per-instrument gamma w.r.t. each demand parameter.

    Matches the inline PyBLP path's `_record_gamma_gradient` helper:
    for each theta_k, re-evaluate markups at +eps/2 and -eps/2 with the
    backend perturbed, apply tax adjustment, compute implied mc,
    run `_compute_iv_correction` per instrument set, and finite-diff
    the last element of cost_param (the gamma coefficient).
    """
    from .. import options
    eps = options.finite_differences_epsilon
    L_inst = problem.L
    prices = problem.products.prices
    grad_gamma: List[_NDArray] = [
        np.zeros((M, n_theta), dtype=options.dtype) for _ in range(L_inst)
    ]

    for k in range(n_theta):
        markups_up, _, _ = _perturb_and_rebuild_markups(
            backend, problem, k, +eps / 2, compute_markups_fn
        )
        markups_dn, _, _ = _perturb_and_rebuild_markups(
            backend, problem, k, -eps / 2, compute_markups_fn
        )
        # Tax-adjust perturbed markups before computing mc (PyBLP convention).
        markups_up_adj = [
            (advalorem_tax_adj[m] * markups_up[m]) / (1 + cost_scaling[m])
            for m in range(M)
        ]
        markups_dn_adj = [
            (advalorem_tax_adj[m] * markups_dn[m]) / (1 + cost_scaling[m])
            for m in range(M)
        ]
        mc_up = [prices - markups_up_adj[m] for m in range(M)]
        mc_dn = [prices - markups_dn_adj[m] for m in range(M)]
        for inst in range(L_inst):
            cp_up, _, _ = problem._compute_iv_correction(inst, M, N, mc_up)
            cp_dn, _, _ = problem._compute_iv_correction(inst, M, N, mc_dn)
            for m in range(M):
                grad_gamma[inst][m, k] = (
                    float(cp_up[m][-1]) - float(cp_dn[m][-1])
                ) / eps

    return grad_gamma
