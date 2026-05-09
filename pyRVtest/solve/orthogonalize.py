"""Orthogonalization stage: residualize markups and marginal cost on cost shifters.

Hosts the stage that absorbs cost-side fixed
effects and residualizes markups / marginal cost on the (exogenous)
cost-shifter matrix ``w``. The linear-regression coefficient on the
exogenous shifters is also returned as ``tau_list``. Exported:

* :func:`qr_residualize` — the low-level QR-based projection helper
  (formerly ``pyRVtest.problem._qr_residualize``).
* :func:`residualize` — the full orthogonalize stage (formerly
  ``Problem._prepare_orthogonal_variables``).

The stage is a pure function: it reads ``problem.products.w``,
``problem._w_formulation``, ``problem._absorb_cost_ids`` and
``problem.endogenous_cost_component``, but does not mutate the
``Problem``. See ``pyRVtest/problem.py`` for the orchestrator that
composes this stage with the others.
"""

from __future__ import annotations

import logging
from typing import Any, List, Tuple

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypeAlias

from .. import options


__all__ = ['qr_residualize', 'residualize']


_NDArray: TypeAlias = NDArray[Any]


# per-module logger. Users can silence this
# stage specifically with
# ``logging.getLogger('pyRVtest.solve.orthogonalize').setLevel(logging.WARNING)``.
logger = logging.getLogger(__name__)


def qr_residualize(Y: _NDArray, X: _NDArray) -> _NDArray:
    """Project out ``X`` from ``Y`` via QR decomposition.

    Equivalent to OLS residuals without statsmodels overhead. Works for
    1D or 2D ``Y``. If ``X`` has zero columns, ``Y`` is returned
    unchanged.

    Parameters
    ----------
    Y
        Left-hand side, shape ``(N,)`` or ``(N, k)``.
    X
        Right-hand side regressors, shape ``(N, m)``.

    Returns
    -------
    ndarray
        Residuals ``Y - X (X' X)^{-1} X' Y``.
    """
    if X.shape[1] == 0:
        return Y
    Q, _ = np.linalg.qr(X, mode='reduced')
    result: _NDArray = Y - Q @ (Q.T @ Y)
    return result


def residualize(
        problem: Any, M: int, N: int,
        markups_effective: List[_NDArray], marginal_cost: _NDArray,
) -> Tuple[_NDArray, _NDArray, _NDArray]:
    """Absorb fixed effects and residualize markups / mc on cost shifters.

    Moved from ``Problem._prepare_orthogonal_variables``.
    Math is unchanged.

    When ``problem.endogenous_cost_component`` is set, the endogenous
    column is excluded from the OLS projection (its coefficient was
    estimated via IV in :func:`solve.endogenous_cost.iv_correct` and the
    correction has already been applied to ``marginal_cost``). ``tau_list``
    therefore corresponds to the exogenous cost-shifter coefficients.

    Parameters
    ----------
    problem
        The :class:`pyRVtest.Problem` instance. Accessed for
        ``products.w``, ``_w_formulation``, ``_absorb_cost_ids``, and
        ``endogenous_cost_component``.
    M
        Number of candidate conduct models.
    N
        Total number of observations.
    markups_effective
        Per-model tax-adjusted markups, length ``M``; each entry is shape
        ``(N, 1)``.
    marginal_cost
        Per-model marginal cost, shape ``(M, N)`` or a list of ``(N, 1)``.

    Returns
    -------
    markups_orthogonal : ndarray
        Shape ``(M, N)`` — markups with cost shifters projected out.
    omega : ndarray
        Shape ``(M, N)`` — marginal-cost residuals.
    tau_list : ndarray
        Shape ``(M, K_w_exog)`` — OLS coefficients on exogenous cost
        shifters.
    """
    markups_orthogonal = np.zeros((M, N), dtype=options.dtype)
    marginal_cost_orthogonal = np.zeros((M, N), dtype=options.dtype)

    # When endogenous cost component(s) are present, their coefficients were estimated via IV
    # and the correction has already been applied to marginal_cost. The OLS projection therefore
    # uses only the exogenous columns of w so that tau_list corresponds to the exogenous
    # cost-shifter coefficients. Generalized to K_endog >= 1.
    if problem.endogenous_cost_component is not None:
        name_to_w_idx = {str(f): i for i, f in enumerate(problem._w_formulation)}
        endog_col_indices = {
            name_to_w_idx[name] for name in problem._endogenous_cost_columns
        }
        exog_col_indices = [
            i for i in range(problem.products.w.shape[1])
            if i not in endog_col_indices
        ]
        w_for_ols = problem.products.w[:, exog_col_indices]
    else:
        w_for_ols = problem.products.w

    tau_list = np.zeros((M, w_for_ols.shape[1]), dtype=options.dtype)
    omega = np.zeros((M, N), dtype=options.dtype)

    if problem._absorb_cost_ids is not None:
        logger.info("Absorbing cost-side fixed effects ...")
        w_for_ols, _ = problem._absorb_cost_ids(w_for_ols)
        for m in range(M):
            value, _ = problem._absorb_cost_ids(markups_effective[m])
            markups_orthogonal[m] = np.squeeze(value)
            value, _ = problem._absorb_cost_ids(marginal_cost[m])
            marginal_cost_orthogonal[m] = np.squeeze(value)
    else:
        for m in range(M):
            markups_orthogonal[m] = np.squeeze(markups_effective[m])
            marginal_cost_orthogonal[m] = np.squeeze(marginal_cost[m])

    if w_for_ols.any():
        Q_w, R_w = np.linalg.qr(w_for_ols, mode='reduced')
        for m in range(M):
            markups_orthogonal[m] = markups_orthogonal[m] - Q_w @ (Q_w.T @ markups_orthogonal[m])
            mc_vec = marginal_cost_orthogonal[m]
            tau_list[m] = np.linalg.solve(R_w, Q_w.T @ mc_vec)
            omega[m] = mc_vec - Q_w @ (Q_w.T @ mc_vec)
    else:
        omega = marginal_cost_orthogonal

    return markups_orthogonal, omega, tau_list
