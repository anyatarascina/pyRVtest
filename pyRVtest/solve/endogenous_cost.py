"""Endogenous cost component (scale economies) IV correction.

Hosts :func:`iv_correct`, the per-instrument-
set stage that runs a 2SLS first stage for the coefficient ``gamma_m``
on an endogenous cost component (e.g., log market shares, cost scaling
column). The correction turns ``mc[m] = price - markup[m]`` into
``mc[m] - gamma_m * endogenous_variable``, and the fitted values
``endog_hat`` flow into the Appendix B correction in
:func:`solve.test_engine.compute`.

Moved verbatim from ``Problem._compute_iv_correction``. No math change.
See §4.1 of ``.claude/plans/v0.4-refactor.md`` for the plan.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypeAlias


__all__ = ['iv_correct']


_NDArray: TypeAlias = NDArray[Any]


logger = logging.getLogger(__name__)


def iv_correct(
        problem: Any, instrument: int, M: int, N: int,
        marginal_cost: List[_NDArray],
) -> Tuple[List[Optional[_NDArray]], List[Optional[_NDArray]], _NDArray]:
    """Run per-model 2SLS for the endogenous-cost coefficient(s) ``gamma_m``.

    For each model ``m`` the dependent variable is the implied marginal
    cost (price minus markup). The endogenous cost component(s)
    (``problem.endogenous_cost_component``) are instrumented using the
    specified instrument set and the exogenous cost-shifters. With
    ``K_endog`` endogenous columns the second-stage parameter vector
    ``cost_param[m]`` ends with a length-``K_endog`` gamma block; the
    correction ``mc_correction[m] = - endog_cols_raw @ gamma_m`` is the
    multi-column generalization of the single-column ``-gamma * z``
    formula.

    Parameters
    ----------
    problem
        The :class:`pyRVtest.Problem` instance.
    instrument
        Index of the instrument set ``Z_l`` to use for the first stage.
    M
        Number of candidate conduct models.
    N
        Total number of observations.
    marginal_cost
        Per-model ``(N, 1)`` marginal-cost vector (price minus tax-
        adjusted markup, before the IV correction).

    Returns
    -------
    cost_param : list of ndarray
        Per-model 2SLS parameter vectors. Each vector concatenates the
        ``(K_w − K_endog)`` exogenous cost-shifter coefficients followed
        by ``K_endog`` gamma coefficients.
    mc_correction : list of ndarray
        Per-model ``(N, 1)`` correction arrays to be added to marginal
        cost.
    endog_hat : ndarray
        First-stage fitted values for the endogenous column(s), shape
        ``(N, K_endog)``. Shared across all models in this instrument
        set. ``K_endog == 1`` returns the same shape as before this
        method was generalized to the multi-column case (paper DMQSS
        A.4, examples for q+q² and scale+scope).
    """
    # Identify the endogenous column(s) in w. K_endog >= 1 supported:
    # str on problem.endogenous_cost_component => 1 column;
    # tuple/list => multiple columns (DMQSS A.4: scale+scope, q+q^2, etc.).
    endog_names = problem._endogenous_cost_columns  # tuple of str
    K_endog = len(endog_names)
    name_to_idx = {str(f): i for i, f in enumerate(problem._w_formulation)}
    endog_col_indices = [name_to_idx[name] for name in endog_names]
    endog_set = set(endog_col_indices)
    exog_col_indices = [
        i for i in range(problem.products.w.shape[1]) if i not in endog_set
    ]
    endog_cols_raw = problem.products.w[:, endog_col_indices]  # (N, K_endog) — raw, used for mc_correction
    exog_w = problem.products.w[:, exog_col_indices]            # (N, K_w - K_endog)

    # Use only the instrument set for this test (keeps each instrument set's correction independent)
    Z_inst = problem.products["Z{0}".format(instrument)]  # (N, K_l)

    # Absorb cost-side fixed effects before running 2SLS, so that gamma_m is estimated
    # within-group rather than in levels (consistent with _prepare_orthogonal_variables)
    endog_cols = endog_cols_raw
    if problem._absorb_cost_ids is not None:
        exog_w, _ = problem._absorb_cost_ids(exog_w)
        endog_cols, _ = problem._absorb_cost_ids(endog_cols)
        Z_inst, _ = problem._absorb_cost_ids(Z_inst)

    # First stage: project each endogenous column on [exog_w, Z_inst].
    # The same projection matrix Q_fs Q_fs' applies to every column, so we
    # can compute K_endog first-stage predictions in one shot via matmul.
    first_stage_X = np.hstack([exog_w, Z_inst])
    Q_fs, _ = np.linalg.qr(first_stage_X, mode='reduced')
    endog_hat = Q_fs @ (Q_fs.T @ endog_cols)  # (N, K_endog)

    # Second stage design matrix: replace endogenous columns with first-stage fitted values
    X_2sls = np.hstack([exog_w, endog_hat])  # (N, K_w)

    Q_2sls, R_2sls = np.linalg.qr(X_2sls, mode='reduced')
    cost_param: List[Optional[_NDArray]] = [None] * M
    mc_correction: List[Optional[_NDArray]] = [None] * M
    for m in range(M):
        y_m = marginal_cost[m]  # (N, 1)
        params = np.linalg.solve(R_2sls, Q_2sls.T @ y_m)
        # The last K_endog elements are the gamma vector; preceding entries
        # are the exogenous-cost-shifter coefficients tau.
        gamma_m = params[-K_endog:]                                  # (K_endog, 1)
        cost_param[m] = params                                       # [tau_exog..., gamma_1, ..., gamma_K]
        mc_correction[m] = -endog_cols_raw @ gamma_m                 # (N, 1) — uses raw (un-absorbed) endog

    return cost_param, mc_correction, endog_hat
