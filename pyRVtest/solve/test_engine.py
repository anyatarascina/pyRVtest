"""RV test statistic, F-statistic, and MCS p-value computation.

v0.4 step 8d extraction. Hosts the test-engine computations moved from
``Problem._compute_instrument_results``, ``Problem._compute_mcs``, and
the block-gram bookkeeping helpers ``Problem._compute_block_gram`` /
``Problem._extract_block``. No math change relative to the pre-step-8
inline code — pure code move.

The :class:`pyRVtest.Problem` orchestrator calls
:func:`compute_instrument_results` once per instrument set and folds
the per-instrument results into the final :class:`ProblemResults`.
"""

from __future__ import annotations

import itertools
import logging
import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypeAlias
from pyblp.utilities.basics import Array
from scipy.linalg import inv
from scipy.stats import norm

from .. import options
from .orthogonalize import qr_residualize


__all__ = [
    'compute_block_gram',
    'compute_instrument_results',
    'compute_mcs',
    'extract_block',
]


_NDArray: TypeAlias = NDArray[Any]


logger = logging.getLogger(__name__)


# F-stat reliability diagnostic thresholds.
#
# Redesign (2026-05-01): under the worst-case-CV verdict architecture, lambda
# is no longer the gating diagnostic. It is retained as a numerical-fragility
# footnote on plug-in-dependent cells (see ``compute_instrument_results``).
# The principled robustness signal is whether F clears the *worst-case* CV
# across rho^2 in [0, 0.99], which is a free lookup on the existing critical-
# value table.
RELIABILITY_LAMBDA_THRESHOLD = 0.05  # numerical-fragility annotation on plug-in cells
RELIABILITY_CI_LEVEL = 1.96  # 95% CI half-width; retained for the SE column

# Worst-case size and best-case power levels corresponding to the three CVs
# in the published table. Used for human-readable claim labels.
_SIZE_LEVELS_PCT = (12.5, 10.0, 7.5)  # claim "worst-case size <= X%" if F > r_X (column order: r_125, r_10, r_075)
_POWER_LEVELS_PCT = (50.0, 75.0, 95.0)  # claim "best-case power >= Y%" if F > r_Y (column order: r_50, r_75, r_95)


def _lookup_cv_row(critical_values: Array, K_lookup: int, rho_lookup: float) -> int:
    """Return the row index in a published-CV table for given (K, rho).

    Helper for the worst-case CV scan. ``rho_lookup`` is matched against the
    ``rho`` column to two-decimal precision (the table grid)."""
    ind = np.where(
        (critical_values['K'] == K_lookup)
        & (np.round(critical_values['rho'], 2) == round(rho_lookup, 2))
    )[0]
    if ind.size == 0:
        raise KeyError(
            f"No critical-value table row for K={K_lookup}, rho={rho_lookup}."
        )
    return int(ind[0])


def _recompute_F_high_precision(
        phi_blocks: Tuple[_NDArray, _NDArray, _NDArray],
        W_inverse: _NDArray,
        weight_matrix: _NDArray,
        g_i: _NDArray,
        g_m: _NDArray,
        K: int,
        N: int,
        prec: int = 50,
) -> Tuple[float, float]:
    """Recompute (F̂, ρ̂²) at high precision via mpmath.

    Used in the 'conditional' / 'always' precision-check paths to obtain
    a high-precision reference value for F̂ and ρ̂² so we can flag cells
    where the double-precision computation has lost so much precision to
    cancellation that the F̂-vs-CV decision could flip.

    Recomputes σ̂_0, σ̂_1, σ̂_2 at high precision from the phi_gram blocks
    + W_inverse — not just the F formula given σ̂'s — so cancellation
    anywhere in the chain is captured.

    Parameters
    ----------
    phi_blocks
        Tuple ``(V_ii, V_mm, V_im)``; each KxK numpy array.
    W_inverse
        KxK demand-side inverse weighting matrix (the same one trace-
        contracted to produce double-precision σ̂'s).
    weight_matrix
        GMM weighting matrix (the same one used in F's quadratic forms).
    g_i, g_m
        K-vectors (instrument-projected moments for each model).
    K
        Number of instruments / effective dz used by σ̂ scaling.
    N
        Sample size factor in F's overall scaling.
    prec
        mpmath working decimal digits. Default 50, ample for any
        cancellation we might encounter.

    Returns
    -------
    tuple of (F_high_precision, rho_squared_high_precision)
        Both as Python floats. Caller compares to double-precision values
        and decides whether the precision loss matters.
    """
    import mpmath  # type: ignore[import-untyped]

    with mpmath.workdps(prec):
        Vi = mpmath.matrix(phi_blocks[0].tolist())
        Vm = mpmath.matrix(phi_blocks[1].tolist())
        Vim = mpmath.matrix(phi_blocks[2].tolist())
        Winv = mpmath.matrix(W_inverse.tolist())
        Wgmm = mpmath.matrix(weight_matrix.tolist())
        gi = mpmath.matrix([float(x) for x in np.asarray(g_i).flatten()])
        gm = mpmath.matrix([float(x) for x in np.asarray(g_m).flatten()])

        def trace_mp(M: Any) -> Any:
            return sum(M[k, k] for k in range(M.rows))

        sig0 = trace_mp(Vi * Winv) / K
        sig1 = trace_mp(Vm * Winv) / K
        sig2 = trace_mp(Vim * Winv) / K

        sigma_sum = sig0 + sig1
        # Factored denominator: cancellation in the (sigma_sum - 2 sig2)
        # term, computed at high precision so the cancellation is
        # absorbed by the available digits.
        D_rho = (sigma_sum - 2 * sig2) * (sigma_sum + 2 * sig2)

        rho_sq_num = (sig0 - sig1) ** 2
        rho_squared = rho_sq_num / D_rho

        def quad(g: Any, W: Any, h: Any) -> Any:
            return (g.T * W * h)[0, 0]

        F_num = (
            sig1 * quad(gi, Wgmm, gi)
            + sig0 * quad(gm, Wgmm, gm)
            - 2 * sig2 * quad(gi, Wgmm, gm)
        )
        F = 2 * N * F_num / (K * D_rho)
        return float(F), float(rho_squared)


def _worst_case_cv_size(
        critical_values_size: Array, K_effective: int,
) -> _NDArray:
    """Return the maximum size CV across rho ∈ [0, 0.99] at this K.

    Size CVs are monotone increasing in rho^2 (paper Table 1, panel A);
    the worst case sits at the high-rho corner of the table. The
    returned array has three entries matching the column order
    ``[r_125, r_10, r_075]``.

    A user whose F exceeds this worst-case CV makes a strength claim
    that is robust to ρ̂² estimation error: even if ρ̂² mismeasures the
    population ρ², the claim still holds at the population's CV.
    """
    K_lookup = min(int(K_effective), 30)
    # Walk the table at this K and pick the maximum at each column.
    mask = critical_values_size['K'] == K_lookup
    if not mask.any():
        return np.array([np.nan, np.nan, np.nan])
    block = critical_values_size[mask]
    return np.array([
        float(block['r_125'].max()),
        float(block['r_10'].max()),
        float(block['r_075'].max()),
    ])


def _worst_case_cv_power(
        critical_values_power: Array, K_effective: int,
) -> _NDArray:
    """Return the maximum power CV across rho ∈ [0, 0.99] at this K.

    Power CVs are monotone *decreasing* in rho^2 (paper Table 1, panel B),
    so the hardest CV to clear sits at the low-rho corner of the table.
    Returned in column order ``[r_50, r_75, r_95]``.
    """
    K_lookup = min(int(K_effective), 30)
    mask = critical_values_power['K'] == K_lookup
    if not mask.any():
        return np.array([np.nan, np.nan, np.nan])
    block = critical_values_power[mask]
    return np.array([
        float(block['r_50'].max()),
        float(block['r_75'].max()),
        float(block['r_95'].max()),
    ])


def compute_instrument_results(
        problem: Any, instrument: int, M: int, N: int, omega: Array,
        demand_adjustment: bool, gradient_markups: Optional[Array],
        H_prime_wd: Optional[Array], H: Optional[Array],
        h_i: Optional[Array], h: Optional[Array],
        clustering_adjustment: bool,
        critical_values_size: Array, critical_values_power: Array,
        endog_hat: Optional[Array] = None,
        gradient_gamma: Optional[Array] = None,
        reliability_check: str = 'conditional',
        reliability_precision_dps: int = 50,
) -> Dict[str, Any]:
    """Compute all test statistics for a single instrument set.

    Moved from ``Problem._compute_instrument_results`` in v0.4 step 8d.
    Math is unchanged.
    """
    instruments = problem.products["Z{0}".format(instrument)]
    K = np.shape(instruments)[1]
    K_effective = K - 1 if endog_hat is not None else K

    # v0.4.0rc1: the tabulated F-statistic critical values (size and power) are
    # defined for K=1..30. For larger instrument sets we fall back to the K=30
    # row, which can produce conservative / stale diagnostics. Emit a
    # ``UserWarning`` so the user knows the reported stars are approximate.
    if K_effective > 30:
        warnings.warn(
            f"K_effective={K_effective} (instrument set {instrument}) exceeds "
            f"the range of the tabulated F-statistic critical values (K=1..30). "
            f"Falling back to the K=30 row; reported size/power stars may be "
            f"conservative. "
            f"Fix: use a smaller instrument set, or treat the F stars as "
            f"indicative only in this regime.",
            UserWarning, stacklevel=2,
        )

    # Use only exogenous cost-shifter columns when an endogenous component has been IV-corrected
    if problem.endogenous_cost_component is not None:
        endog_col_idx = next(
            i for i, f in enumerate(problem._w_formulation)
            if str(f) == problem.endogenous_cost_component
        )
        exog_col_indices = [i for i in range(problem.products.w.shape[1]) if i != endog_col_idx]
        w_for_ols = problem.products.w[:, exog_col_indices]
    else:
        w_for_ols = problem.products.w

    if problem._absorb_cost_ids is not None:
        Z_orthogonal, _ = problem._absorb_cost_ids(instruments)
        w_absorbed, _ = problem._absorb_cost_ids(w_for_ols)
        endog_hat_absorbed = endog_hat
        if endog_hat is not None:
            endog_hat_absorbed, _ = problem._absorb_cost_ids(endog_hat)
    else:
        Z_orthogonal = instruments
        w_absorbed = w_for_ols
        endog_hat_absorbed = endog_hat

    # Residualize instruments on exogenous cost-shifters and (when applicable) first-stage
    # fitted values of the endogenous cost component jointly, so that Z_orthogonal is orthogonal
    # to both simultaneously. The rank reduction from adding endog_hat is handled by pinv.
    # All inputs must be in the same basis (absorbed or raw) for valid FWL.
    if endog_hat_absorbed is not None:
        controls = np.hstack([w_absorbed, endog_hat_absorbed]) if w_absorbed.shape[1] > 0 else endog_hat_absorbed
        Z_orthogonal = np.reshape(qr_residualize(Z_orthogonal, controls), [N, K])
    elif w_absorbed.any():
        Z_orthogonal = np.reshape(qr_residualize(Z_orthogonal, w_absorbed), [N, K])

    W_inverse = np.reshape(1 / N * (Z_orthogonal.T @ Z_orthogonal), [K, K])
    if options.pseudo_inverses:
        weight_matrix = np.linalg.pinv(W_inverse)
    else:
        try:
            weight_matrix = np.linalg.inv(W_inverse)
        except np.linalg.LinAlgError:
            weight_matrix = np.linalg.pinv(W_inverse)

    # GMM moments and fit for each model
    g = np.zeros((M, K), dtype=options.dtype)
    Q = np.zeros(M, dtype=options.dtype)
    for m in range(M):
        g[m] = 1 / N * (Z_orthogonal.T @ omega[m])
        Q[m] = g[m].T @ weight_matrix @ g[m]

    # RV numerator
    test_statistic_numerator = np.zeros((M, M))
    for m in range(M):
        for i in range(m):
            test_statistic_numerator[i, m] = math.sqrt(N) * (Q[i] - Q[m])

    # psi for each model and RV denominator
    # Use eigendecomposition with non-negative clipping to avoid complex values that arise when
    # floating-point errors push a zero eigenvalue (from a rank-deficient Z_orthogonal) slightly
    # negative, which would cause fractional_matrix_power to return complex results.
    _eigvals, _eigvecs = np.linalg.eigh((weight_matrix + weight_matrix.T) / 2)
    _eigvals = np.maximum(_eigvals, 0)
    W_12 = (_eigvecs * (_eigvals ** 0.50)) @ _eigvecs.T
    W_34 = (_eigvecs * (_eigvals ** 0.75)) @ _eigvecs.T
    psi = np.zeros((M, N, K), dtype=options.dtype)
    if demand_adjustment:
        adjustment_value = np.zeros((M, K, H_prime_wd.shape[1]), dtype=options.dtype)

    # Precompute first-stage correction ingredients when endogenous cost component is present.
    # Per Appendix B of Duarte-Magnolfi-Quint-Solvsten-Sullivan (2025), the influence function
    # psi includes a correction for estimation of the linear predictor q_tilde.
    endog_correction_data = None
    if endog_hat is not None:
        endog_col_idx_local = next(
            i for i, f in enumerate(problem._w_formulation)
            if str(f) == problem.endogenous_cost_component
        )
        endog_col = problem.products.w[:, [endog_col_idx_local]]  # (N, 1) raw endogenous variable
        q_e = endog_col - endog_hat                            # (N, 1) first-stage residual

        # z^r = z residualized on w only (not on endog_hat)
        z_r = qr_residualize(instruments, w_for_ols) if w_for_ols.shape[1] > 0 else instruments.copy()
        if problem._absorb_cost_ids is not None:
            z_r, _ = problem._absorb_cost_ids(z_r)
        z_r = z_r.reshape(N, K)

        # Z_prec = (1/n sum z^r z^{r'})^{-1}
        Z_cov = (1 / N) * z_r.T @ z_r                         # (K, K)
        Z_prec = np.linalg.pinv(Z_cov)                        # (K, K)

        # lambda_q: coefficient on endog_hat in projection z^e = lambda_q * q_tilde + Lambda_w * w
        # This is the coefficient from projecting z on [endog_hat, w], taking the endog_hat part
        proj_X = np.hstack([endog_hat, w_for_ols]) if w_for_ols.shape[1] > 0 else endog_hat
        Q_proj, R_proj = np.linalg.qr(proj_X, mode='reduced')
        lambda_coefs = np.linalg.solve(R_proj, Q_proj.T @ instruments)  # (d_proj, K)
        lambda_q = lambda_coefs[0, :]  # (K,) — coefficient on endog_hat (first column)

        # Precompute the (K, K) matrix M_correction = W^{3/4} W^+ Z_prec, used in correction
        W_plus = weight_matrix  # this is already the pseudo-inverse of W_inverse
        M_corr = W_34 @ W_plus @ Z_prec  # (K, K)

        endog_correction_data = (z_r, q_e, lambda_q, M_corr, Z_prec, W_plus)

    for m in range(M):
        psi_bar = W_12 @ g[m] - .5 * W_34 @ W_inverse @ W_34 @ g[m]
        W_34_Zg = (Z_orthogonal @ W_34 @ g[m])[:, np.newaxis]
        mc_col = omega[m][:, np.newaxis]
        psi_i = (mc_col * Z_orthogonal) @ W_12 - 0.5 * W_34_Zg * (Z_orthogonal @ W_34.T)
        psi[m] = psi_i - np.transpose(psi_bar)

        # First-stage correction: Appendix B of Duarte-Magnolfi-Quint-Solvsten-Sullivan (2025).
        # Per observation i, the correction to psi[m][i,:] is:
        #   (1/2) W^{3/4} (W^+ Z_prec u_i lambda'_q + lambda_q u'_i Z_prec W^+) W^{3/4} g_m
        # where u_i = z^r_i * q^e_i, M_corr = W^{3/4} W^+ Z_prec.
        # Term 1 contracts to: M_corr @ u_i * (lambda_q . W^{3/4} g_m)
        # Term 2 contracts to: W^{3/4} lambda_q * (u_i . Z_prec W^+ W^{3/4} g_m)
        if endog_correction_data is not None:
            z_r, q_e, lambda_q, M_corr, Z_prec, W_plus = endog_correction_data
            W34_gm = W_34 @ g[m]                                       # (K,)
            v = Z_prec @ W_plus @ W34_gm                               # (K,) right-side contraction
            u = q_e * z_r                                              # (N, K)
            term1 = (u @ M_corr.T) * (lambda_q @ W34_gm)              # (N, K)
            term2 = (u @ v)[:, np.newaxis] * (W_34 @ lambda_q)[np.newaxis, :]  # (N, K)
            psi[m] = psi[m] + 0.5 * (term1 + term2)

        if demand_adjustment:
            G_k = -1 / N * np.transpose(Z_orthogonal) @ gradient_markups[m]
            # When endogenous_cost_component is set, account for d(gamma_m)/d(theta) in G_k.
            # The full gradient of omega w.r.t. theta includes a term from gamma changing,
            # which contributes -(1/N) Z' @ (d_gamma/d_theta * endog_resid) to G_k.
            if gradient_gamma is not None and endog_correction_data is not None:
                endog_col_resid = endog_correction_data[1] + endog_correction_data[0]  # q^e + z^r... no
                # Actually: the endogenous column residualized on w is needed. Reconstruct it.
                endog_col_idx_local2 = next(
                    i for i, f in enumerate(problem._w_formulation)
                    if str(f) == problem.endogenous_cost_component
                )
                endog_col_raw = problem.products.w[:, [endog_col_idx_local2]]
                if problem._absorb_cost_ids is not None:
                    endog_col_for_grad, _ = problem._absorb_cost_ids(endog_col_raw)
                else:
                    endog_col_for_grad = endog_col_raw
                if w_absorbed.shape[1] > 0:
                    endog_col_for_grad = qr_residualize(endog_col_for_grad, w_absorbed)
                # gradient_gamma[m] is (n_theta,) — d gamma_m / d theta_k for each k
                # G_k[:, k] -= (1/N) * Z' @ (d_gamma_m/d_theta_k * endog_resid)
                G_k = G_k - 1 / N * np.transpose(Z_orthogonal) @ (endog_col_for_grad @ gradient_gamma[[m], :])
            adjustment_value[m] = W_12 @ G_k @ inv(H_prime_wd @ H) @ H_prime_wd
            psi[m] = psi[m] - (h_i - np.transpose(h)) @ np.transpose(adjustment_value[m])

    test_statistic_denominator = np.zeros((M, M))
    covariance_mc = np.zeros((M, M))
    psi_gram = compute_block_gram(problem, N, clustering_adjustment, psi)
    for m in range(M):
        for i in range(m):
            vc_ii = extract_block(psi_gram, i, i, K)
            vc_mm = extract_block(psi_gram, m, m, K)
            vc_im = extract_block(psi_gram, i, m, K)
            weighted_variance = np.array([W_12 @ vc_ii @ W_12, W_12 @ vc_mm @ W_12, W_12 @ vc_im @ W_12])
            operations = np.array([1, 1, -2])
            moments = np.array([
                g[i].T @ weighted_variance[0] @ g[i],
                g[m].T @ weighted_variance[1] @ g[m],
                g[i].T @ weighted_variance[2] @ g[m]
            ]).flatten()
            covariance_mc[i, m] = moments[2]
            covariance_mc[m, i] = moments[2]
            covariance_mc[m, m] = moments[1]
            covariance_mc[i, i] = moments[0]
            # operations.T @ moments = Var(g_i - g_m), which is positive-
            # semi-definite by construction. At extreme cancellation
            # (near-identical model moments), float-rounding can push the
            # value slightly below zero. Treat any negative value as
            # numerically degenerate and propagate NaN downstream rather
            # than raising a ValueError from math.sqrt — the cell is
            # already in the trivially-degenerate regime where the test
            # is undefined.
            quad_form = float(operations.T @ moments)
            if quad_form < 0:
                test_statistic_denominator[i, m] = np.nan
            else:
                test_statistic_denominator[i, m] = math.sqrt(4 * quad_form)

    # RV test statistic (upper triangle only; lower triangle and diagonal are NaN)
    rv_test_statistic = np.full((M, M), np.nan)
    symbols_rv = np.empty((M, M), dtype=object)
    symbols_rv.fill("")
    for m in range(M):
        for i in range(m):
            rv_test_statistic[i, m] = test_statistic_numerator[i, m] / test_statistic_denominator[i, m]
            # Two-sided asymptotic significance markers for the RV test
            # statistic (TRV ~ N(0, 1) under H0). Standard econometric
            # thresholds: 10% / 5% / 1% two-sided => |TRV| > 1.64 / 1.96 / 2.58.
            abs_trv = abs(rv_test_statistic[i, m])
            if not np.isfinite(abs_trv):
                symbols_rv[i, m] = " "
            elif abs_trv > 2.58:
                symbols_rv[i, m] = "***"
            elif abs_trv > 1.96:
                symbols_rv[i, m] = "**"
            elif abs_trv > 1.64:
                symbols_rv[i, m] = "*"
            else:
                symbols_rv[i, m] = " "

    # F statistics — residualize omega on Z_orthogonal; precompute QR once for all models
    phi = np.zeros([M, N, K])
    Q_Z, _ = np.linalg.qr(Z_orthogonal, mode='reduced')
    for m in range(M):
        e = np.reshape(omega[m] - Q_Z @ (Q_Z.T @ omega[m]), [N, 1])
        phi[m] = (e * Z_orthogonal) @ weight_matrix
        if demand_adjustment:
            phi[m] = phi[m] - (h_i - np.transpose(h)) @ np.transpose(W_12 @ adjustment_value[m])

    unscaled_F = np.zeros((M, M))
    F = np.full((M, M), np.nan)
    rho = np.zeros((M, M))
    F_cv_size = np.empty((M, M), dtype=object)
    F_cv_power = np.empty((M, M), dtype=object)
    symbols_size = np.empty((M, M), dtype=object)
    symbols_power = np.empty((M, M), dtype=object)

    # F-stat reliability diagnostic outputs.
    #
    # Architecture (2026-05-01 redesign): the verdict turns on whether F
    # clears the *worst-case* CV across rho^2 ∈ [0, 0.99], which is the
    # ρ̂²-plug-in-robust strength claim. The Wald SE on F̂ is retained as
    # an inspection column (F_se / F_ci_low / F_ci_high) but no longer
    # gates a verdict tier. lambda is retained as a numerical-fragility
    # footnote on plug-in-dependent cells.
    #
    # Verdict tiers:
    #   - "robust": F > worst-case CV. Strength claim survives any rho^2.
    #   - "plug-in dependent": F > plug-in CV but F <= worst-case CV.
    #     Claim depends on ρ̂² being well-estimated.
    #   - "weak": F <= plug-in CV. No strength claim available.
    #   - "trivially-degenerate": ρ̂² is NaN (identical-markup boundary).
    lambda_dmss = np.full((M, M), np.nan)
    F_se = np.full((M, M), np.nan)
    F_ci_low = np.full((M, M), np.nan)
    F_ci_high = np.full((M, M), np.nan)
    verdict = np.empty((M, M), dtype=object)
    strongest_claim_size = np.empty((M, M), dtype=object)
    strongest_claim_power = np.empty((M, M), dtype=object)
    worst_case_cv_size = np.empty((M, M), dtype=object)
    worst_case_cv_power = np.empty((M, M), dtype=object)
    # High-precision F̂ / ρ̂² (populated only when the precision check
    # fires under reliability_check='conditional' or 'always'). Stored
    # for inspection in F_reliability_summary; NaN means the precision
    # check did not run for this cell.
    F_high_precision = np.full((M, M), np.nan)
    rho_squared_high_precision = np.full((M, M), np.nan)

    phi_gram = compute_block_gram(problem, N, clustering_adjustment, phi)
    for (m, i) in itertools.product(range(M), range(M)):
        if i < m:
            variance = np.array([
                extract_block(phi_gram, i, i, K),
                extract_block(phi_gram, m, m, K),
                extract_block(phi_gram, i, m, K)
            ])
            sigma = 1 / K_effective * np.array([
                np.trace(variance[0] @ W_inverse),
                np.trace(variance[1] @ W_inverse),
                np.trace(variance[2] @ W_inverse)
            ])
            # F-stat reliability redesign (2026-05-01):
            #   * F is computed via the algebraically simplified form
            #     F = 2N · F_num / (K · D_rho), where
            #     D_rho = (sigma_0 + sigma_1)^2 - 4 sigma_2^2.
            #     Algebra in MEMO_F_reliability_diagnostic_2026-04-28.md
            #     shows this equals (1 - rho^2) · unscaled_F as written in
            #     paper eq. (17), but eliminates one of two redundant
            #     cancellations: the (1 - rho^2) prefactor and the
            #     (sigma_0 sigma_1 - sigma_2^2) F-denominator both go to
            #     zero as rho -> 1 with their ratio finite. Computing each
            #     separately in floating-point loses precision; the
            #     simplified form has only one source of cancellation
            #     (D_rho), concentrated at the genuine boundary.
            #   * rho is computed via the factored denominator,
            #     D_rho = (sigma_0+sigma_1 - 2 sigma_2) * (sigma_0+sigma_1 + 2 sigma_2).
            #     The first factor is where Cauchy-Schwarz cancellation
            #     happens; the second is well away from zero in normal
            #     applications. Computing the product splits the
            #     cancellation from a `large - large` form into a single
            #     `small × large` product, which is more stable in floats.
            sigma_sum = sigma[0] + sigma[1]
            sigma_minus_diff = sigma_sum - 2 * sigma[2]
            sigma_plus_diff = sigma_sum + 2 * sigma[2]
            D_rho = sigma_minus_diff * sigma_plus_diff
            numerator_sqrt = sigma[0] - sigma[1]
            denominator_sqrt = np.sqrt(D_rho)
            rho[i, m] = numerator_sqrt / denominator_sqrt
            rho_squared = rho[i, m] ** 2

            operations = np.array([sigma[1], sigma[0], -2 * sigma[2]])
            moments = np.array([
                g[i].T @ weight_matrix @ g[i],
                g[m].T @ weight_matrix @ g[m],
                g[i].T @ weight_matrix @ g[m]
            ]).flatten()
            F_numerator = operations @ moments
            F_denominator = sigma[0] * sigma[1] - sigma[2] ** 2
            # Kept for backward-compat callers reading ``unscaled_F`` from
            # ProblemResults (rare; not part of the public API but exposed).
            unscaled_F[i, m] = N / (2 * K_effective) * F_numerator / F_denominator
            # Simplified F: avoids the double cancellation of the literal
            # paper formula. Equivalent at population, more accurate at
            # high rho^2 in floating-point.
            F[i, m] = 2 * N * F_numerator / (K_effective * D_rho)

            # F-stat reliability diagnostic — numerical-fragility part.
            # lambda = ((sigma_0+sigma_1)^2 - 4 sigma_2^2) / (sigma_0+sigma_1)^2
            # measures how much of D_rho's natural scale has been lost to
            # cancellation. Under the redesign, lambda is a footnote on
            # plug-in-dependent cells rather than a verdict in its own
            # right (the worst-case CV check below is the principled
            # robustness signal). Retained for inspection in the summary
            # and diagnostic-detail printing.
            if sigma_sum > 0:
                lambda_dmss[i, m] = D_rho / (sigma_sum ** 2)
            else:
                lambda_dmss[i, m] = np.nan

            # F-stat reliability diagnostic — statistical-fragility part.
            # The asymptotic distribution of F at the implied noncentrality
            # gives an asymptotic SE for the population F. The 95% CI is
            # F ± 1.96 SE. The borderline verdict fires below if this CI
            # overlaps the relevant CV for the strongest size or power claim.
            # See MEMO_F_reliability_diagnostic_2026-04-28.md for derivation.
            if rho_squared < 1 and not np.isnan(rho_squared):
                nc_implied = max(0.0, 2 * K_effective * (F[i, m] / (1 - rho_squared) - 1))
                F_se[i, m] = (1 - rho_squared) / (2 * K_effective) * math.sqrt(
                    2 * (2 * K_effective + 2 * nc_implied)
                )
                F_ci_low[i, m] = F[i, m] - RELIABILITY_CI_LEVEL * F_se[i, m]
                F_ci_high[i, m] = F[i, m] + RELIABILITY_CI_LEVEL * F_se[i, m]

            # v0.4.0rc1 follow-up: guard the critical-values lookup against
            # a NaN rho. Two model pairs with identical markups (e.g. a
            # salience test with opt-out producing the same raw markups)
            # push the F-stat denominator to zero, yielding NaN rho.
            # ``np.where(rho == NaN)`` is always empty so the lookup
            # ``[0][0]`` index used to raise ``IndexError``. Numpy 1.x
            # happened to hit slightly different numerical values on the
            # degenerate pair and sidestepped the crash; numpy 2.x
            # exposes the latent bug. Return NaN critical values and a
            # blank significance symbol — the test statistic itself is
            # NaN in this regime, which is semantically correct.
            # Trivially-degenerate gate. Fires when EITHER:
            #   * ρ̂² is NaN — identical-markup boundary (ρ̂² formula is
            #     0/0 because the V matrix is rank-deficient).
            #   * test_statistic_denominator is NaN — the RV variance
            #     Var(g_i - g_m) went numerically below zero from
            #     extreme cancellation. The RV test statistic is then
            #     undefined regardless of the F-stat path. We propagate
            #     the trivially-degenerate label rather than letting
            #     the F-stat verdict claim "robust" on a cell whose
            #     test conclusion is fundamentally NaN.
            rho_val = rho[i, m]
            denom_val = test_statistic_denominator[i, m]
            if np.isnan(rho_val) or np.isnan(denom_val):
                F_cv_size[i, m] = np.array([np.nan, np.nan, np.nan], dtype=object)
                F_cv_power[i, m] = np.array([np.nan, np.nan, np.nan], dtype=object)
                symbols_size[i, m] = " "
                symbols_power[i, m] = " "
                verdict[i, m] = "trivially-degenerate"
                strongest_claim_size[i, m] = None
                strongest_claim_power[i, m] = None
                worst_case_cv_size[i, m] = np.array([np.nan, np.nan, np.nan], dtype=object)
                worst_case_cv_power[i, m] = np.array([np.nan, np.nan, np.nan], dtype=object)
                continue
            rho_lookup = min(np.round(np.abs(rho_val), 2), 0.99)
            K_lookup = min(K_effective, 30)  # warning for K>30 fired above
            ind = np.where(
                (critical_values_size['K'] == K_lookup) & (critical_values_size['rho'] == rho_lookup)
            )[0][0]
            F_cv_size[i, m] = np.array([
                critical_values_size['r_125'][ind],
                critical_values_size['r_10'][ind],
                critical_values_size['r_075'][ind]
            ], dtype=object)
            F_cv_power[i, m] = np.array([
                critical_values_power['r_50'][ind],
                critical_values_power['r_75'][ind],
                critical_values_power['r_95'][ind]
            ], dtype=object)

            symbols_size[i, m] = (
                " " if F[i, m] < F_cv_size[i, m][0] else
                "†" if F[i, m] < F_cv_size[i, m][1] else  # †
                "††" if F[i, m] < F_cv_size[i, m][2] else "†††"  # ††, †††
            )
            symbols_power[i, m] = (
                " " if F[i, m] < F_cv_power[i, m][0] else
                "^" if F[i, m] < F_cv_power[i, m][1] else
                "^^" if F[i, m] < F_cv_power[i, m][2] else "^^^"
            )

            # F-stat reliability verdict — worst-case CV architecture.
            #
            # Determine the strongest size/power claim the user can make.
            # CV columns ordered as in the published table:
            #   F_cv_size: [r_125, r_10, r_075] (worst-case sizes 12.5%, 10%, 7.5%)
            #   F_cv_power: [r_50, r_75, r_95]  (best-case powers 50%, 75%, 95%)
            #
            # At a given (K, rho), a CV of 0 means "DMSS guarantees this
            # claim automatically without any F threshold" — see paper
            # Sec 5.4: with 2-9 instruments and rs >= 0.075, the set S of
            # dangerous noncentralities is empty, so size distortions are
            # bounded without an F-strength check. We treat such auto-
            # claims as supported. F-supported claims require F > CV.
            strongest_size_idx = -1
            for idx in range(3):
                cv = F_cv_size[i, m][idx]
                if cv == 0 or (cv > 0 and F[i, m] > cv):
                    strongest_size_idx = idx  # later iterations overwrite with stricter
            strongest_power_idx = -1
            for idx in range(3):
                cv = F_cv_power[i, m][idx]
                if cv == 0 or (cv > 0 and F[i, m] > cv):
                    strongest_power_idx = idx

            if strongest_size_idx >= 0:
                strongest_claim_size[i, m] = (
                    f"worst-case size <= {_SIZE_LEVELS_PCT[strongest_size_idx]}%"
                )
            else:
                strongest_claim_size[i, m] = None
            if strongest_power_idx >= 0:
                strongest_claim_power[i, m] = (
                    f"best-case power >= {_POWER_LEVELS_PCT[strongest_power_idx]}%"
                )
            else:
                strongest_claim_power[i, m] = None

            # Worst-case CV scan: would the user's strength claim survive
            # if rho^2 were anywhere in [0, 0.99]? This is the principled
            # robustness check from the redesign — it does not require any
            # finite-sample-aware SE, only a different lookup on the same
            # CV table. Size CVs increase in rho; power CVs decrease.
            worst_case_cv_size[i, m] = _worst_case_cv_size(
                critical_values_size, K_effective,
            )
            worst_case_cv_power[i, m] = _worst_case_cv_power(
                critical_values_power, K_effective,
            )

            # Three-tier verdict.
            # A claim is "robust" when it survives at the worst-case CV
            # across rho^2. Auto-claims (CV = 0 at all rho^2) are robust
            # by definition. F-supported claims (F > plug-in CV > 0) are
            # robust if F > worst-case CV. If neither size nor power
            # claim is made, the verdict is "weak".
            size_made = strongest_size_idx >= 0
            power_made = strongest_power_idx >= 0
            size_robust = True
            power_robust = True
            if size_made:
                worst_size_cv = worst_case_cv_size[i, m][strongest_size_idx]
                if worst_size_cv > 0:
                    size_robust = F[i, m] > worst_size_cv
                # else: auto-claim, robust by definition (size_robust=True)
            if power_made:
                worst_power_cv = worst_case_cv_power[i, m][strongest_power_idx]
                if worst_power_cv > 0:
                    power_robust = F[i, m] > worst_power_cv

            if not (size_made or power_made):
                verdict[i, m] = "weak"
            elif size_robust and power_robust:
                verdict[i, m] = "robust"
            else:
                verdict[i, m] = "plug-in dependent"

            # High-precision precision check (reliability_check kwarg).
            #
            # Modes:
            #   - 'off': skip the check entirely (preserves the
            #     three-tier verdict above).
            #   - 'conditional' (default): only fire when the cell is
            #     in the precision-relevant band — λ < threshold AND
            #     verdict is 'plug-in dependent'. The mpmath cost is
            #     paid only when ρ̂² is in the cancellation regime AND
            #     the strength claim might flip.
            #   - 'always': fire on every non-trivial cell. Useful for
            #     paper-table generation where speed isn't critical.
            #
            # When the high-precision F differs enough from the double-
            # precision F to flip the test outcome at the strongest-
            # claim CV, the verdict is upgraded to "numerically unstable".
            should_check = False
            if reliability_check == 'always':
                should_check = True
            elif reliability_check == 'conditional':
                in_low_lambda = (
                    not np.isnan(lambda_dmss[i, m])
                    and lambda_dmss[i, m] < RELIABILITY_LAMBDA_THRESHOLD
                )
                in_band = (verdict[i, m] == 'plug-in dependent')
                should_check = in_low_lambda and in_band
            # else 'off' -> skip

            if should_check:
                phi_blocks = (variance[0], variance[1], variance[2])
                F_hp, rho2_hp = _recompute_F_high_precision(
                    phi_blocks=phi_blocks,
                    W_inverse=W_inverse,
                    weight_matrix=weight_matrix,
                    g_i=g[i],
                    g_m=g[m],
                    K=K_effective,
                    N=N,
                    prec=reliability_precision_dps,
                )
                F_high_precision[i, m] = F_hp
                rho_squared_high_precision[i, m] = rho2_hp

                # Decide whether the high-precision F flips the verdict.
                # The relevant test is whether F_hp would still clear
                # the plug-in CV at the strongest-claim level — if it
                # doesn't, the double-precision F̂'s strength claim is
                # numerically unreliable.
                hp_passes = True
                if size_made and strongest_size_idx >= 0:
                    plug_in_size = F_cv_size[i, m][strongest_size_idx]
                    if plug_in_size > 0 and F_hp <= plug_in_size:
                        hp_passes = False
                if power_made and strongest_power_idx >= 0 and hp_passes:
                    plug_in_power = F_cv_power[i, m][strongest_power_idx]
                    if plug_in_power > 0 and F_hp <= plug_in_power:
                        hp_passes = False
                if not hp_passes:
                    verdict[i, m] = "numerically unstable"
        else:
            symbols_size[i, m] = ""
            symbols_power[i, m] = ""
            verdict[i, m] = None
            strongest_claim_size[i, m] = None
            strongest_claim_power[i, m] = None
            worst_case_cv_size[i, m] = None
            worst_case_cv_power[i, m] = None

    # model confidence set
    all_model_combinations = list(itertools.combinations(range(M), 2))
    n_combinations = len(all_model_combinations)
    model_confidence_set_variance = np.zeros([n_combinations, 1])
    sigma_mcs = np.zeros([n_combinations, n_combinations])
    for index_i, model_i in enumerate(all_model_combinations):
        model_confidence_set_variance[index_i] = test_statistic_denominator[model_i[0], model_i[1]] / 2
        for index_j, model_j in enumerate(all_model_combinations):
            term1 = covariance_mc[model_i[0], model_j[0]] - covariance_mc[model_i[1], model_j[0]]
            term2 = covariance_mc[model_i[0], model_j[1]] - covariance_mc[model_i[1], model_j[1]]
            sigma_mcs[index_j, index_i] = term1 - term2
    sigma_mcs = sigma_mcs / (model_confidence_set_variance @ model_confidence_set_variance.T)

    mcs_pvalues = compute_mcs(
        rv_test_statistic, sigma_mcs, model_confidence_set_variance, M, all_model_combinations
    )

    return {
        'g': g, 'Q': Q,
        'RV_numerator': test_statistic_numerator,
        'RV_denominator': test_statistic_denominator,
        'rv_test_statistic': rv_test_statistic,
        'F': F, 'unscaled_F': unscaled_F,
        'mcs_pvalues': mcs_pvalues,
        'rho': rho,
        'F_cv_size': F_cv_size,
        'F_cv_power': F_cv_power,
        'symbols_size': symbols_size,
        'symbols_power': symbols_power,
        'symbols_rv': symbols_rv,
        'lambda_dmss': lambda_dmss,
        'F_se': F_se,
        'F_ci_low': F_ci_low,
        'F_ci_high': F_ci_high,
        'verdict': verdict,
        'strongest_claim_size': strongest_claim_size,
        'strongest_claim_power': strongest_claim_power,
        'worst_case_cv_size': worst_case_cv_size,
        'worst_case_cv_power': worst_case_cv_power,
        'F_high_precision': F_high_precision,
        'rho_squared_high_precision': rho_squared_high_precision,
    }


def compute_mcs(
        rv_test_statistic: Array, sigma_mcs: Array,
        model_confidence_set_variance: Array, M: int,
        all_model_combinations: List[Tuple[int, int]],
) -> Array:
    """Compute model confidence set p-values by iteratively eliminating the worst-fitting model.

    Moved from ``Problem._compute_mcs`` in v0.4 step 8d. Math is
    unchanged. Stateless (no ``self``): ``options.random_seed`` and
    ``options.ndraws`` are read directly from the global options module.
    """
    rng = np.random.default_rng(options.random_seed)
    model_confidence_set = np.array(range(M))
    mcs_pvalues = np.ones([M, 1])
    converged = False
    while not converged:
        if np.shape(model_confidence_set)[0] == 2:
            max_test_statistic = rv_test_statistic[model_confidence_set[0], model_confidence_set[1]]
            if np.sign(max_test_statistic) >= 0:
                worst_fit = model_confidence_set[0]
                max_test_statistic = -max_test_statistic
            else:
                worst_fit = model_confidence_set[1]
            mcs_pvalues[worst_fit] = 2 * norm.cdf(max_test_statistic)
            converged = True
        else:
            current_combinations = list(itertools.combinations(model_confidence_set, 2))
            sigma_index = np.array(
                [all_model_combinations.index(c) for c in current_combinations], dtype=int
            )
            model_1 = [c[0] for c in current_combinations]
            model_2 = [c[1] for c in current_combinations]
            test_stats = rv_test_statistic[model_1, model_2]
            index = np.argmax(abs(test_stats))
            max_test_statistic = test_stats[index]

            if np.sign(max_test_statistic) >= 0:
                worst_fit = model_1[index]
            else:
                worst_fit = model_2[index]
                max_test_statistic = -max_test_statistic

            cov = sigma_mcs[sigma_index[:, None], sigma_index]
            simulated = rng.multivariate_normal(np.zeros(len(current_combinations)), cov, options.ndraws)
            mcs_pvalues[worst_fit] = np.mean(np.amax(abs(simulated), 1) > max_test_statistic)
            model_confidence_set = np.delete(
                model_confidence_set, np.where(model_confidence_set == worst_fit)
            )
    return mcs_pvalues


def compute_block_gram(
        problem: Any, N: int, clustering_adjustment: bool, var: Array,
) -> Array:
    """Compute the ``(M*K, M*K)`` block Gram matrix for all model pairs at once.

    Moved from ``Problem._compute_block_gram`` in v0.4 step 8d. Math is
    unchanged.

    Returns gram such that the ``(K, K)`` variance block for model pair
    ``(i, m)`` is ``gram[i*K:(i+1)*K, m*K:(m+1)*K]``.

    With clustering, uses the Cameron-Gelbach-Miller cluster-sum formula:
    ``V_clustered = (1/N) * sum_c s_c s_c'`` where ``s_c`` is the sum of
    ``var`` within cluster ``c``. Mathematically equivalent to the
    per-pair roll-based computation but runs in ``O(M*N)`` numpy
    operations instead of ``O(C * S * M^2)`` Python loop iterations.

    The only access to ``problem`` is ``problem.products.clustering_ids``
    when ``clustering_adjustment`` is True; the function is otherwise
    stateless.
    """
    M, _, K = var.shape
    if clustering_adjustment:
        cluster_ids_flat = problem.products.clustering_ids.flatten()
        unique_clusters = np.unique(cluster_ids_flat)
        C = len(unique_clusters)
        cluster_map = {c: idx for idx, c in enumerate(unique_clusters)}
        cluster_idx = np.array([cluster_map[c] for c in cluster_ids_flat])
        cluster_sums = np.zeros((M, C, K), dtype=var.dtype)
        for m in range(M):
            np.add.at(cluster_sums[m], cluster_idx, var[m])
        cs_flat = cluster_sums.transpose(1, 0, 2).reshape(C, M * K)
        return (1 / N) * cs_flat.T @ cs_flat
    else:
        var_flat = var.transpose(1, 0, 2).reshape(N, M * K)
        return (1 / N) * var_flat.T @ var_flat


def extract_block(gram: Array, i: int, m: int, K: int) -> Array:
    """Extract a ``(K, K)`` block from the block Gram matrix.

    Moved from ``Problem._extract_block`` in v0.4 step 8d.
    """
    return gram[i * K:(i + 1) * K, m * K:(m + 1) * K]
