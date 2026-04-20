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


def compute_instrument_results(
        problem: Any, instrument: int, M: int, N: int, omega: Array,
        demand_adjustment: bool, gradient_markups: Optional[Array],
        H_prime_wd: Optional[Array], H: Optional[Array],
        h_i: Optional[Array], h: Optional[Array],
        clustering_adjustment: bool,
        critical_values_size: Array, critical_values_power: Array,
        endog_hat: Optional[Array] = None,
        gradient_gamma: Optional[Array] = None,
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
            test_statistic_denominator[i, m] = math.sqrt(4 * (operations.T @ moments))

    # RV test statistic (upper triangle only; lower triangle and diagonal are NaN)
    rv_test_statistic = np.full((M, M), np.nan)
    for m in range(M):
        for i in range(m):
            rv_test_statistic[i, m] = test_statistic_numerator[i, m] / test_statistic_denominator[i, m]

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
            numerator_sqrt = sigma[0] - sigma[1]
            denominator_sqrt = np.sqrt((sigma[0] + sigma[1]) ** 2 - 4 * sigma[2] ** 2)
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
            unscaled_F[i, m] = N / (2 * K_effective) * F_numerator / F_denominator
            F[i, m] = (1 - rho_squared) * unscaled_F[i, m]

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
            rho_val = rho[i, m]
            if np.isnan(rho_val):
                F_cv_size[i, m] = np.array([np.nan, np.nan, np.nan], dtype=object)
                F_cv_power[i, m] = np.array([np.nan, np.nan, np.nan], dtype=object)
                symbols_size[i, m] = " "
                symbols_power[i, m] = " "
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
                "*" if F[i, m] < F_cv_size[i, m][1] else
                "**" if F[i, m] < F_cv_size[i, m][2] else "***"
            )
            symbols_power[i, m] = (
                " " if F[i, m] < F_cv_power[i, m][0] else
                "^" if F[i, m] < F_cv_power[i, m][1] else
                "^^" if F[i, m] < F_cv_power[i, m][2] else "^^^"
            )
        else:
            symbols_size[i, m] = ""
            symbols_power[i, m] = ""

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
