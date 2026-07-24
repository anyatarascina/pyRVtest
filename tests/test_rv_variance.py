"""Unit tests for the scalar-score RV variance (2026-07 correction).

The RV denominator and the MCS covariances are computed from the scalar
score ``phi_m,i = 2 v_mi omega_mi - v_mi^2 - Q_m`` with ``v_mi = z_i' W g_m``
(the exact delta method on ``Q_m = g_m' W g_m``); see
``notes/variance_proof_note.pdf`` and ``notes/Memo_appendixB.pdf``.

Four invariants pinned here:
1. Basis invariance: T_RV is unchanged under an invertible linear
   transformation of the instruments (the former W^{3/4} formula was not).
2. K = 1 equivalence: with a single instrument the scalar score coincides
   exactly with the former psi-based formula (the scalar case is the one
   regime where the old matrix-power algebra was valid).
3. Clustered variance equals the explicit cluster-sum formula
   (memo eq. (13)).
4. MCS covariance consistency: the pair-level correlation matrix built
   from the scalar scores is a valid correlation matrix and reproduces
   the package's MCS p-values through ``compute_mcs``.
"""

import itertools

import numpy as np
import pytest

import pyRVtest
from pyRVtest.solve.test_engine import compute_mcs

from .test_analytical import _build_base_dgp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hand_scalar_scores(product_data, markup_cols, iv_cols):
    """Independently compute z_orth, W, g, Q and the scalar RV scores."""
    N = len(product_data)
    w_full = np.hstack([np.ones((N, 1)), product_data[['cost_shifter']].to_numpy()])
    z = product_data[iv_cols].to_numpy()
    prices = product_data['prices'].to_numpy()[:, None]

    Q_w, _ = np.linalg.qr(w_full, mode='reduced')
    z_orth = z - Q_w @ (Q_w.T @ z)

    W_inv = (1 / N) * z_orth.T @ z_orth
    W = np.linalg.pinv(W_inv)

    scores, Qs = [], []
    for col in markup_cols:
        mc = prices - product_data[col].to_numpy()[:, None]
        omega = (mc - Q_w @ (Q_w.T @ mc)).flatten()
        g = (1 / N) * z_orth.T @ omega
        Q_m = float(g @ W @ g)
        v = z_orth @ (W @ g)
        scores.append(2 * v * omega - v ** 2 - Q_m)
        Qs.append(Q_m)
    return np.array(scores), np.array(Qs)


def _solve_problem(product_data, markup_cols, iv_terms, **solve_kwargs):
    models = [
        pyRVtest.UserSuppliedMarkups(markups=col) for col in markup_cols
    ]
    problem = pyRVtest.Problem(
        cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
        instrument_formulation=pyRVtest.Formulation(f'0 + {iv_terms}'),
        models=models,
        product_data=product_data,
        demand_results=None,
    )
    verbose = pyRVtest.options.verbose
    pyRVtest.options.verbose = False
    try:
        return problem.solve(
            demand_adjustment=False,
            clustering_adjustment=solve_kwargs.pop('clustering_adjustment', False),
            **solve_kwargs,
        )
    finally:
        pyRVtest.options.verbose = verbose


@pytest.fixture(scope='module')
def base_data():
    product_data, dgp = _build_base_dgp()
    return product_data, dgp


# ---------------------------------------------------------------------------
# 1. Basis invariance
# ---------------------------------------------------------------------------

class TestBasisInvariance:
    """T_RV must not depend on the basis in which instruments are supplied."""

    def test_invariant_under_invertible_transformation(self, base_data):
        product_data, _ = base_data
        product_data = product_data.copy()

        transform = np.array([
            [2.0, 1.0, 0.0],
            [0.0, 1.0, 3.0],
            [1.0, 0.0, 1.0],
        ])
        assert abs(np.linalg.det(transform)) > 1e-8
        z = product_data[['iv0', 'iv1', 'iv2']].to_numpy()
        z_t = z @ transform
        for k in range(3):
            product_data[f'ivt{k}'] = z_t[:, k]

        results = _solve_problem(
            product_data, ['markups_m1', 'markups_m2'], 'iv0 + iv1 + iv2'
        )
        results_t = _solve_problem(
            product_data, ['markups_m1', 'markups_m2'], 'ivt0 + ivt1 + ivt2'
        )

        np.testing.assert_allclose(
            results_t.TRV[0][0, 1], results.TRV[0][0, 1], rtol=1e-8,
            err_msg="T_RV is not invariant to an invertible instrument transformation",
        )
        np.testing.assert_allclose(
            results_t.RV_denominator[0][0, 1], results.RV_denominator[0][0, 1], rtol=1e-8,
        )
        np.testing.assert_allclose(
            np.asarray(results_t.MCS_pvalues[0], dtype=float),
            np.asarray(results.MCS_pvalues[0], dtype=float),
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# 2. K = 1 equivalence with the former psi-based formula
# ---------------------------------------------------------------------------

class TestSingleInstrumentEquivalence:
    """With one instrument the old W^{1/2}/W^{3/4} algebra was exact; the new
    scalar score must reproduce it to float precision."""

    def test_matches_old_formula_at_k1(self, base_data):
        product_data, _ = base_data
        N = len(product_data)

        results = _solve_problem(
            product_data, ['markups_m1', 'markups_m2'], 'iv0'
        )

        # Hand-compute the OLD psi-based denominator in the scalar case.
        w_full = np.hstack([np.ones((N, 1)), product_data[['cost_shifter']].to_numpy()])
        z = product_data[['iv0']].to_numpy()
        prices = product_data['prices'].to_numpy()[:, None]
        Q_w, _ = np.linalg.qr(w_full, mode='reduced')
        z_orth = z - Q_w @ (Q_w.T @ z)
        w_inv = float(((1 / N) * z_orth.T @ z_orth).item())
        w_scalar = 1.0 / w_inv

        psis, gs = [], []
        for col in ['markups_m1', 'markups_m2']:
            mc = prices - product_data[col].to_numpy()[:, None]
            omega = (mc - Q_w @ (Q_w.T @ mc)).flatten()
            g = float((1 / N) * z_orth.flatten() @ omega)
            z_flat = z_orth.flatten()
            psi = w_scalar ** 0.5 * z_flat * omega - 0.5 * w_scalar ** 1.5 * g * z_flat ** 2
            psis.append(psi - psi.mean())
            gs.append(g)

        v11 = (1 / N) * psis[0] @ psis[0]
        v22 = (1 / N) * psis[1] @ psis[1]
        v12 = (1 / N) * psis[0] @ psis[1]
        sigma2_old = 4 * w_scalar * (
            gs[0] ** 2 * v11 + gs[1] ** 2 * v22 - 2 * gs[0] * gs[1] * v12
        )

        np.testing.assert_allclose(
            results.RV_denominator[0][0, 1], np.sqrt(sigma2_old), rtol=1e-8,
            err_msg="new scalar-score denominator disagrees with the old formula at K=1",
        )


# ---------------------------------------------------------------------------
# 3. Clustered variance = explicit cluster-sum formula (memo eq. (13))
# ---------------------------------------------------------------------------

class TestClusteredVariance:

    def test_cluster_sum_formula(self, base_data):
        product_data, _ = base_data

        results = _solve_problem(
            product_data, ['markups_m1', 'markups_m2'], 'iv0 + iv1 + iv2',
            clustering_adjustment=True,
        )

        scores, _ = _hand_scalar_scores(
            product_data, ['markups_m1', 'markups_m2'], ['iv0', 'iv1', 'iv2']
        )
        N = scores.shape[1]
        cluster_ids = product_data['clustering_ids'].to_numpy()
        unique_c = np.unique(cluster_ids)
        sums = np.zeros((2, len(unique_c)))
        for j, c in enumerate(unique_c):
            mask = cluster_ids == c
            sums[:, j] = scores[:, mask].sum(axis=1)
        diff = sums[0] - sums[1]
        sigma2_cluster = (1 / N) * diff @ diff

        np.testing.assert_allclose(
            results.RV_denominator[0][0, 1], np.sqrt(sigma2_cluster), rtol=1e-8,
        )


# ---------------------------------------------------------------------------
# 4. MCS covariance consistency
# ---------------------------------------------------------------------------

class TestMCSCovariance:

    def test_correlation_matrix_and_pvalues(self, base_data):
        product_data, _ = base_data
        product_data = product_data.copy()
        product_data['markups_m3'] = 1.3 * product_data['markups_m1']
        markup_cols = ['markups_m1', 'markups_m2', 'markups_m3']
        M = 3

        results = _solve_problem(product_data, markup_cols, 'iv0 + iv1 + iv2')

        scores, _ = _hand_scalar_scores(
            product_data, markup_cols, ['iv0', 'iv1', 'iv2']
        )
        N = scores.shape[1]
        C = (1 / N) * scores @ scores.T  # (M, M) score covariances

        combos = list(itertools.combinations(range(M), 2))
        denom = np.array([
            np.sqrt(C[a, a] + C[b, b] - 2 * C[a, b]) for (a, b) in combos
        ])
        np.testing.assert_allclose(
            denom[0], results.RV_denominator[0][0, 1], rtol=1e-8,
        )

        sigma_mcs = np.zeros((len(combos), len(combos)))
        for j, (j0, j1) in enumerate(combos):
            for i, (i0, i1) in enumerate(combos):
                cov = C[i0, j0] - C[i1, j0] - C[i0, j1] + C[i1, j1]
                sigma_mcs[j, i] = cov / (denom[i] * denom[j])

        # A correlation matrix: unit diagonal, symmetric, PSD.
        np.testing.assert_allclose(np.diag(sigma_mcs), np.ones(len(combos)), atol=1e-10)
        np.testing.assert_allclose(sigma_mcs, sigma_mcs.T, atol=1e-10)
        assert np.linalg.eigvalsh(sigma_mcs).min() > -1e-8

        # Reproduce the package's MCS p-values from the hand-built covariance.
        trv = np.asarray(results.TRV[0], dtype=float)
        mcs = compute_mcs(
            trv, sigma_mcs, denom.reshape(-1, 1), M, combos
        )
        np.testing.assert_allclose(
            np.asarray(mcs, dtype=float).flatten(),
            np.asarray(results.MCS_pvalues[0], dtype=float).flatten(),
            atol=1e-12,
        )
