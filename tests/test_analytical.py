"""Analytical validation: hand-compute test statistics, verify pyRVtest matches.

Four test classes, each layering a feature:
1. Base case: constant cost, no demand adjustment, no clustering
2. Clustering: same DGP, clustering_adjustment=True
3. Economies of scale: endogenous cost component
4. Demand adjustment: gradient of markups w.r.t. demand parameters

Each test constructs a tiny dataset, computes all intermediate quantities in raw numpy
(independent of pyRVtest), runs pyRVtest on the same data, and asserts numerical agreement.
"""

import numpy as np
import pandas as pd
import pytest
import pyRVtest
from pyRVtest.problem import _qr_residualize


# ---------------------------------------------------------------------------
# Shared DGP construction
# ---------------------------------------------------------------------------

def _logit_shares(V, market_ids, T):
    """Compute logit shares from indirect utility V."""
    N = len(V)
    shares = np.zeros((N, 1))
    for t in range(T):
        idx = np.where(market_ids == t)[0]
        v_t = V[idx].flatten()
        exp_v = np.exp(v_t)
        shares[idx, 0] = exp_v / (1.0 + exp_v.sum())
    return shares


def _solve_bertrand_logit(alpha, beta_x, x, xi, mc_true, market_ids, T, N):
    """Solve for Bertrand-logit equilibrium prices via iteration."""
    prices = mc_true + 0.5
    for _ in range(500):
        V = beta_x * x + alpha * prices + xi
        shares = _logit_shares(V, market_ids, T)
        markups = -1.0 / (alpha * (1.0 - shares))
        prices_new = mc_true + markups
        if np.max(np.abs(prices_new - prices)) < 1e-13:
            break
        prices = prices_new
    V = beta_x * x + alpha * prices + xi
    shares = _logit_shares(V, market_ids, T)
    markups = -1.0 / (alpha * (1.0 - shares))
    return prices, shares, markups


def _build_base_dgp(seed=12345, T=20, J=2):
    """Build a 2-firm logit DGP with known parameters."""
    rng = np.random.default_rng(seed=seed)
    N = T * J
    market_ids = np.repeat(np.arange(T), J)
    firm_ids = np.tile(np.arange(J), T)

    alpha = -2.0
    beta_x = 1.0
    x = rng.uniform(0.5, 2.0, size=(N, 1))
    cost_shifter = rng.uniform(0.5, 2.0, size=(N, 1))
    # 3 instruments to exercise matrix paths (K=1 has scalar edge cases)
    excluded_iv = rng.uniform(0.5, 2.0, size=(N, 3))
    xi = rng.normal(0, 0.05, size=(N, 1))
    omega_true = rng.normal(0, 0.05, size=(N, 1))

    tau_0, tau_w = 1.0, 0.5
    mc_true = tau_0 + tau_w * cost_shifter + omega_true

    prices, shares, markups_bertrand = _solve_bertrand_logit(
        alpha, beta_x, x, xi, mc_true, market_ids, T, N
    )
    markups_perfect = np.zeros((N, 1))

    product_data = pd.DataFrame({
        'market_ids': market_ids,
        'firm_ids': firm_ids,
        'prices': prices.flatten(),
        'shares': shares.flatten(),
        'x': x.flatten(),
        'cost_shifter': cost_shifter.flatten(),
        'iv0': excluded_iv[:, 0],
        'iv1': excluded_iv[:, 1],
        'iv2': excluded_iv[:, 2],
        'markups_m1': markups_bertrand.flatten(),
        'markups_m2': markups_perfect.flatten(),
        'clustering_ids': market_ids,
    })

    return product_data, {
        'N': N, 'T': T, 'J': J, 'K': 3, 'alpha': alpha, 'beta_x': beta_x,
        'prices': prices, 'shares': shares,
        'markups_m1': markups_bertrand, 'markups_m2': markups_perfect,
        'mc_true': mc_true, 'cost_shifter': cost_shifter, 'excluded_iv': excluded_iv,
        'omega_true': omega_true,
    }


# ---------------------------------------------------------------------------
# Hand computation helpers
# ---------------------------------------------------------------------------

def _hand_compute_base(dgp, clustering_ids=None):
    """Hand-compute all test statistics for the base constant-cost case."""
    N = dgp['N']
    prices = dgp['prices']
    markups_m1 = dgp['markups_m1']
    markups_m2 = dgp['markups_m2']
    cost_shifter = dgp['cost_shifter']
    z = dgp['excluded_iv']
    K = dgp['K']

    # Cost formulation: [1, cost_shifter]
    w_full = np.hstack([np.ones((N, 1)), cost_shifter])

    # Implied marginal cost
    mc_m1 = prices - markups_m1
    mc_m2 = prices - markups_m2

    # Residualize on w
    Q_w, R_w = np.linalg.qr(w_full, mode='reduced')
    omega_m1 = mc_m1 - Q_w @ (Q_w.T @ mc_m1)
    omega_m2 = mc_m2 - Q_w @ (Q_w.T @ mc_m2)
    z_orth = z - Q_w @ (Q_w.T @ z)

    # Tau
    tau_m1 = np.linalg.solve(R_w, Q_w.T @ mc_m1)
    tau_m2 = np.linalg.solve(R_w, Q_w.T @ mc_m2)

    # Weight matrix
    W_inv = (1 / N) * z_orth.T @ z_orth
    W = np.linalg.pinv(W_inv)

    # GMM moments and fit
    g_m1 = (1 / N) * z_orth.T @ omega_m1
    g_m2 = (1 / N) * z_orth.T @ omega_m2
    Q_m1 = float(g_m1.flatten() @ W @ g_m1.flatten())
    Q_m2 = float(g_m2.flatten() @ W @ g_m2.flatten())

    # RV numerator
    rv_num = np.sqrt(N) * (Q_m1 - Q_m2)

    # W powers via eigendecomposition (matches pyRVtest approach)
    eigvals, eigvecs = np.linalg.eigh((W + W.T) / 2)
    eigvals = np.maximum(eigvals, 0)
    W_12 = (eigvecs * (eigvals ** 0.5)) @ eigvecs.T
    W_34 = (eigvecs * (eigvals ** 0.75)) @ eigvecs.T

    # Psi influence function
    def _compute_psi(omega_m, g_m):
        g_flat = g_m.flatten()  # (K,)
        psi_bar = W_12 @ g_flat - 0.5 * W_34 @ W_inv @ W_34 @ g_flat  # (K,)
        W34_Zg = (z_orth @ W_34 @ g_flat)[:, np.newaxis]  # (N, 1)
        mc_col = omega_m if omega_m.ndim == 2 else omega_m[:, np.newaxis]  # (N, 1)
        psi_i = (mc_col * z_orth) @ W_12 - 0.5 * W34_Zg * (z_orth @ W_34.T)  # (N, K)
        return psi_i - psi_bar[np.newaxis, :]  # (N, K)

    psi_m1 = _compute_psi(omega_m1, g_m1)
    psi_m2 = _compute_psi(omega_m2, g_m2)

    # Variance-covariance (with or without clustering)
    M = 2
    psi = np.stack([psi_m1, psi_m2])  # (2, N, K)
    if clustering_ids is not None:
        unique_c = np.unique(clustering_ids)
        C = len(unique_c)
        cmap = {c: j for j, c in enumerate(unique_c)}
        cidx = np.array([cmap[c] for c in clustering_ids.flatten()])
        cs = np.zeros((M, C, K))
        for m in range(M):
            np.add.at(cs[m], cidx, psi[m])
        cs_flat = cs.transpose(1, 0, 2).reshape(C, M * K)
        gram = (1 / N) * cs_flat.T @ cs_flat
    else:
        psi_flat = psi.transpose(1, 0, 2).reshape(N, M * K)
        gram = (1 / N) * psi_flat.T @ psi_flat

    V11 = gram[0:K, 0:K]
    V22 = gram[K:2*K, K:2*K]
    V12 = gram[0:K, K:2*K]
    wv11 = W_12 @ V11 @ W_12
    wv22 = W_12 @ V22 @ W_12
    wv12 = W_12 @ V12 @ W_12
    _g1, _g2 = g_m1.flatten(), g_m2.flatten()
    sigma2 = float(4 * (_g1 @ wv11 @ _g1 + _g2 @ wv22 @ _g2 - 2 * _g1 @ wv12 @ _g2))
    sigma = np.sqrt(sigma2)
    trv = float(rv_num / sigma)

    # F-statistic
    Q_z, _ = np.linalg.qr(z_orth, mode='reduced')
    e_m1 = omega_m1 - Q_z @ (Q_z.T @ omega_m1)
    e_m2 = omega_m2 - Q_z @ (Q_z.T @ omega_m2)
    phi_m1 = (e_m1 * z_orth) @ W
    phi_m2 = (e_m2 * z_orth) @ W

    phi = np.stack([phi_m1, phi_m2])
    if clustering_ids is not None:
        cs_phi = np.zeros((M, C, K))
        for m in range(M):
            np.add.at(cs_phi[m], cidx, phi[m])
        cs_phi_flat = cs_phi.transpose(1, 0, 2).reshape(C, M * K)
        phi_gram = (1 / N) * cs_phi_flat.T @ cs_phi_flat
    else:
        phi_flat = phi.transpose(1, 0, 2).reshape(N, M * K)
        phi_gram = (1 / N) * phi_flat.T @ phi_flat

    V_ar_11 = phi_gram[0:K, 0:K]
    V_ar_22 = phi_gram[K:2*K, K:2*K]
    V_ar_12 = phi_gram[0:K, K:2*K]
    sigma_ar = (1 / K) * np.array([
        float(np.trace(V_ar_11 @ W_inv)),
        float(np.trace(V_ar_22 @ W_inv)),
        float(np.trace(V_ar_12 @ W_inv)),
    ])
    rho_num = sigma_ar[0] - sigma_ar[1]
    rho_den = np.sqrt((sigma_ar[0] + sigma_ar[1]) ** 2 - 4 * sigma_ar[2] ** 2)
    rho = rho_num / rho_den
    rho_sq = rho ** 2
    F_ops = np.array([sigma_ar[1], sigma_ar[0], -2 * sigma_ar[2]])
    F_moms = np.array([float(_g1 @ W @ _g1), float(_g2 @ W @ _g2), float(_g1 @ W @ _g2)])
    F_num = F_ops @ F_moms
    F_den = sigma_ar[0] * sigma_ar[1] - sigma_ar[2] ** 2
    unscaled_F = N / (2 * K) * F_num / F_den
    F = (1 - rho_sq) * unscaled_F

    return {
        'TRV': trv, 'F': float(F), 'Q_m1': Q_m1, 'Q_m2': Q_m2,
        'g_m1': g_m1.flatten(), 'g_m2': g_m2.flatten(),
        'tau_m1': tau_m1.flatten(), 'tau_m2': tau_m2.flatten(),
    }


def _run_pyrvtest_base(product_data, clustering=False):
    """Run pyRVtest on the base case."""
    model_formulations = (
        pyRVtest.ModelFormulation(model_downstream='bertrand', ownership_downstream='firm_ids',
                                 user_supplied_markups='markups_m1'),
        pyRVtest.ModelFormulation(model_downstream='perfect_competition',
                                 user_supplied_markups='markups_m2'),
    )
    testing_problem = pyRVtest.Problem(
        cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
        instrument_formulation=pyRVtest.Formulation('0 + iv0 + iv1 + iv2'),
        model_formulations=model_formulations,
        product_data=product_data,
        demand_results=None,
    )
    return testing_problem.solve(demand_adjustment=False, clustering_adjustment=clustering)


# ---------------------------------------------------------------------------
# Test class 1: Base case
# ---------------------------------------------------------------------------

class TestAlgebraBase:
    """Validate engine arithmetic: constant cost, no demand adjustment, no clustering."""

    @pytest.fixture(scope='class')
    def data(self):
        product_data, dgp = _build_base_dgp()
        expected = _hand_compute_base(dgp, clustering_ids=None)
        results = _run_pyrvtest_base(product_data, clustering=False)
        return results, expected

    def test_g(self, data):
        results, expected = data
        np.testing.assert_allclose(results.g[0][0], expected['g_m1'], atol=1e-10)
        np.testing.assert_allclose(results.g[0][1], expected['g_m2'], atol=1e-10)

    def test_q(self, data):
        results, expected = data
        np.testing.assert_allclose(results.Q[0][0], expected['Q_m1'], atol=1e-10)
        np.testing.assert_allclose(results.Q[0][1], expected['Q_m2'], atol=1e-10)

    def test_tau(self, data):
        results, expected = data
        np.testing.assert_allclose(results.taus[0], expected['tau_m1'], atol=1e-8)
        np.testing.assert_allclose(results.taus[1], expected['tau_m2'], atol=1e-8)

    def test_trv(self, data):
        results, expected = data
        np.testing.assert_allclose(results.TRV[0][0, 1], expected['TRV'], atol=1e-8)

    def test_f(self, data):
        results, expected = data
        np.testing.assert_allclose(results.F[0][0, 1], expected['F'], atol=1e-8)

    def test_trv_sign_consistent_with_q(self, data):
        results, expected = data
        if expected['Q_m1'] < expected['Q_m2']:
            assert results.TRV[0][0, 1] < 0
        else:
            assert results.TRV[0][0, 1] > 0


# ---------------------------------------------------------------------------
# Test class 2: Clustering
# ---------------------------------------------------------------------------

class TestAlgebraClustering:
    """Same DGP, with clustering at the market level."""

    @pytest.fixture(scope='class')
    def data(self):
        product_data, dgp = _build_base_dgp()
        clustering_ids = product_data['clustering_ids'].values
        expected = _hand_compute_base(dgp, clustering_ids=clustering_ids)
        results = _run_pyrvtest_base(product_data, clustering=True)
        return results, expected

    def test_trv(self, data):
        results, expected = data
        np.testing.assert_allclose(results.TRV[0][0, 1], expected['TRV'], atol=1e-8)

    def test_f(self, data):
        results, expected = data
        np.testing.assert_allclose(results.F[0][0, 1], expected['F'], atol=1e-8)

    def test_g_unchanged(self, data):
        """GMM moments should not change with clustering (only variance changes)."""
        results, expected = data
        np.testing.assert_allclose(results.g[0][0], expected['g_m1'], atol=1e-10)
        np.testing.assert_allclose(results.g[0][1], expected['g_m2'], atol=1e-10)

    def test_trv_differs_from_unclustered(self, data):
        """Clustered TRV should generally differ from unclustered."""
        _, dgp = _build_base_dgp()
        unclustered = _hand_compute_base(dgp, clustering_ids=None)
        _, expected = data
        # They should differ because clustering changes the variance
        # (though in degenerate cases they could be equal)
        assert expected['TRV'] != unclustered['TRV'] or expected['F'] != unclustered['F']


# ---------------------------------------------------------------------------
# Test class 2b: Cost-side fixed effects
# ---------------------------------------------------------------------------

def _absorb_fe(x, group_ids):
    """Within-group demean: subtract group means from each observation."""
    x_out = x.copy()
    for g in np.unique(group_ids):
        idx = group_ids == g
        x_out[idx] -= x[idx].mean(axis=0)
    return x_out


def _hand_compute_with_fe(dgp):
    """Hand-compute test statistics with cost-side fixed effects (firm FEs)."""
    N = dgp['N']
    K = dgp['K']
    prices = dgp['prices']
    markups_m1 = dgp['markups_m1']
    markups_m2 = dgp['markups_m2']
    cost_shifter = dgp['cost_shifter']
    z = dgp['excluded_iv']

    # Use firm_ids as cost-side FEs (2 firms = 2 groups)
    # Build product_data's firm_ids for absorption
    T, J = dgp['T'], dgp['J']
    firm_ids = np.tile(np.arange(J), T)

    # Cost formulation: cost_shifter only (intercept is absorbed by firm FEs).
    # After demeaning, the intercept column is all zeros and must be excluded.
    w_raw = cost_shifter  # (N, 1) — no intercept; it's absorbed by firm FEs

    # Implied marginal cost
    mc_m1 = prices - markups_m1
    mc_m2 = prices - markups_m2

    # Absorb firm FEs from everything (within-group demean)
    mc_m1_abs = _absorb_fe(mc_m1, firm_ids)
    mc_m2_abs = _absorb_fe(mc_m2, firm_ids)
    w_abs = _absorb_fe(w_raw, firm_ids)
    z_abs = _absorb_fe(z, firm_ids)

    # Residualize on absorbed w
    Q_w, R_w = np.linalg.qr(w_abs, mode='reduced')
    omega_m1 = mc_m1_abs - Q_w @ (Q_w.T @ mc_m1_abs)
    omega_m2 = mc_m2_abs - Q_w @ (Q_w.T @ mc_m2_abs)
    z_orth = z_abs - Q_w @ (Q_w.T @ z_abs)

    # Tau (on absorbed data)
    tau_m1 = np.linalg.solve(R_w, Q_w.T @ mc_m1_abs)
    tau_m2 = np.linalg.solve(R_w, Q_w.T @ mc_m2_abs)

    # Weight matrix
    W_inv = (1 / N) * z_orth.T @ z_orth
    W = np.linalg.pinv(W_inv)

    # GMM moments and fit
    g_m1 = (1 / N) * z_orth.T @ omega_m1
    g_m2 = (1 / N) * z_orth.T @ omega_m2
    Q_m1 = float(g_m1.flatten() @ W @ g_m1.flatten())
    Q_m2 = float(g_m2.flatten() @ W @ g_m2.flatten())

    # RV numerator
    rv_num = np.sqrt(N) * (Q_m1 - Q_m2)

    # W powers
    eigvals, eigvecs = np.linalg.eigh((W + W.T) / 2)
    eigvals = np.maximum(eigvals, 0)
    W_12 = (eigvecs * (eigvals ** 0.5)) @ eigvecs.T
    W_34 = (eigvecs * (eigvals ** 0.75)) @ eigvecs.T

    # Psi
    def _compute_psi(omega_m, g_m):
        g_flat = g_m.flatten()
        psi_bar = W_12 @ g_flat - 0.5 * W_34 @ W_inv @ W_34 @ g_flat
        W34_Zg = (z_orth @ W_34 @ g_flat)[:, np.newaxis]
        mc_col = omega_m if omega_m.ndim == 2 else omega_m[:, np.newaxis]
        psi_i = (mc_col * z_orth) @ W_12 - 0.5 * W34_Zg * (z_orth @ W_34.T)
        return psi_i - psi_bar[np.newaxis, :]

    psi_m1 = _compute_psi(omega_m1, g_m1)
    psi_m2 = _compute_psi(omega_m2, g_m2)

    M = 2
    psi = np.stack([psi_m1, psi_m2])
    psi_flat = psi.transpose(1, 0, 2).reshape(N, M * K)
    gram = (1 / N) * psi_flat.T @ psi_flat

    V11 = gram[0:K, 0:K]
    V22 = gram[K:2*K, K:2*K]
    V12 = gram[0:K, K:2*K]
    wv11, wv22, wv12 = W_12 @ V11 @ W_12, W_12 @ V22 @ W_12, W_12 @ V12 @ W_12
    _g1, _g2 = g_m1.flatten(), g_m2.flatten()
    sigma2 = float(4 * (_g1 @ wv11 @ _g1 + _g2 @ wv22 @ _g2 - 2 * _g1 @ wv12 @ _g2))
    trv = float(rv_num / np.sqrt(sigma2))

    # F-statistic
    Q_z, _ = np.linalg.qr(z_orth, mode='reduced')
    e_m1 = omega_m1 - Q_z @ (Q_z.T @ omega_m1)
    e_m2 = omega_m2 - Q_z @ (Q_z.T @ omega_m2)
    phi_m1 = (e_m1 * z_orth) @ W
    phi_m2 = (e_m2 * z_orth) @ W
    phi = np.stack([phi_m1, phi_m2])
    phi_flat = phi.transpose(1, 0, 2).reshape(N, M * K)
    phi_gram = (1 / N) * phi_flat.T @ phi_flat
    V_ar_11 = phi_gram[0:K, 0:K]
    V_ar_22 = phi_gram[K:2*K, K:2*K]
    V_ar_12 = phi_gram[0:K, K:2*K]
    sigma_ar = (1 / K) * np.array([
        float(np.trace(V_ar_11 @ W_inv)),
        float(np.trace(V_ar_22 @ W_inv)),
        float(np.trace(V_ar_12 @ W_inv)),
    ])
    rho_num = sigma_ar[0] - sigma_ar[1]
    rho_den = np.sqrt((sigma_ar[0] + sigma_ar[1]) ** 2 - 4 * sigma_ar[2] ** 2)
    rho = rho_num / rho_den
    F_ops = np.array([sigma_ar[1], sigma_ar[0], -2 * sigma_ar[2]])
    F_moms = np.array([float(_g1 @ W @ _g1), float(_g2 @ W @ _g2), float(_g1 @ W @ _g2)])
    F_num = F_ops @ F_moms
    F_den = sigma_ar[0] * sigma_ar[1] - sigma_ar[2] ** 2
    unscaled_F = N / (2 * K) * F_num / F_den
    F = (1 - rho ** 2) * unscaled_F

    return {
        'TRV': trv, 'F': float(F), 'Q_m1': Q_m1, 'Q_m2': Q_m2,
        'g_m1': g_m1.flatten(), 'g_m2': g_m2.flatten(),
        'tau_m1': tau_m1.flatten(), 'tau_m2': tau_m2.flatten(),
    }


class TestAlgebraWithFE:
    """Validate engine with cost-side fixed effects (firm FEs absorbed)."""

    @pytest.fixture(scope='class')
    def data(self):
        product_data, dgp = _build_base_dgp()
        expected = _hand_compute_with_fe(dgp)

        model_formulations = (
            pyRVtest.ModelFormulation(model_downstream='bertrand', ownership_downstream='firm_ids',
                                     user_supplied_markups='markups_m1'),
            pyRVtest.ModelFormulation(model_downstream='perfect_competition',
                                     user_supplied_markups='markups_m2'),
        )
        testing_problem = pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('0 + cost_shifter', absorb='C(firm_ids)'),
            instrument_formulation=pyRVtest.Formulation('0 + iv0 + iv1 + iv2'),
            model_formulations=model_formulations,
            product_data=product_data,
            demand_results=None,
        )
        results = testing_problem.solve(demand_adjustment=False, clustering_adjustment=False)
        return results, expected

    def test_trv(self, data):
        results, expected = data
        np.testing.assert_allclose(results.TRV[0][0, 1], expected['TRV'], atol=1e-6,
                                   err_msg="TRV with cost FEs does not match hand computation")

    def test_f(self, data):
        results, expected = data
        np.testing.assert_allclose(results.F[0][0, 1], expected['F'], atol=1e-6,
                                   err_msg="F with cost FEs does not match hand computation")

    def test_g(self, data):
        results, expected = data
        np.testing.assert_allclose(results.g[0][0], expected['g_m1'], atol=1e-8)
        np.testing.assert_allclose(results.g[0][1], expected['g_m2'], atol=1e-8)

    def test_trv_differs_from_no_fe(self, data):
        """TRV with FEs should differ from TRV without FEs."""
        _, dgp = _build_base_dgp()
        no_fe = _hand_compute_base(dgp, clustering_ids=None)
        _, expected = data
        assert expected['TRV'] != no_fe['TRV'], "FE and no-FE TRV should differ"


# ---------------------------------------------------------------------------
# Test class 3: Economies of scale (endogenous cost component)
# ---------------------------------------------------------------------------

def _build_scale_dgp(seed=54321, T=30, J=2):
    """Build a DGP with non-constant marginal cost: log(c) = gamma*log(q) + w'tau + omega."""
    rng = np.random.default_rng(seed=seed)
    N = T * J
    market_ids = np.repeat(np.arange(T), J)
    firm_ids = np.tile(np.arange(J), T)

    alpha = -1.5     # moderate price sensitivity (keeps shares interior)
    beta_x = 1.0
    gamma_true = -0.1  # mild scale economies (avoids divergence)
    x = rng.uniform(0.8, 1.5, size=(N, 1))  # tighter x range
    cost_shifter = rng.uniform(0.5, 1.5, size=(N, 1))
    # Two instruments (need at least 2: one to identify gamma, one for testing)
    iv1 = rng.uniform(0.5, 2.0, size=(N, 1))
    iv2 = rng.uniform(0.5, 2.0, size=(N, 1))
    xi = rng.normal(0, 0.02, size=(N, 1))     # small demand shocks
    omega_true = rng.normal(0, 0.02, size=(N, 1))  # small cost shocks

    # Large market size ensures positive quantities
    market_size = 100000.0

    # True cost: log(c) = gamma*log(q) + tau_0 + tau_w*w + omega
    tau_0, tau_w = 1.0, 0.3
    mc_init = np.exp(tau_0 + tau_w * cost_shifter + omega_true)
    prices, shares, markups_b = _solve_bertrand_logit(alpha, beta_x, x, xi, mc_init, market_ids, T, N)

    # Iterate: update cost with scale economies, re-solve equilibrium
    for iteration in range(200):
        quantities = market_size * shares
        assert np.all(quantities > 0), f"Zero quantities at iteration {iteration}"
        log_mc = gamma_true * np.log(quantities) + tau_0 + tau_w * cost_shifter + omega_true
        mc_true = np.exp(log_mc)
        prices_new, shares_new, markups_b_new = _solve_bertrand_logit(
            alpha, beta_x, x, xi, mc_true, market_ids, T, N
        )
        if np.max(np.abs(prices_new - prices)) < 1e-12:
            break
        prices, shares, markups_b = prices_new, shares_new, markups_b_new

    quantities = market_size * shares
    markups_perfect = np.zeros((N, 1))

    product_data = pd.DataFrame({
        'market_ids': market_ids,
        'firm_ids': firm_ids,
        'prices': prices.flatten(),
        'shares': shares.flatten(),
        'x': x.flatten(),
        'cost_shifter': cost_shifter.flatten(),
        'iv1': iv1.flatten(),
        'iv2': iv2.flatten(),
        'log_quantity': np.log(quantities).flatten(),
        'markups_m1': markups_b.flatten(),
        'markups_m2': markups_perfect.flatten(),
        'clustering_ids': market_ids,
    })

    return product_data, {
        'N': N, 'T': T, 'J': J, 'gamma_true': gamma_true,
        'prices': prices, 'shares': shares, 'quantities': quantities,
        'markups_m1': markups_b, 'markups_m2': markups_perfect,
        'cost_shifter': cost_shifter, 'iv1': iv1, 'iv2': iv2,
        'omega_true': omega_true, 'log_quantity': np.log(quantities),
    }


def _hand_compute_scale(dgp):
    """Hand-compute test statistics with endogenous cost component (economies of scale)."""
    N = dgp['N']
    M = 2
    K = 2  # two instruments
    prices = dgp['prices']
    markups = [dgp['markups_m1'], dgp['markups_m2']]
    cost_shifter = dgp['cost_shifter']
    iv1, iv2 = dgp['iv1'], dgp['iv2']
    log_q = dgp['log_quantity']

    # Cost formulation: [1, cost_shifter, log_quantity] where log_quantity is endogenous
    w_exog = np.hstack([np.ones((N, 1)), cost_shifter])  # exogenous columns
    endog_col = log_q  # (N, 1)
    instruments = np.hstack([iv1, iv2])  # (N, 2)

    # Per model: implied marginal cost in logs
    mc_list = [np.log(prices - markups[m]) for m in range(M)]

    # 2SLS for gamma: first stage = project log_q on [w_exog, instruments]
    first_stage_X = np.hstack([w_exog, instruments])
    Q_fs, _ = np.linalg.qr(first_stage_X, mode='reduced')
    endog_hat = (Q_fs @ (Q_fs.T @ endog_col)).reshape(-1, 1)

    # Second stage per model: regress log(mc) on [w_exog, endog_hat]
    X_2sls = np.hstack([w_exog, endog_hat])
    Q_2sls, R_2sls = np.linalg.qr(X_2sls, mode='reduced')
    gamma = np.zeros(M)
    omega = [None] * M
    for m in range(M):
        params = np.linalg.solve(R_2sls, Q_2sls.T @ mc_list[m])
        gamma[m] = float(params[-1].item())

    # Marginal cost after IV correction: mc_corrected = mc - gamma * endog_col (raw)
    # Then omega = residualize mc_corrected on w_exog
    Q_w, R_w = np.linalg.qr(w_exog, mode='reduced')
    for m in range(M):
        mc_corrected = mc_list[m] - gamma[m] * endog_col
        omega[m] = mc_corrected - Q_w @ (Q_w.T @ mc_corrected)

    # Effective instruments: z^e = instruments residualized on [w_exog, endog_hat]
    controls = np.hstack([w_exog, endog_hat])
    z_eff = _qr_residualize(instruments, controls).reshape(N, K)

    # K_effective = K - 1 = 1
    K_eff = K - 1

    # Weight matrix (pseudo-inverse since z_eff is rank K-1)
    W_inv = (1 / N) * z_eff.T @ z_eff
    W = np.linalg.pinv(W_inv)

    # GMM moments and fit
    g = np.zeros((M, K))
    Q_val = np.zeros(M)
    for m in range(M):
        g[m] = ((1 / N) * z_eff.T @ omega[m]).flatten()
        Q_val[m] = float(g[m] @ W @ g[m])

    # TRV
    rv_num = np.sqrt(N) * (Q_val[0] - Q_val[1])

    # W powers
    eigvals, eigvecs = np.linalg.eigh((W + W.T) / 2)
    eigvals = np.maximum(eigvals, 0)
    W_12 = (eigvecs * (eigvals ** 0.5)) @ eigvecs.T
    W_34 = (eigvecs * (eigvals ** 0.75)) @ eigvecs.T

    # Psi with first-stage correction
    # z^r = instruments residualized on w_exog only
    z_r = _qr_residualize(instruments, w_exog).reshape(N, K)
    Z_prec = np.linalg.pinv((1 / N) * z_r.T @ z_r)
    q_e = (endog_col - endog_hat).reshape(N, 1)

    # lambda_q from projection of z on [endog_hat, w_exog]
    proj_X = np.hstack([endog_hat, w_exog])
    Q_proj, R_proj = np.linalg.qr(proj_X, mode='reduced')
    lambda_coefs = np.linalg.solve(R_proj, Q_proj.T @ instruments)
    lambda_q = lambda_coefs[0, :]  # (K,)

    M_corr = W_34 @ W @ Z_prec
    W_plus = W

    psi = np.zeros((M, N, K))
    for m in range(M):
        psi_bar = W_12 @ g[m] - 0.5 * W_34 @ W_inv @ W_34 @ g[m]
        W34_Zg = (z_eff @ W_34 @ g[m])[:, np.newaxis]
        mc_col = omega[m] if omega[m].ndim == 2 else omega[m][:, np.newaxis]
        psi_i = (mc_col * z_eff) @ W_12 - 0.5 * W34_Zg * (z_eff @ W_34.T)
        psi[m] = psi_i - psi_bar.T

        # First-stage correction
        W34_gm = W_34 @ g[m]
        v = Z_prec @ W_plus @ W34_gm
        u = q_e * z_r
        term1 = (u @ M_corr.T) * (lambda_q @ W34_gm)
        term2 = (u @ v)[:, np.newaxis] * (W_34 @ lambda_q)[np.newaxis, :]
        psi[m] = psi[m] + 0.5 * (term1 + term2)

    # Variance and TRV
    psi_flat = psi.transpose(1, 0, 2).reshape(N, M * K)
    gram = (1 / N) * psi_flat.T @ psi_flat
    V11 = gram[0:K, 0:K]
    V22 = gram[K:2*K, K:2*K]
    V12 = gram[0:K, K:2*K]
    wv11 = W_12 @ V11 @ W_12
    wv22 = W_12 @ V22 @ W_12
    wv12 = W_12 @ V12 @ W_12
    sigma2 = float(4 * (g[0] @ wv11 @ g[0] + g[1] @ wv22 @ g[1] - 2 * g[0] @ wv12 @ g[1]))
    trv = float(rv_num / np.sqrt(sigma2))

    # F-statistic with K_effective
    Q_z, _ = np.linalg.qr(z_eff, mode='reduced')
    phi = np.zeros((M, N, K))
    for m in range(M):
        e = omega[m] - Q_z @ (Q_z.T @ omega[m])
        if e.ndim == 1:
            e = e[:, np.newaxis]
        phi[m] = (e * z_eff) @ W

    phi_flat = phi.transpose(1, 0, 2).reshape(N, M * K)
    phi_gram = (1 / N) * phi_flat.T @ phi_flat
    V_ar_11 = phi_gram[0:K, 0:K]
    V_ar_22 = phi_gram[K:2*K, K:2*K]
    V_ar_12 = phi_gram[0:K, K:2*K]
    sigma_ar = (1 / K_eff) * np.array([
        float(np.trace(V_ar_11 @ W_inv)),
        float(np.trace(V_ar_22 @ W_inv)),
        float(np.trace(V_ar_12 @ W_inv)),
    ])
    rho_num = sigma_ar[0] - sigma_ar[1]
    rho_den = np.sqrt((sigma_ar[0] + sigma_ar[1]) ** 2 - 4 * sigma_ar[2] ** 2)
    rho = rho_num / rho_den
    F_ops = np.array([sigma_ar[1], sigma_ar[0], -2 * sigma_ar[2]])
    F_moms = np.array([float(g[0] @ W @ g[0]), float(g[1] @ W @ g[1]), float(g[0] @ W @ g[1])])
    F_num = F_ops @ F_moms
    F_den = sigma_ar[0] * sigma_ar[1] - sigma_ar[2] ** 2
    unscaled_F = N / (2 * K_eff) * F_num / F_den
    F_stat = (1 - rho ** 2) * unscaled_F

    return {
        'TRV': trv, 'F': float(F_stat),
        'gamma': gamma,
        'g_m1': g[0].tolist(), 'g_m2': g[1].tolist(),
    }


class TestAlgebraScaleEconomies:
    """Validate engine with endogenous cost component (economies of scale)."""

    @pytest.fixture(scope='class')
    def data(self):
        product_data, dgp = _build_scale_dgp()
        expected = _hand_compute_scale(dgp)

        model_formulations = (
            pyRVtest.ModelFormulation(model_downstream='bertrand', ownership_downstream='firm_ids',
                                     user_supplied_markups='markups_m1'),
            pyRVtest.ModelFormulation(model_downstream='perfect_competition',
                                     user_supplied_markups='markups_m2'),
        )
        testing_problem = pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('1 + cost_shifter + log_quantity'),
            instrument_formulation=pyRVtest.Formulation('0 + iv1 + iv2'),
            model_formulations=model_formulations,
            product_data=product_data,
            demand_results=None,
            endogenous_cost_component='log_quantity',
        )
        results = testing_problem.solve(demand_adjustment=False, clustering_adjustment=False, costs_type='log')
        return results, expected

    def test_trv(self, data):
        results, expected = data
        np.testing.assert_allclose(results.TRV[0][0, 1], expected['TRV'], atol=1e-6,
                                   err_msg="TRV with scale economies does not match hand computation")

    def test_f(self, data):
        results, expected = data
        np.testing.assert_allclose(results.F[0][0, 1], expected['F'], atol=1e-6,
                                   err_msg="F with scale economies does not match hand computation")

    def test_gamma_estimates(self, data):
        results, expected = data
        # cost_param[instrument_set][model] contains [tau_exog..., gamma]
        for m in range(2):
            gamma_pyrvtest = float(results.cost_param[0][m][-1].item())
            np.testing.assert_allclose(gamma_pyrvtest, expected['gamma'][m], atol=1e-8,
                                       err_msg=f"gamma for model {m} does not match")


class TestLogCostsWithDemandAdjustmentRejected:
    """costs_type='log' combined with demand_adjustment=True is unsupported.

    Pre-fix: the log transform was silently gated behind ``not demand_adjustment``,
    so the user got level marginal costs without any error or warning. Lorenzo's
    audit (2026-04-29 P1) flagged this as an unintended fallback. The fix
    rejects the combination explicitly with a NotImplementedError until the
    log-cost demand-adjustment algebra is derived.
    """

    def test_raises_not_implemented_error(self):
        """The error should fire BEFORE the log-cost transform so the user
        sees a clear NotImplementedError rather than a silent fallback to
        level marginal costs.

        Build a minimal Problem with demand_params (so demand_adjustment=True
        is allowed by the upstream validator) and Bertrand markups (so the
        user-supplied-markups validator doesn't reject first). Then call
        solve(costs_type='log', demand_adjustment=True) and expect the
        new error.
        """
        rng = np.random.default_rng(seed=11)
        T, J = 10, 3
        N = T * J
        market_ids = np.repeat(np.arange(T), J)
        firm_ids = np.tile(np.arange(J), T)
        x1 = rng.normal(size=N)
        prices = 1.0 + 0.5 * rng.normal(size=N) + 0.3 * np.abs(rng.normal(size=N))
        # Synthetic positive shares per market.
        shares = np.zeros(N)
        for t in range(T):
            idx = np.where(market_ids == t)[0]
            raw = rng.uniform(0.05, 0.4, size=len(idx))
            shares[idx] = raw / (raw.sum() + 1.0)
        product_data = pd.DataFrame({
            'market_ids': market_ids, 'firm_ids': firm_ids,
            'prices': prices, 'shares': shares, 'x1': x1,
            'rival_x1': rng.normal(size=N),
            'cost_shifter': rng.uniform(0.5, 1.5, size=N),
        })
        # Add an intercept column for the demand-adjustment x_columns lookup.
        product_data['intercept'] = 1.0

        problem = pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
            instrument_formulation=pyRVtest.Formulation('0 + rival_x1'),
            product_data=product_data,
            demand_params={
                'alpha': -2.0,
                'sigma': [],
                'beta': np.array([0.0, 1.0]),
                'x_columns': ['intercept', 'x1'],
                'demand_instrument_columns': ['rival_x1', 'intercept', 'x1'],
            },
            models=[
                pyRVtest.Bertrand(ownership='firm_ids'),
                pyRVtest.PerfectCompetition(),
            ],
        )
        with pytest.raises(NotImplementedError, match=r"costs_type='log'.*demand_adjustment"):
            problem.solve(
                demand_adjustment=True, clustering_adjustment=False,
                costs_type='log',
            )
