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


class TestAbsorbOnlyCostFormulation:
    """Absorb-only cost spec (all shifters absorbed, zero-column w) is valid.

    By Frisch-Waugh-Lovell, absorbing ``C(firm_ids)`` is equivalent to entering
    the firm dummies as explicit cost shifters, so the two specifications must
    produce identical test statistics. This exercises the relaxation that lets
    pyRVtest construct ``Formulation('0', absorb='C(firm_ids)')`` (PyBLP rejects
    it as "formula has no terms").
    """

    @pytest.fixture(scope='class')
    def results_pair(self):
        product_data, _ = _build_base_dgp()
        model_formulations = (
            pyRVtest.ModelFormulation(model_downstream='bertrand', ownership_downstream='firm_ids',
                                      user_supplied_markups='markups_m1'),
            pyRVtest.ModelFormulation(model_downstream='perfect_competition',
                                      user_supplied_markups='markups_m2'),
        )
        instrument_formulation = pyRVtest.Formulation('0 + iv0 + iv1 + iv2')

        def _solve(cost_formulation):
            problem = pyRVtest.Problem(
                cost_formulation=cost_formulation,
                instrument_formulation=instrument_formulation,
                model_formulations=model_formulations,
                product_data=product_data,
                demand_results=None,
            )
            results = problem.solve(demand_adjustment=False, clustering_adjustment=False)
            return problem, results

        explicit = _solve(pyRVtest.Formulation('0 + C(firm_ids)'))
        absorbed = _solve(pyRVtest.Formulation('0', absorb='C(firm_ids)'))
        return explicit, absorbed

    def test_absorb_only_constructs(self, results_pair):
        """The absorb-only Problem builds with a zero-column w and EC == 1."""
        (_explicit_problem, _), (absorbed_problem, _) = results_pair
        assert absorbed_problem.products.w.shape[1] == 0
        assert absorbed_problem.EC == 1

    def test_trv_matches_explicit_dummies(self, results_pair):
        (_, explicit_results), (_, absorbed_results) = results_pair
        np.testing.assert_allclose(
            absorbed_results.TRV[0], explicit_results.TRV[0], atol=1e-8,
            err_msg="absorb-only TRV must match explicit firm dummies (FWL)",
        )

    def test_f_matches_explicit_dummies(self, results_pair):
        (_, explicit_results), (_, absorbed_results) = results_pair
        np.testing.assert_allclose(
            absorbed_results.F[0], explicit_results.F[0], atol=1e-8,
            err_msg="absorb-only F must match explicit firm dummies (FWL)",
        )

    def test_mcs_matches_explicit_dummies(self, results_pair):
        (_, explicit_results), (_, absorbed_results) = results_pair
        np.testing.assert_allclose(
            absorbed_results.MCS_pvalues[0], explicit_results.MCS_pvalues[0], atol=1e-8,
            err_msg="absorb-only MCS p-values must match explicit firm dummies (FWL)",
        )


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
        # endogenous_cost_coefficient is shape (L, M); element [l, m] is gamma for instrument l, model m
        for m in range(2):
            gamma_pyrvtest = float(results.endogenous_cost_coefficient[0, m].item())
            np.testing.assert_allclose(gamma_pyrvtest, expected['gamma'][m], atol=1e-8,
                                       err_msg=f"gamma for model {m} does not match")


class TestKInstValidationWithEndogenousCost:
    """K_inst > K_endog gate when endogenous_cost_component is set.

    The IV correction absorbs K_endog instruments to identify cost
    parameters, leaving K_inst - K_endog testing dimensions. Without at
    least one testing dimension the test is mechanically degenerate
    (DMQSS 2026, Remark 1) and the F-stat denominator collapses to zero.
    The validation in ``_validate_solve_args`` rejects this case with a
    clear ``ValueError`` before ``test_engine`` would otherwise crash
    with ``ZeroDivisionError``.
    """

    @pytest.fixture(scope='class')
    def data_with_log_q(self):
        data = pyRVtest.data.load_example()
        data['log_q'] = np.log(np.maximum(data['shares'], 1e-8))
        return data

    def test_single_iv_set_with_one_instrument_raises(self, data_with_log_q):
        """K_inst = 1, K_endog = 1 -> hard error (would otherwise ZeroDivisionError)."""
        problem = pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('1 + z1 + log_q'),
            instrument_formulation=pyRVtest.Formulation('0 + rival_z1'),
            models=[
                pyRVtest.Bertrand(ownership='firm_ids'),
                pyRVtest.PerfectCompetition(),
            ],
            product_data=data_with_log_q,
            demand_params={
                'estimate': 'logit',
                'formulation_X': pyRVtest.Formulation('1 + x1'),
                'formulation_Z': pyRVtest.Formulation('0 + z1'),
            },
            endogenous_cost_component='log_q',
        )
        with pytest.raises(ValueError) as exc_info:
            problem.solve(demand_adjustment=False)
        msg = str(exc_info.value)
        assert 'K_endog + 1' in msg
        assert 'set 0 (K_inst = 1)' in msg
        assert 'DMQSS' in msg

    def test_two_iv_sets_only_one_bad_lists_only_bad(self, data_with_log_q):
        """Multi-set listing: only the bad set is named in the error."""
        problem = pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('1 + z1 + log_q'),
            instrument_formulation=[
                pyRVtest.Formulation('0 + rival_z1 + rival_z2'),  # K=2, fine
                pyRVtest.Formulation('0 + rival_z1'),               # K=1, bad
            ],
            models=[
                pyRVtest.Bertrand(ownership='firm_ids'),
                pyRVtest.PerfectCompetition(),
            ],
            product_data=data_with_log_q,
            demand_params={
                'estimate': 'logit',
                'formulation_X': pyRVtest.Formulation('1 + x1'),
                'formulation_Z': pyRVtest.Formulation('0 + z1'),
            },
            endogenous_cost_component='log_q',
        )
        with pytest.raises(ValueError) as exc_info:
            problem.solve(demand_adjustment=False)
        msg = str(exc_info.value)
        assert 'set 1 (K_inst = 1)' in msg
        assert 'set 0' not in msg  # the good set should NOT be listed

    def test_boundary_kinst_equals_kendog_plus_one_passes(self, data_with_log_q):
        """K_inst = K_endog + 1 = 2: minimum testing dimensions; test runs.

        F may collapse near zero (per existing behavior), but solve()
        succeeds without error. The post-solve F-stat reliability output
        is the right diagnostic for this regime, not a construction-time
        warning.
        """
        problem = pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('1 + z1 + log_q'),
            instrument_formulation=pyRVtest.Formulation('0 + rival_z1 + rival_z2'),
            models=[
                pyRVtest.Bertrand(ownership='firm_ids'),
                pyRVtest.PerfectCompetition(),
            ],
            product_data=data_with_log_q,
            demand_params={
                'estimate': 'logit',
                'formulation_X': pyRVtest.Formulation('1 + x1'),
                'formulation_Z': pyRVtest.Formulation('0 + z1'),
            },
            endogenous_cost_component='log_q',
        )
        # Should not raise.
        results = problem.solve(demand_adjustment=False)
        # Sanity: F is well-defined (a finite number).
        assert np.isfinite(results.F[0][0, 1])

    def test_no_endogenous_cost_no_validation(self, data_with_log_q):
        """K_inst = 1 with no endogenous_cost_component: no error.

        The K_inst > K_endog gate only applies when an endogenous cost
        component is present (otherwise K_endog = 0).
        """
        problem = pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('1 + z1 + z2'),
            instrument_formulation=pyRVtest.Formulation('0 + rival_z1'),
            models=[
                pyRVtest.Bertrand(ownership='firm_ids'),
                pyRVtest.PerfectCompetition(),
            ],
            product_data=data_with_log_q,
            demand_params={
                'estimate': 'logit',
                'formulation_X': pyRVtest.Formulation('1 + x1'),
                'formulation_Z': pyRVtest.Formulation('0 + z1'),
            },
        )
        results = problem.solve(demand_adjustment=False)
        assert np.isfinite(results.F[0][0, 1])


class TestMultiColumnEndogenousCostRoundTrip:
    """Multi-column endogenous_cost_component API: K_endog >= 1.

    The API accepts ``Optional[Union[str, Sequence[str]]]``. Single-string
    and length-one-list specifications must produce bit-identical output
    (no regression). Multi-column specifications must run end-to-end on a
    fixture with K_endog distinct endogenous columns and at least
    K_endog + 1 instruments per set.
    """

    @pytest.fixture(scope='class')
    def scale_data(self):
        product_data, dgp = _build_scale_dgp()
        return product_data, dgp

    def _build_problem(self, product_data, *, endogenous_cost_component):
        """Build a fresh Problem with non-shared Formulation objects.

        Reusing Formulation objects across two Problem constructions can
        leak _build_matrix state (the in-place w-design-matrix expansion);
        building fresh per call keeps the tests deterministic.
        """
        return pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('1 + cost_shifter + log_quantity'),
            instrument_formulation=pyRVtest.Formulation('0 + iv1 + iv2'),
            model_formulations=(
                pyRVtest.ModelFormulation(
                    model_downstream='bertrand',
                    ownership_downstream='firm_ids',
                    user_supplied_markups='markups_m1',
                ),
                pyRVtest.ModelFormulation(
                    model_downstream='perfect_competition',
                    user_supplied_markups='markups_m2',
                ),
            ),
            product_data=product_data,
            demand_results=None,
            endogenous_cost_component=endogenous_cost_component,
        )

    def test_string_and_list_of_one_produce_identical_results(self, scale_data):
        """endogenous_cost_component='log_quantity' and ['log_quantity']
        must produce bit-identical TRV / F / gamma.
        """
        product_data, _ = scale_data

        problem_str = self._build_problem(
            product_data, endogenous_cost_component='log_quantity',
        )
        problem_list = self._build_problem(
            product_data, endogenous_cost_component=['log_quantity'],
        )

        # Internal normalization: both should yield the same tuple of cost columns.
        assert problem_str._endogenous_cost_columns == ('log_quantity',)
        assert problem_list._endogenous_cost_columns == ('log_quantity',)

        results_str = problem_str.solve(
            demand_adjustment=False, clustering_adjustment=False, costs_type='log',
        )
        results_list = problem_list.solve(
            demand_adjustment=False, clustering_adjustment=False, costs_type='log',
        )

        np.testing.assert_array_equal(results_str.TRV[0], results_list.TRV[0])
        np.testing.assert_array_equal(results_str.F[0], results_list.F[0])
        np.testing.assert_array_equal(
            np.asarray(results_str.endogenous_cost_coefficient),
            np.asarray(results_list.endogenous_cost_coefficient),
        )

    def test_multi_column_solve_runs_end_to_end(self, scale_data):
        """K_endog = 2 with 3 instruments runs to completion.

        Use ``log_quantity`` and ``log_quantity_sq = log_quantity ** 2``
        as two endogenous columns. Three instruments (iv1, iv2, and
        ``iv1 * iv2`` as a manufactured-distinct third) satisfy
        ``K_inst > K_endog``. The 2SLS first stage is well-conditioned
        (the synthetic IVs are independent uniforms whose products give
        a mild nonlinearity vs. the originals).
        """
        product_data, _ = scale_data
        product_data = product_data.copy()
        product_data['log_quantity_sq'] = product_data['log_quantity'] ** 2
        product_data['iv12'] = product_data['iv1'] * product_data['iv2']

        problem = pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation(
                '1 + cost_shifter + log_quantity + log_quantity_sq'
            ),
            instrument_formulation=pyRVtest.Formulation('0 + iv1 + iv2 + iv12'),
            model_formulations=(
                pyRVtest.ModelFormulation(
                    model_downstream='bertrand',
                    ownership_downstream='firm_ids',
                    user_supplied_markups='markups_m1',
                ),
                pyRVtest.ModelFormulation(
                    model_downstream='perfect_competition',
                    user_supplied_markups='markups_m2',
                ),
            ),
            product_data=product_data,
            demand_results=None,
            endogenous_cost_component=['log_quantity', 'log_quantity_sq'],
        )

        assert problem._endogenous_cost_columns == ('log_quantity', 'log_quantity_sq')

        results = problem.solve(
            demand_adjustment=False, clustering_adjustment=False, costs_type='log',
        )

        assert np.isfinite(results.TRV[0][0, 1])
        assert np.isfinite(results.F[0][0, 1])

        # K_endog = 2 absorbs 2 of 3 instruments, leaving K_eff = 1 testing dimension.
        # endogenous_cost_coefficient is shape (L, M); each entry is the LAST
        # cost-param (the gamma block) — for K_endog = 2, this is a length-2 vector.
        # We just check it's finite, since this is a smoke test, not a hand-derived
        # correctness test (those land in a separate quadratic-cost / scale+scope
        # fixture with closed-form expected values).
        for m in range(2):
            cell = results.endogenous_cost_coefficient[0, m]
            arr = np.asarray(cell).ravel()
            assert arr.size >= 2, f"Expected K_endog >= 2 gamma coefficients; got {arr.size}"
            assert np.all(np.isfinite(arr))

    def test_empty_list_rejected(self, scale_data):
        product_data, _ = scale_data
        with pytest.raises(ValueError, match='non-empty'):
            self._build_problem(product_data, endogenous_cost_component=[])

    def test_duplicate_columns_rejected(self, scale_data):
        product_data, _ = scale_data
        with pytest.raises(ValueError, match='distinct'):
            self._build_problem(
                product_data,
                endogenous_cost_component=['log_quantity', 'log_quantity'],
            )

    def test_non_string_entry_rejected(self, scale_data):
        product_data, _ = scale_data
        with pytest.raises(TypeError, match='column-name string'):
            self._build_problem(
                product_data,
                endogenous_cost_component=['log_quantity', 42],
            )

    def test_missing_column_in_cost_formulation_rejected(self, scale_data):
        product_data, _ = scale_data
        with pytest.raises(ValueError, match='no matching term'):
            pyRVtest.Problem(
                cost_formulation=pyRVtest.Formulation('1 + cost_shifter + log_quantity'),
                # 'not_a_column' isn't in cost_formulation
                instrument_formulation=pyRVtest.Formulation('0 + iv1 + iv2'),
                model_formulations=(
                    pyRVtest.ModelFormulation(
                        model_downstream='bertrand',
                        ownership_downstream='firm_ids',
                        user_supplied_markups='markups_m1',
                    ),
                    pyRVtest.ModelFormulation(
                        model_downstream='perfect_competition',
                        user_supplied_markups='markups_m2',
                    ),
                ),
                product_data=product_data,
                demand_results=None,
                endogenous_cost_component=['log_quantity', 'not_a_column'],
            )
