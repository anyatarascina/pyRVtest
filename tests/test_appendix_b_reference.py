"""Pin the package against the revised Duarte et al. (2026), Appendix B.

The revised Duarte et al. (2026), Appendix B writes the RV influence function
for a cost specification with ``d_q`` endogenous regressors (instrumented by
``d_z`` excluded instruments) as an ``r``-vector, ``r = d_z - d_q``::

    psi_m,i = W^{1/2}( z^e_i omega_mi - g_m - S_e Lambda_q q^e_i z^{r'}_i Z g^0_m )
              + (g_m' (x) I_r) A_W vec( z^e_i z^{e'}_i - W^{-1}
                                         - S_e Sigma^0_e Z z^r_i q^{e'}_i Lambda_q' S_e'
                                         - S_e Lambda_q q^e_i z^{r'}_i Z Sigma^0_e S_e' )

with ``S_e`` an ``r x d_z`` selection of residualized instruments with
nonsingular covariance, ``W = (E z^e z^{e'})^{-1}`` and ``A_W`` the exact
derivative of ``W^{1/2}`` with respect to ``W^{-1}`` (a Sylvester solution).
The package computes the RV denominator from the scalar score
``phi_mi = 2 v_mi omega_mi - v_mi^2 - Q_m`` (``pyRVtest/solve/test_engine.py``).
These tests establish, on the endogenous-cost path, that

1. ``A_W`` is the derivative of the matrix square root;
2. ``2 (W^{1/2} g_m)' psi_m,i == phi_mi`` observation by observation, so the
   package's ``RV_denominator`` / ``TRV`` equal the vector formula;
3. the two q_tilde terms change the vector covariance ``V^RV`` but cancel
   exactly in the scalar contraction;
4. the choice of ``S_e`` is immaterial and the package's ``d_z``-dimensional
   pseudo-inverse form equals the full-rank ``r``-form (Q, sigma_RV, F, rho);
5. with two endogenous cost regressors (``d_q = 2``) the package uses
   ``r = d_z - 2`` and its F equals eq. (F) of Duarte et al. (2026), Appendix B;
6. the same holds with absorbed cost-side fixed effects;
7. the effective-rank audit warns on redundant instruments.

The reference formula is hand-coded here (it lives in no package module);
the same code, with a Monte Carlo, is in ``notes/mc_rv_variance_appendix_b.py``.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest
import scipy.linalg

import pyRVtest
from pyRVtest.solve.test_engine import _audit_effective_rank


# ---------------------------------------------------------------------------
# Reference implementation of the revised Duarte et al. (2026), Appendix B
# ---------------------------------------------------------------------------

def _sym_power(W: np.ndarray, power: float) -> np.ndarray:
    vals, vecs = np.linalg.eigh((W + W.T) / 2)
    return (vecs * vals ** power) @ vecs.T


def _residualize(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    if X.shape[1] == 0:
        return Y
    Q, _ = np.linalg.qr(X, mode='reduced')
    return Y - Q @ (Q.T @ Y)


def _ols(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, Y, rcond=None)[0]


def _sqrt_derivative(W: np.ndarray) -> np.ndarray:
    """A_W = -(I (x) W^{-1/2} + W^{-1/2} (x) I)^{-1} (W^{1/2} (x) W^{1/2})."""
    r = W.shape[0]
    W12, Wm12, I = _sym_power(W, 0.5), _sym_power(W, -0.5), np.eye(r)
    return -np.linalg.solve(np.kron(I, Wm12) + np.kron(Wm12, I), np.kron(W12, W12))


class _Reference:
    """All Duarte et al. (2026), Appendix B objects for one instrument set."""

    def __init__(self, z: np.ndarray, w: np.ndarray, q: np.ndarray, costs: Sequence[np.ndarray],
                 sel: Optional[Sequence[int]] = None):
        N, d_z = z.shape
        d_q = q.shape[1]
        self.N, self.d_z, self.d_q, self.r = N, d_z, d_q, d_z - d_q
        zw = np.hstack([z, w])
        q_tilde = zw @ _ols(q, zw)
        self.q_e = q - q_tilde
        qw_hat = np.hstack([q_tilde, w])
        lam = _ols(z, qw_hat)
        self.Lambda_q = lam[:d_q, :].T                                # (d_z, d_q)
        self.z_e0 = z - qw_hat @ lam                                  # (N, d_z), rank r
        self.z_r = _residualize(z, w)
        self.Z_hat = np.linalg.inv(self.z_r.T @ self.z_r / N)
        self.Sigma_e0 = self.z_e0.T @ self.z_e0 / N
        # 2SLS residuals per model
        qw = np.hstack([q, w])
        self.omega = [
            (c - qw @ np.linalg.solve(qw_hat.T @ qw, qw_hat.T @ c)).reshape(-1) for c in costs
        ]
        # full-rank selection S_e
        if sel is None:
            _, _, piv = scipy.linalg.qr(self.z_e0, mode='economic', pivoting=True)
            sel = np.sort(piv[:self.r])
        self.sel = np.asarray(sel)
        self.z_e = self.z_e0[:, self.sel]
        self.W = np.linalg.inv(self.z_e.T @ self.z_e / N)
        self.g = [self.z_e.T @ o / N for o in self.omega]
        self.g0 = [self.z_e0.T @ o / N for o in self.omega]
        self.Q = [float(g @ self.W @ g) for g in self.g]

    def psi(self, m: int, include_qtilde: bool = True) -> np.ndarray:
        N, r, W = self.N, self.r, self.W
        z_e, omega, g, g0 = self.z_e, self.omega[m], self.g[m], self.g0[m]
        line1 = z_e * omega[:, None] - g[None, :]
        D = z_e[:, :, None] * z_e[:, None, :] - np.linalg.inv(W)[None, :, :]
        if include_qtilde:
            t = (self.q_e @ self.Lambda_q.T)[:, self.sel]           # S_e Lambda_q q^e_i
            c = self.z_r @ (self.Z_hat @ g0)                         # z^{r'}_i Z g^0
            line1 = line1 - t * c[:, None]
            u = (self.z_r @ self.Z_hat @ self.Sigma_e0)[:, self.sel]  # S_e Sigma^0 Z z^r_i
            D = D - u[:, :, None] * t[:, None, :] - t[:, :, None] * u[:, None, :]
        line1 = line1 @ _sym_power(W, 0.5)
        vecX = D.transpose(0, 2, 1).reshape(N, r * r) @ _sqrt_derivative(W).T
        X = vecX.reshape(N, r, r).transpose(0, 2, 1)
        return line1 + np.einsum('nij,j->ni', X, g)

    def sigma2_rv(self, include_qtilde: bool = True) -> Tuple[float, np.ndarray]:
        p1, p2 = self.psi(0, include_qtilde), self.psi(1, include_qtilde)
        W12 = _sym_power(self.W, 0.5)
        V11, V22, V12 = p1.T @ p1 / self.N, p2.T @ p2 / self.N, p1.T @ p2 / self.N
        a1, a2 = W12 @ self.g[0], W12 @ self.g[1]
        return 4.0 * float(a1 @ V11 @ a1 + a2 @ V22 @ a2 - 2 * a1 @ V12 @ a2), V11

    def scalar_scores(self) -> List[np.ndarray]:
        out = []
        for m in range(2):
            v = self.z_e @ (self.W @ self.g[m])
            out.append(2 * v * self.omega[m] - v ** 2 - self.Q[m])
        return out

    def trv(self) -> float:
        return float(np.sqrt(self.N) * (self.Q[0] - self.Q[1]) / np.sqrt(self.sigma2_rv()[0]))

    def f_statistic(self) -> Tuple[float, float]:
        """Eq. (F) of Duarte et al. (2026), Appendix B in the r-dimensional form."""
        N, r, W, z_e = self.N, self.r, self.W, self.z_e
        W_inv = np.linalg.inv(W)
        e = [self.omega[m] - z_e @ (W @ self.g[m]) for m in range(2)]
        phi = [(e[m][:, None] * z_e) @ W for m in range(2)]
        s = [np.trace((phi[a].T @ phi[b] / N) @ W_inv) / r for a, b in ((0, 0), (1, 1), (0, 1))]
        rho2 = (s[0] - s[1]) ** 2 / ((s[0] + s[1]) ** 2 - 4 * s[2] ** 2)
        g1, g2 = self.g
        num = s[1] * (g1 @ W @ g1) + s[0] * (g2 @ W @ g2) - 2 * s[2] * (g1 @ W @ g2)
        F = (1 - rho2) * N / (2 * r) * num / (s[0] * s[1] - s[2] ** 2)
        return float(F), float(np.sign(s[0] - s[1]) * np.sqrt(rho2))


# ---------------------------------------------------------------------------
# DGP: log-cost regression with endogenous log quantity (and its square)
# ---------------------------------------------------------------------------

def _make_data(seed: int, N: int = 600, d_z: int = 4, d_q: int = 1, n_firms: int = 3) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    Qm, _ = np.linalg.qr(rng.normal(size=(d_z, d_z)))
    L = Qm @ np.diag(np.sqrt(np.linspace(9.0, 0.5, d_z))) @ Qm.T
    z = (rng.exponential(1.0, size=(N, d_z)) - 1.0) @ L.T          # skewed, correlated, unequal eigenvalues
    w1 = rng.uniform(-1, 1, size=N)
    firm_ids = rng.integers(0, n_firms, size=N)
    firm_effect = np.array([0.0, 0.4, -0.3])[firm_ids]
    shocks = rng.multivariate_normal([0, 0], [[0.16, 0.12], [0.12, 0.25]], size=N)
    u, omega0 = shocks[:, 0], shocks[:, 1]
    zeta = rng.uniform(0.4, 1.0, size=d_z) * rng.choice([-1, 1], size=d_z)
    log_q = 3.0 + z @ zeta + 0.5 * w1 + u
    q = log_q[:, None] if d_q == 1 else np.column_stack([log_q, log_q ** 2])
    gamma = np.array([-0.15, 0.02])[:d_q]
    log_c = q @ gamma + 1.0 + 0.3 * w1 + firm_effect + omega0
    # two misspecified models: deviations linear in (z, w) plus small idiosyncratic noise
    dev = []
    for m in range(2):
        a = rng.normal(size=d_z) * 0.15
        dev.append(z @ a + rng.normal(scale=0.1, size=N))
    costs = [log_c + dev[0], log_c + dev[1]]
    prices = np.exp(log_c) + 1.0
    df = pd.DataFrame({
        'market_ids': np.arange(N) // 2, 'firm_ids': firm_ids,
        'prices': prices, 'shares': np.full(N, 0.1), 'w1': w1, 'log_q': log_q,
        'markups_m1': prices - np.exp(costs[0]), 'markups_m2': prices - np.exp(costs[1]),
    })
    for k in range(d_z):
        df[f'iv{k}'] = z[:, k]
    if d_q == 2:
        df['log_q_sq'] = log_q ** 2
    return {'df': df, 'z': z, 'w1': w1, 'q': q, 'costs': costs, 'firm_ids': firm_ids, 'd_z': d_z, 'd_q': d_q}


def _solve(data: Dict[str, object], absorb: bool = False, iv_cols: Optional[Sequence[str]] = None):
    df = data['df']
    d_z, d_q = data['d_z'], data['d_q']
    iv_cols = list(iv_cols) if iv_cols is not None else [f'iv{k}' for k in range(d_z)]
    terms = ('0 + w1 + log_q' if absorb else '1 + w1 + log_q') + (' + log_q_sq' if d_q == 2 else '')
    endog = ('log_q', 'log_q_sq') if d_q == 2 else 'log_q'
    kwargs = {'absorb': 'C(firm_ids)'} if absorb else {}
    problem = pyRVtest.Problem(
        cost_formulation=pyRVtest.Formulation(terms, **kwargs),
        instrument_formulation=pyRVtest.Formulation('0 + ' + ' + '.join(iv_cols)),
        models=[pyRVtest.UserSuppliedMarkups(markups='markups_m1'),
                pyRVtest.UserSuppliedMarkups(markups='markups_m2')],
        product_data=df, demand_results=None, endogenous_cost_component=endog,
    )
    verbose = pyRVtest.options.verbose
    pyRVtest.options.verbose = False
    try:
        return problem.solve(demand_adjustment=False, clustering_adjustment=False, costs_type='log')
    finally:
        pyRVtest.options.verbose = verbose


def _reference(data: Dict[str, object], absorb: bool = False, sel: Optional[Sequence[int]] = None) -> _Reference:
    N = data['z'].shape[0]
    if absorb:
        # FWL: absorbing firm effects == including firm dummies among the exogenous shifters
        dummies = np.eye(3)[data['firm_ids']]
        w = np.column_stack([dummies, data['w1']])
    else:
        w = np.column_stack([np.ones(N), data['w1']])
    return _Reference(data['z'], w, data['q'], data['costs'], sel=sel)


RTOL = 1e-9


# ---------------------------------------------------------------------------
# 1. A_W is the derivative of the matrix square root
# ---------------------------------------------------------------------------

def test_sqrt_derivative_matches_finite_difference():
    rng = np.random.default_rng(0)
    B = rng.normal(size=(4, 4))
    W = B @ B.T + 4 * np.eye(4)
    D = rng.normal(size=(4, 4))
    D = (D + D.T) / 2
    eps = 1e-5
    W_inv = np.linalg.inv(W)
    fd = (scipy.linalg.sqrtm(np.linalg.inv(W_inv + eps * D))
          - scipy.linalg.sqrtm(np.linalg.inv(W_inv - eps * D))) / (2 * eps)   # central difference
    analytic = (_sqrt_derivative(W) @ D.reshape(-1, order='F')).reshape(4, 4, order='F')
    np.testing.assert_allclose(analytic, np.real(fd), rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------------------
# 2-4. d_q = 1
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def data_dq1():
    return _make_data(seed=11, d_z=4, d_q=1)


@pytest.fixture(scope='module')
def results_dq1(data_dq1):
    return _solve(data_dq1)


class TestSingleEndogenousRegressor:

    def test_scalar_contraction_equals_package(self, data_dq1, results_dq1):
        ref = _reference(data_dq1)
        assert ref.r == 3
        # observation-by-observation bridge from the vector psi to the scalar score
        phi = ref.scalar_scores()
        for m in range(2):
            bridged = 2.0 * ref.psi(m) @ (_sym_power(ref.W, 0.5) @ ref.g[m])
            np.testing.assert_allclose(bridged, phi[m], rtol=RTOL, atol=1e-12)
        # package outputs
        np.testing.assert_allclose(results_dq1.Q[0], ref.Q, rtol=RTOL)
        np.testing.assert_allclose(results_dq1.RV_denominator[0][0, 1], np.sqrt(ref.sigma2_rv()[0]), rtol=RTOL)
        np.testing.assert_allclose(results_dq1.TRV[0][0, 1], ref.trv(), rtol=RTOL)

    def test_qtilde_terms_cancel_in_scalar_but_not_in_vector(self, data_dq1):
        ref = _reference(data_dq1)
        s2_with, V11_with = ref.sigma2_rv(include_qtilde=True)
        s2_without, V11_without = ref.sigma2_rv(include_qtilde=False)
        assert np.linalg.norm(V11_with - V11_without) > 1e-3 * np.linalg.norm(V11_with)
        np.testing.assert_allclose(s2_with, s2_without, rtol=RTOL)

    def test_selection_invariance_and_pseudo_inverse_equivalence(self, data_dq1, results_dq1):
        ref_a = _reference(data_dq1)
        # any other admissible selection of r = 3 out of 4 residualized instruments
        other = [1, 2, 3] if list(ref_a.sel) == [0, 1, 2] else [0, 1, 2]
        ref_b = _reference(data_dq1, sel=other)
        assert not np.array_equal(ref_a.sel, ref_b.sel)
        np.testing.assert_allclose(ref_a.Q, ref_b.Q, rtol=RTOL)
        np.testing.assert_allclose(ref_a.sigma2_rv()[0], ref_b.sigma2_rv()[0], rtol=RTOL)
        Fa, rho_a = ref_a.f_statistic()
        Fb, rho_b = ref_b.f_statistic()
        np.testing.assert_allclose(Fa, Fb, rtol=RTOL)
        np.testing.assert_allclose(rho_a, rho_b, rtol=RTOL)
        # the package's d_z-dimensional pseudo-inverse form gives the same F and rho
        np.testing.assert_allclose(results_dq1.F[0][0, 1], Fa, rtol=1e-8)
        np.testing.assert_allclose(results_dq1.rho[0][0, 1], rho_a, rtol=1e-8)


# ---------------------------------------------------------------------------
# 5. d_q = 2: r = d_z - 2
# ---------------------------------------------------------------------------

class TestTwoEndogenousRegressors:

    @pytest.fixture(scope='class')
    def data(self):
        return _make_data(seed=23, N=800, d_z=5, d_q=2)

    def test_package_matches_reference_with_r_equal_dz_minus_2(self, data):
        results = _solve(data)
        ref = _reference(data)
        assert ref.r == 3
        np.testing.assert_allclose(results.Q[0], ref.Q, rtol=RTOL)
        np.testing.assert_allclose(results.RV_denominator[0][0, 1], np.sqrt(ref.sigma2_rv()[0]), rtol=RTOL)
        np.testing.assert_allclose(results.TRV[0][0, 1], ref.trv(), rtol=RTOL)
        F, rho = ref.f_statistic()
        np.testing.assert_allclose(results.F[0][0, 1], F, rtol=1e-8)
        np.testing.assert_allclose(results.rho[0][0, 1], rho, rtol=1e-8)
        # Note: the value of F is algebraically invariant to the divisor (it
        # cancels between the sigma-hats and the N / (2 r) prefactor); r enters
        # through the critical-value row, which the package looks up at
        # K_effective = K - K_endog = r.


# ---------------------------------------------------------------------------
# 6. absorbed cost-side fixed effects
# ---------------------------------------------------------------------------

def test_fixed_effects_path_matches_reference():
    data = _make_data(seed=37, d_z=4, d_q=1)
    results = _solve(data, absorb=True)
    ref = _reference(data, absorb=True)
    np.testing.assert_allclose(results.Q[0], ref.Q, rtol=1e-8)
    np.testing.assert_allclose(results.RV_denominator[0][0, 1], np.sqrt(ref.sigma2_rv()[0]), rtol=1e-8)
    np.testing.assert_allclose(results.TRV[0][0, 1], ref.trv(), rtol=1e-8)
    np.testing.assert_allclose(results.F[0][0, 1], ref.f_statistic()[0], rtol=1e-7)


# ---------------------------------------------------------------------------
# 7. effective-rank audit
# ---------------------------------------------------------------------------

def test_rank_audit_warns_when_rank_differs_from_k_effective():
    # A rank-3 covariance of 5 residualized instruments with K_endog = 1
    # (K_effective = 4): one redundant instrument column went undetected.
    rng = np.random.default_rng(0)
    B = rng.normal(size=(5, 3))
    W_inverse = B @ B.T
    with pytest.warns(UserWarning, match='K_effective = K - K_endog = 5 - 1 = 4'):
        observed = _audit_effective_rank(W_inverse, K=5, K_endog=1, instrument=0)
    assert observed == 3
    # consistent rank: silent
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        assert _audit_effective_rank(B[:3] @ B[:3].T + np.eye(3), K=4, K_endog=1, instrument=0) == 3


def test_no_rank_warning_in_regular_case(data_dq1):
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        _solve(data_dq1)
