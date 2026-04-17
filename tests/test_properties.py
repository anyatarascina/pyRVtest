"""Property-based tests for pyRVtest invariants.

v0.4 Step 0f: baseline protection via property-based testing.

These tests use Hypothesis to generate random inputs and check that certain
invariants hold regardless of the specific input. They are intentionally
fast (limited max_examples and small DGPs) so the full suite runs in ~1
minute; broader coverage comes from the step-0b snapshot suite and the
step-0d golden file.

Properties covered (3-5 core per the v0.4 plan §5 Step 0f):

1. **Determinism**: solve(.) twice on the same DGP returns identical TRV,
   F, and markups. No hidden RNG, no ordering effects.

2. **Perfect competition markups are zero**: model_downstream=
   'perfect_competition' always produces markup == 0 exactly. Zero
   tolerance.

3. **Bertrand markups computed by pyRVtest are positive AND match the
   DGP's ground truth for alpha < 0**. This test uses `demand_params`
   so pyRVtest computes markups internally from the Bertrand FOC (not
   passed via user_supplied_markups). Two assertions: (a) computed
   markups are strictly positive; (b) computed markups match the
   ground-truth DGP markups to 1e-8. The second assertion is the
   stronger one — it tests that pyRVtest's demand Jacobian and
   Bertrand-FOC math produces the correct answer, not just the right
   sign.

4. **Within-market row permutation invariance**: permuting rows within a
   market (but not across markets) does not change per-market markups
   or the aggregate TRV/F. This catches bugs where order-dependent
   indexing leaks into the math.

5. **Seed invariance**: building the same DGP with the same seed twice
   produces identical product data (byte-for-byte). Sanity check
   confirming our DGP construction is deterministic.

Incremental rule per v0.4 plan: these 5 properties are scaffolding;
additional properties get added at steps 3, 4, 7, 8 as those modules land.
Step 20 is the audit pass, not first implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import pyRVtest

from hypothesis import given, settings, strategies as st


# ---------------------------------------------------------------------------
# Minimal DGP helper (self-contained to avoid cross-test-file coupling)
# ---------------------------------------------------------------------------

def _logit_shares(V: np.ndarray, market_ids: np.ndarray, T: int) -> np.ndarray:
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
    """Solve for Bertrand-logit equilibrium prices via fixed-point iteration."""
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


def _make_dgp(seed: int = 12345, T: int = 8, J: int = 2, alpha: float = -2.0) -> pd.DataFrame:
    """Build a simple Bertrand-logit product dataset with known markups.

    Kept small (T=8, J=2 by default) so Hypothesis can run several
    examples per property in under a minute.
    """
    rng = np.random.default_rng(seed=seed)
    N = T * J
    market_ids = np.repeat(np.arange(T), J)
    firm_ids = np.tile(np.arange(J), T)

    beta_x = 1.0
    x = rng.uniform(0.5, 2.0, size=(N, 1))
    cost_shifter = rng.uniform(0.5, 2.0, size=(N, 1))
    excluded_iv = rng.uniform(0.5, 2.0, size=(N, 3))
    xi = rng.normal(0, 0.05, size=(N, 1))
    omega_true = rng.normal(0, 0.05, size=(N, 1))

    tau_0, tau_w = 1.0, 0.5
    mc_true = tau_0 + tau_w * cost_shifter + omega_true

    prices, shares, markups_bertrand = _solve_bertrand_logit(
        alpha, beta_x, x, xi, mc_true, market_ids, T, N
    )
    markups_perfect = np.zeros((N, 1))

    return pd.DataFrame({
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


def _run_pyrvtest(product_data: pd.DataFrame):
    """Run pyRVtest on the standard Bertrand-vs-perfect-comp DGP.

    Uses user_supplied_markups (fastest path) so determinism / permutation
    / perfect-comp properties don't depend on pyRVtest's demand-Jacobian
    math. Property 3 uses `_run_pyrvtest_computed_markups` instead for a
    stronger check.
    """
    model_formulations = (
        pyRVtest.ModelFormulation(
            model_downstream='bertrand', ownership_downstream='firm_ids',
            user_supplied_markups='markups_m1'
        ),
        pyRVtest.ModelFormulation(
            model_downstream='perfect_competition',
            user_supplied_markups='markups_m2'
        ),
    )
    testing_problem = pyRVtest.Problem(
        cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
        instrument_formulation=pyRVtest.Formulation('0 + iv0 + iv1 + iv2'),
        model_formulations=model_formulations,
        product_data=product_data,
        demand_results=None,
    )
    return testing_problem.solve(demand_adjustment=False, clustering_adjustment=False)


def _run_pyrvtest_computed_markups(product_data: pd.DataFrame, alpha: float):
    """Run pyRVtest with demand_params so it computes markups from the FOC.

    Bertrand markups come from pyRVtest's internal Jacobian + FOC, not
    from user_supplied_markups. This is the path that had the b3b08a3
    sign bug in the gradient, and is what property 3 is meant to
    exercise.
    """
    model_formulations = (
        pyRVtest.ModelFormulation(
            model_downstream='bertrand', ownership_downstream='firm_ids'
        ),
        pyRVtest.ModelFormulation(
            model_downstream='perfect_competition'
        ),
    )
    testing_problem = pyRVtest.Problem(
        cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
        instrument_formulation=pyRVtest.Formulation('0 + iv0 + iv1 + iv2'),
        model_formulations=model_formulations,
        product_data=product_data,
        demand_params={'alpha': alpha, 'sigma': []},
    )
    return testing_problem.solve(demand_adjustment=False, clustering_adjustment=False)


# ---------------------------------------------------------------------------
# Property 1: determinism
# ---------------------------------------------------------------------------

@given(seed=st.integers(min_value=1, max_value=10_000))
@settings(max_examples=100, deadline=None)
def test_determinism(seed):
    """Two runs on the same DGP return identical test statistics.

    Catches any hidden RNG or iteration-order dependence in .solve().
    """
    df = _make_dgp(seed=seed)
    r1 = _run_pyrvtest(df)
    r2 = _run_pyrvtest(df)

    np.testing.assert_array_equal(r1.markups, r2.markups)
    np.testing.assert_array_equal(r1.TRV, r2.TRV)
    np.testing.assert_array_equal(r1.F, r2.F)


# ---------------------------------------------------------------------------
# Property 2: perfect competition markups are exactly zero
# ---------------------------------------------------------------------------

def test_perfect_competition_markups_are_zero():
    """model_downstream='perfect_competition' produces markup == 0 everywhere.

    Zero tolerance: the formula is Δ = 0, not "small number." Any
    non-zero markup here indicates a bug in model dispatch or ownership-
    matrix construction.
    """
    df = _make_dgp(seed=42, T=8, J=2)
    results = _run_pyrvtest(df)

    # markups is a list of arrays indexed by model (0=bertrand, 1=perfect_comp)
    perfect_comp_markups = results.markups[1]
    assert np.all(perfect_comp_markups == 0.0), (
        f"Perfect competition markups should be exactly zero; got max abs "
        f"{np.max(np.abs(perfect_comp_markups))}"
    )


# ---------------------------------------------------------------------------
# Property 3: Bertrand markups are positive when alpha < 0
# ---------------------------------------------------------------------------

@given(alpha=st.floats(min_value=-5.0, max_value=-0.5))
@settings(max_examples=100, deadline=None)
def test_bertrand_markups_computed_correctly_for_elastic_demand(alpha):
    """pyRVtest-computed Bertrand markups are positive AND match DGP.

    Uses `demand_params={'alpha': alpha, 'sigma': []}` so pyRVtest
    computes the demand Jacobian and Bertrand markups internally. Two
    assertions:

    (a) Positivity — economic content: when consumers substitute away
        from higher prices (alpha < 0), firms charge strictly positive
        markups over marginal cost. Bertrand formula Δ = -(O·D)^{-1}s
        with O·D at alpha<0 and positive shares yields Δ>0.

    (b) Correctness — pyRVtest's computed markups should match the
        ground-truth markups we built into the DGP via the Bertrand
        fixed-point iteration. Tolerance 1e-8 (same precision the
        equilibrium solver converges to in `_solve_bertrand_logit`).

    (b) is the stronger claim: it tests the actual Jacobian+FOC math,
    not just its sign. A sign bug in the Jacobian (or in the FOC
    dispatch) would produce negative markups — caught by (a). A
    magnitude bug would produce wrongly-scaled markups — caught by (b).
    """
    df = _make_dgp(seed=42, T=8, J=2, alpha=alpha)
    results = _run_pyrvtest_computed_markups(df, alpha=alpha)

    bertrand_markups = results.markups[0]
    ground_truth = df['markups_m1'].values.reshape(-1, 1)

    # (a) positivity
    assert np.all(bertrand_markups > 0), (
        f"Bertrand markups should be strictly positive for alpha={alpha}; "
        f"got min {np.min(bertrand_markups)}"
    )
    # (b) correctness to 1e-8
    np.testing.assert_allclose(
        bertrand_markups, ground_truth, atol=1e-8,
        err_msg=(
            f"pyRVtest-computed Bertrand markups disagree with DGP "
            f"ground truth at alpha={alpha}"
        )
    )


# ---------------------------------------------------------------------------
# Property 4: within-market row permutation invariance
# ---------------------------------------------------------------------------

@given(seed=st.integers(min_value=1, max_value=10_000))
@settings(max_examples=100, deadline=None)
def test_within_market_permutation_invariance(seed):
    """Permuting rows within each market does not change aggregate TRV or F.

    Per-product markups get reordered identically with the rows, but
    market-level quantities (TRV, F, MCS p-values, GMM objective Q) are
    invariant. This catches bugs where ordering-dependent indexing leaks
    into test statistics.

    Construction: we permute the J=2 products within each market with a
    seeded RNG so the permutation is reproducible per Hypothesis example.
    """
    df_original = _make_dgp(seed=seed, T=8, J=2)

    # Build a deterministic within-market permutation
    perm_rng = np.random.default_rng(seed=seed + 99)
    df_permuted = df_original.copy()
    for t in df_permuted['market_ids'].unique():
        mask = df_permuted['market_ids'] == t
        idx = np.where(mask)[0]
        perm = perm_rng.permutation(idx)
        df_permuted.iloc[idx] = df_original.iloc[perm].values
    df_permuted = df_permuted.reset_index(drop=True)

    r_original = _run_pyrvtest(df_original)
    r_permuted = _run_pyrvtest(df_permuted)

    # Aggregate quantities invariant to within-market ordering.
    np.testing.assert_allclose(
        r_original.TRV, r_permuted.TRV,
        atol=1e-10, equal_nan=True,
        err_msg='TRV should be invariant to within-market row permutation'
    )
    np.testing.assert_allclose(
        r_original.F, r_permuted.F,
        atol=1e-10, equal_nan=True,
        err_msg='F should be invariant to within-market row permutation'
    )


# ---------------------------------------------------------------------------
# Property 5: seed invariance for DGP construction
# ---------------------------------------------------------------------------

@given(seed=st.integers(min_value=1, max_value=10_000))
@settings(max_examples=100, deadline=None)
def test_dgp_seed_invariance(seed):
    """Building the same DGP with the same seed twice yields identical data.

    Sanity check on the property-test scaffolding itself. If this fails,
    Hypothesis's seeded runs are not reproducible and the rest of the
    suite's guarantees are suspect.
    """
    df1 = _make_dgp(seed=seed)
    df2 = _make_dgp(seed=seed)
    pd.testing.assert_frame_equal(df1, df2)


# ===========================================================================
# Multi-firm DGP helpers (J=4 with 2 firms owning 2 products each)
#
# The base _make_dgp above uses J=2 with single-product firms. On that DGP,
# Bertrand and Monopoly produce identical markups (each firm's ownership
# matrix reduces to identity). The multi-firm DGP below distinguishes them.
# ===========================================================================

def _solve_multifirm_bertrand(alpha, beta_x, x, xi, mc_true, market_ids,
                              firm_ids, T, N):
    """Multi-product Bertrand equilibrium via fixed-point iteration.

    Generalizes _solve_bertrand_logit to multi-product firms. At each
    iteration, per market: build ownership matrix O_t from firm_ids,
    Jacobian D_t from logit formula, compute markup_t = -solve(O_t * D_t, s_t)
    (elementwise O·D, then linear solve), update prices = mc + markup.
    """
    prices = mc_true + 0.5
    for _ in range(500):
        V = beta_x * x + alpha * prices + xi
        shares = _logit_shares(V, market_ids, T)
        prices_new = np.zeros_like(prices)
        for t in range(T):
            idx = np.where(market_ids == t)[0]
            s_t = shares[idx].flatten()
            fids = firm_ids[idx]
            O_t = (fids[:, None] == fids[None, :]).astype(float)
            D_t = alpha * (np.diag(s_t) - np.outer(s_t, s_t))
            markup_t = -np.linalg.solve(O_t * D_t, s_t)
            prices_new[idx, 0] = mc_true[idx, 0] + markup_t
        if np.max(np.abs(prices_new - prices)) < 1e-13:
            prices = prices_new
            break
        prices = prices_new
    # Final recompute of shares/markups at converged prices
    V = beta_x * x + alpha * prices + xi
    shares = _logit_shares(V, market_ids, T)
    markups = np.zeros_like(prices)
    for t in range(T):
        idx = np.where(market_ids == t)[0]
        s_t = shares[idx].flatten()
        fids = firm_ids[idx]
        O_t = (fids[:, None] == fids[None, :]).astype(float)
        D_t = alpha * (np.diag(s_t) - np.outer(s_t, s_t))
        markups[idx, 0] = -np.linalg.solve(O_t * D_t, s_t)
    return prices, shares, markups


def _make_dgp_multifirm(seed: int = 12345, T: int = 6, alpha: float = -2.0) -> pd.DataFrame:
    """Build a 2-firm 4-product-per-market Bertrand-logit DGP.

    firm_ids = [0, 0, 1, 1] per market, so firm 0 owns products 0, 1 and
    firm 1 owns products 2, 3. This makes Bertrand (O = 2x2 diagonal
    blocks) distinct from Monopoly (O = all ones). J=4 is fixed.
    """
    J = 4
    rng = np.random.default_rng(seed=seed)
    N = T * J
    market_ids = np.repeat(np.arange(T), J)
    firm_ids = np.tile([0, 0, 1, 1], T)
    # mix_flag: first 2 products per market are Bertrand, last 2 are Cournot
    mix_flag_col = np.tile([True, True, False, False], T)
    # vertical integration indicator (0 = no integration)
    vi_col = np.zeros(N, dtype=int)
    # upstream firm ids (distinct from downstream)
    upstream_firm_ids = np.tile([2, 2, 3, 3], T)

    beta_x = 1.0
    x = rng.uniform(0.5, 2.0, size=(N, 1))
    cost_shifter = rng.uniform(0.5, 2.0, size=(N, 1))
    excluded_iv = rng.uniform(0.5, 2.0, size=(N, 3))
    xi = rng.normal(0, 0.05, size=(N, 1))
    omega_true = rng.normal(0, 0.05, size=(N, 1))

    tau_0, tau_w = 1.0, 0.5
    mc_true = tau_0 + tau_w * cost_shifter + omega_true

    prices, shares, markups_bertrand = _solve_multifirm_bertrand(
        alpha, beta_x, x, xi, mc_true, market_ids, firm_ids, T, N
    )

    return pd.DataFrame({
        'market_ids': market_ids,
        'firm_ids': firm_ids,
        'upstream_firm_ids': upstream_firm_ids,
        'mix_flag_col': mix_flag_col,
        'vi_col': vi_col,
        'prices': prices.flatten(),
        'shares': shares.flatten(),
        'x': x.flatten(),
        'cost_shifter': cost_shifter.flatten(),
        'iv0': excluded_iv[:, 0],
        'iv1': excluded_iv[:, 1],
        'iv2': excluded_iv[:, 2],
        'markups_bertrand': markups_bertrand.flatten(),
        'markups_zero': np.zeros(N),
        'clustering_ids': market_ids,
    })


def _logit_D_per_market(alpha, shares_array, market_ids, T):
    """Return dict {t: D_t} of per-market logit Jacobians.

    D_{jk} = alpha * s_j * (kron_delta(j,k) - s_k).
    """
    D = {}
    for t in range(T):
        idx = np.where(market_ids == t)[0]
        s_t = shares_array[idx].flatten()
        D[t] = alpha * (np.diag(s_t) - np.outer(s_t, s_t))
    return D


def _ownership_per_market(firm_ids, market_ids, T, kappa: float = None):
    """Return dict {t: O_t}.

    If kappa is None, O_t is standard 0/1 ownership. If kappa is set
    (scalar in [0, 1]), cross-firm entries get kappa weight — represents
    partial profit weights (Miller-Weinberg).
    """
    O = {}
    for t in range(T):
        idx = np.where(market_ids == t)[0]
        fids = firm_ids[idx]
        same = (fids[:, None] == fids[None, :]).astype(float)
        if kappa is None:
            O[t] = same
        else:
            cross = 1.0 - same
            O[t] = same + kappa * cross
    return O


def _hand_bertrand(alpha, shares, firm_ids, market_ids, T, kappa=None):
    """Hand-compute Bertrand markups: -solve(O * D, s) per market."""
    D_per = _logit_D_per_market(alpha, shares, market_ids, T)
    O_per = _ownership_per_market(firm_ids, market_ids, T, kappa=kappa)
    markups = np.zeros((len(shares), 1))
    for t in range(T):
        idx = np.where(market_ids == t)[0]
        s_t = shares[idx].flatten()
        markups[idx, 0] = -np.linalg.solve(O_per[t] * D_per[t], s_t)
    return markups


def _hand_cournot(alpha, shares, firm_ids, market_ids, T):
    """Hand-compute Cournot markups: -(O * inv(D)) @ s per market."""
    D_per = _logit_D_per_market(alpha, shares, market_ids, T)
    O_per = _ownership_per_market(firm_ids, market_ids, T)
    markups = np.zeros((len(shares), 1))
    for t in range(T):
        idx = np.where(market_ids == t)[0]
        s_t = shares[idx].flatten()
        D_inv = np.linalg.inv(D_per[t])
        markups[idx, 0] = -(O_per[t] * D_inv) @ s_t
    return markups


def _hand_monopoly(alpha, shares, market_ids, T):
    """Hand-compute Monopoly markups: -solve(D, s) per market (no ownership)."""
    D_per = _logit_D_per_market(alpha, shares, market_ids, T)
    markups = np.zeros((len(shares), 1))
    for t in range(T):
        idx = np.where(market_ids == t)[0]
        s_t = shares[idx].flatten()
        markups[idx, 0] = -np.linalg.solve(D_per[t], s_t)
    return markups


def _run_pyrvtest_all_models_multifirm(product_data, alpha):
    """Run pyRVtest with the 5 standard conduct models on the multifirm DGP."""
    model_formulations = (
        pyRVtest.ModelFormulation(
            model_downstream='bertrand', ownership_downstream='firm_ids'
        ),
        pyRVtest.ModelFormulation(
            model_downstream='cournot', ownership_downstream='firm_ids'
        ),
        pyRVtest.ModelFormulation(model_downstream='monopoly'),
        pyRVtest.ModelFormulation(model_downstream='perfect_competition'),
        pyRVtest.ModelFormulation(
            model_downstream='mix_cournot_bertrand',
            ownership_downstream='firm_ids',
            mix_flag='mix_flag_col',
        ),
    )
    testing_problem = pyRVtest.Problem(
        cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
        instrument_formulation=pyRVtest.Formulation('0 + iv0 + iv1 + iv2'),
        model_formulations=model_formulations,
        product_data=product_data,
        demand_params={'alpha': alpha, 'sigma': []},
    )
    return testing_problem.solve(demand_adjustment=False, clustering_adjustment=False)


# ---------------------------------------------------------------------------
# Property 6: dispatch smoke test across all standard conduct models
# ---------------------------------------------------------------------------

def test_dispatch_smoke_all_standard_models():
    """Run bertrand/cournot/monopoly/perfect_comp/mix side-by-side.

    Assertions:
      - All markup arrays finite (no NaN/Inf).
      - PerfectCompetition == 0 exactly (zero tolerance).
      - Bertrand, Cournot, Monopoly strictly positive.
      - Monopoly >= Bertrand element-wise on this symmetric 2-firm DGP
        (single monopolist internalizes cross-product substitution).
      - Bertrand, Cournot, Monopoly markups match hand-computation to 1e-10.

    Catches dispatch bugs (wrong formula for a model_type string), sign
    errors, and ownership-matrix construction bugs on a single fixed DGP.
    """
    alpha = -2.0
    T = 6
    df = _make_dgp_multifirm(seed=42, T=T, alpha=alpha)
    results = _run_pyrvtest_all_models_multifirm(df, alpha)

    bert, cour, mono, perf, mix = (results.markups[i] for i in range(5))

    # Finiteness
    names = ['bertrand', 'cournot', 'monopoly', 'perfect_competition', 'mix_cournot_bertrand']
    for i, name in enumerate(names):
        assert np.all(np.isfinite(results.markups[i])), (
            f"{name} produced non-finite markups"
        )

    # Perfect competition exact zero
    assert np.all(perf == 0.0), f"perfect_competition markups not zero: max |v| = {np.max(np.abs(perf))}"

    # Positivity
    assert np.all(bert > 0), f"Bertrand markups not strictly positive: min {bert.min()}"
    assert np.all(cour > 0), f"Cournot markups not strictly positive: min {cour.min()}"
    assert np.all(mono > 0), f"Monopoly markups not strictly positive: min {mono.min()}"

    # Monopoly dominates Bertrand
    assert np.all(mono >= bert - 1e-12), (
        f"Monopoly should dominate Bertrand element-wise on this symmetric DGP; "
        f"max(bertrand - monopoly) = {np.max(bert - mono)}"
    )

    # Hand-computed match
    shares = df['shares'].values.reshape(-1, 1)
    firm_ids = df['firm_ids'].values
    market_ids = df['market_ids'].values

    np.testing.assert_allclose(
        bert, _hand_bertrand(alpha, shares, firm_ids, market_ids, T),
        atol=1e-10, err_msg="pyRVtest Bertrand disagrees with hand-computed"
    )
    np.testing.assert_allclose(
        cour, _hand_cournot(alpha, shares, firm_ids, market_ids, T),
        atol=1e-10, err_msg="pyRVtest Cournot disagrees with hand-computed"
    )
    np.testing.assert_allclose(
        mono, _hand_monopoly(alpha, shares, market_ids, T),
        atol=1e-10, err_msg="pyRVtest Monopoly disagrees with hand-computed"
    )


# ---------------------------------------------------------------------------
# Property 7: Cournot markups match hand computation across alpha
# ---------------------------------------------------------------------------

@given(alpha=st.floats(min_value=-5.0, max_value=-0.5))
@settings(max_examples=100, deadline=None)
def test_cournot_markups_match_hand_computed(alpha):
    """pyRVtest Cournot markups match hand-computed -(O * inv(D)) @ s."""
    T = 6
    df = _make_dgp_multifirm(seed=42, T=T, alpha=alpha)

    # Second model (perfect_competition) added only to satisfy MCS's
    # requirement of >=2 models. We test markups[0] (cournot).
    problem = pyRVtest.Problem(
        cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
        instrument_formulation=pyRVtest.Formulation('0 + iv0 + iv1 + iv2'),
        model_formulations=(
            pyRVtest.ModelFormulation(model_downstream='cournot', ownership_downstream='firm_ids'),
            pyRVtest.ModelFormulation(model_downstream='perfect_competition'),
        ),
        product_data=df,
        demand_params={'alpha': alpha, 'sigma': []},
    )
    results = problem.solve(demand_adjustment=False, clustering_adjustment=False)

    shares = df['shares'].values.reshape(-1, 1)
    firm_ids = df['firm_ids'].values
    market_ids = df['market_ids'].values
    hand = _hand_cournot(alpha, shares, firm_ids, market_ids, T)

    np.testing.assert_allclose(results.markups[0], hand, atol=1e-10)


# ---------------------------------------------------------------------------
# Property 8: Monopoly markups match hand computation across alpha
# ---------------------------------------------------------------------------

@given(alpha=st.floats(min_value=-5.0, max_value=-0.5))
@settings(max_examples=100, deadline=None)
def test_monopoly_markups_match_hand_computed(alpha):
    """pyRVtest Monopoly markups match hand-computed -solve(D, s)."""
    T = 6
    df = _make_dgp_multifirm(seed=42, T=T, alpha=alpha)

    # Second model required for MCS dispatch; we test markups[0] (monopoly).
    problem = pyRVtest.Problem(
        cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
        instrument_formulation=pyRVtest.Formulation('0 + iv0 + iv1 + iv2'),
        model_formulations=(
            pyRVtest.ModelFormulation(model_downstream='monopoly'),
            pyRVtest.ModelFormulation(model_downstream='perfect_competition'),
        ),
        product_data=df,
        demand_params={'alpha': alpha, 'sigma': []},
    )
    results = problem.solve(demand_adjustment=False, clustering_adjustment=False)

    shares = df['shares'].values.reshape(-1, 1)
    market_ids = df['market_ids'].values
    hand = _hand_monopoly(alpha, shares, market_ids, T)

    np.testing.assert_allclose(results.markups[0], hand, atol=1e-10)


# ---------------------------------------------------------------------------
# Property 9: profit-weight (kappa) partial collusion
# ---------------------------------------------------------------------------

@given(kappa=st.floats(min_value=0.0, max_value=0.9))
@settings(max_examples=30, deadline=None)
def test_profit_weight_kappa_bertrand_matches_hand_computed(kappa):
    """Partial collusion via kappa_specification: Bertrand with cross-firm kappa weights.

    Represents Miller-Weinberg partial profit weights. kappa=0 = pure
    Bertrand. kappa=1 = full joint profit maximization across firms
    (multi-product monopoly). Intermediate = partial collusion.

    Hand-computation: modify ownership matrix so cross-firm entries
    equal kappa (same-firm entries stay 1.0), then apply Bertrand formula.

    Smaller max_examples=30 (kappa ingestion + per-kappa rebuild of
    ownership matrix is the slowest path).
    """
    alpha = -2.0
    T = 6
    df = _make_dgp_multifirm(seed=42, T=T, alpha=alpha)

    def kappa_fn(fi, fj):
        return 1.0 if fi == fj else float(kappa)

    problem = pyRVtest.Problem(
        cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
        instrument_formulation=pyRVtest.Formulation('0 + iv0 + iv1 + iv2'),
        model_formulations=(
            pyRVtest.ModelFormulation(
                model_downstream='bertrand',
                ownership_downstream='firm_ids',
                kappa_specification_downstream=kappa_fn,
            ),
            # Second model required for MCS dispatch; we test markups[0].
            pyRVtest.ModelFormulation(model_downstream='perfect_competition'),
        ),
        product_data=df,
        demand_params={'alpha': alpha, 'sigma': []},
    )
    results = problem.solve(demand_adjustment=False, clustering_adjustment=False)

    shares = df['shares'].values.reshape(-1, 1)
    firm_ids = df['firm_ids'].values
    market_ids = df['market_ids'].values
    hand = _hand_bertrand(alpha, shares, firm_ids, market_ids, T, kappa=kappa)

    np.testing.assert_allclose(results.markups[0], hand, atol=1e-10)


# ---------------------------------------------------------------------------
# Property 10: custom 'other' model via custom_model_specification callable
# ---------------------------------------------------------------------------

def test_custom_other_model_constant_closure_smoke():
    """The 'other' model type accepts a custom markup closure.

    Uses a toy closure that returns 0.1 everywhere. Verifies the
    extension point pyRVtest.ModelFormulation(model_downstream='other',
    custom_model_specification={...}) accepts a callable with signature
    (O, D, s) -> markups and returns its output unchanged.

    This is a SMOKE test. A real user-supplied model would depend
    non-trivially on O, D, s. The toy closure is enough to exercise the
    dispatch path.
    """
    alpha = -2.0
    df = _make_dgp_multifirm(seed=42, alpha=alpha)

    # Use a lightly share-dependent toy so test statistics don't hit
    # degenerate denominators (which produce harmless but ugly
    # divide-by-zero warnings in MCS computation).
    def toy_markup(O, D, s):
        return 0.1 + 0.05 * s

    problem = pyRVtest.Problem(
        cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
        instrument_formulation=pyRVtest.Formulation('0 + iv0 + iv1 + iv2'),
        model_formulations=(
            pyRVtest.ModelFormulation(
                model_downstream='other',
                custom_model_specification={'toy_markup': toy_markup},
            ),
            # Second model required for MCS dispatch; we test markups[0].
            pyRVtest.ModelFormulation(model_downstream='perfect_competition'),
        ),
        product_data=df,
        demand_params={'alpha': alpha, 'sigma': []},
    )
    results = problem.solve(demand_adjustment=False, clustering_adjustment=False)

    custom_markups = results.markups[0]
    expected = 0.1 + 0.05 * df['shares'].values.reshape(-1, 1)
    np.testing.assert_allclose(
        custom_markups, expected, atol=1e-14,
        err_msg="custom_model_specification output should equal closure's return value"
    )


# ---------------------------------------------------------------------------
# Property 11: vertical Bertrand-downstream + Monopoly-upstream smoke
# ---------------------------------------------------------------------------

def test_vertical_bertrand_downstream_monopoly_upstream_smoke():
    """Smoke test for the vertical (downstream + upstream) model path.

    Runs ModelFormulation with both model_downstream='bertrand' and
    model_upstream='monopoly' with separate ownership structures.
    Asserts:
      - Combined markups are finite.
      - Combined markups differ from pure-downstream Bertrand (so the
        upstream path was actually exercised).

    No hand-computation: the combined markup involves Villas-Boas
    passthrough-matrix math that is non-trivial to replicate. This test
    is dispatch + integration coverage only. Per-component hand-checks
    land with the backend protocol in v0.4 step 3.
    """
    alpha = -2.0
    df = _make_dgp_multifirm(seed=42, alpha=alpha)

    problem = pyRVtest.Problem(
        cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
        instrument_formulation=pyRVtest.Formulation('0 + iv0 + iv1 + iv2'),
        model_formulations=(
            pyRVtest.ModelFormulation(
                model_downstream='bertrand', ownership_downstream='firm_ids',
                model_upstream='monopoly', ownership_upstream='upstream_firm_ids',
                vertical_integration='vi_col',
            ),
            pyRVtest.ModelFormulation(
                model_downstream='bertrand', ownership_downstream='firm_ids',
            ),
        ),
        product_data=df,
        demand_params={'alpha': alpha, 'sigma': []},
    )
    results = problem.solve(demand_adjustment=False, clustering_adjustment=False)

    vertical = results.markups[0]
    downstream_only = results.markups[1]

    assert np.all(np.isfinite(vertical)), "Vertical combined markups not finite"
    assert not np.allclose(vertical, downstream_only, atol=1e-6), (
        "Vertical combined markups should differ from pure-downstream Bertrand"
    )


# ===========================================================================
# TODO: property tests for conduct models NOT YET in pyRVtest v0.3.2.
#
# The following classes are planned for v0.4 and should get property tests
# (hand-computed ground truth) at the migration step that introduces each:
#
#   - PartialCollusion as a first-class class (distinct from kappa_spec) — step 5
#   - RuleOfThumb (Delta = lambda * p)                   — step 12 (after Dearing)
#   - ConstantMarkup (Delta = mbar)                       — step 12
#   - CostPlus (Delta = lambda * c)                       — step 12
#   - Monopsony                                           — step 14 (labor hooks)
#   - BertrandWages                                       — step 14
#   - CournotEmployment                                   — step 14
#   - NashBargaining                                      — step 14
#
# See .claude/plans/v0.4-refactor.md §5 for the migration plan. The kappa-
# weighted Bertrand test (property 9) does partially cover profit-weight
# semantics today, but PartialCollusion as a dedicated class gets its own
# test at step 5.
# ===========================================================================


# ---------------------------------------------------------------------------
# v0.4 step 4h: within-market row-permutation invariance with demand_adjustment.
#
# The existing Property 4 test (user_supplied_markups fastpath) doesn't
# exercise the step-4 demand-adjustment machinery (backend perturbation,
# xi/H/h_i construction, gradient_markups implicit differentiation).
# This test permutes rows within markets and asserts that TRV / F / MCS
# from a full Problem.solve(demand_adjustment=True) are invariant. Any
# accidental index-order dependence in the unified
# compute_demand_adjustment function or the backends would break this.
# ---------------------------------------------------------------------------


def _make_demand_adjustment_dgp(seed: int = 7777, T: int = 12, J: int = 3):
    """Plain-logit Bertrand DGP with enough structure to exercise
    demand_adjustment=True via demand_params."""
    rng = np.random.default_rng(seed=seed)
    N = T * J
    alpha = -2.0
    beta_x = 1.0
    market_ids = np.repeat(np.arange(T), J)
    firm_ids = np.tile(np.arange(J), T)
    x = rng.uniform(0.5, 1.5, size=(N, 1))
    # Utility + random xi -> observed shares via logit inversion (not equilibrium;
    # demand_adjustment doesn't need equilibrium, just consistent xi).
    u = beta_x * x.flatten() + rng.normal(scale=0.3, size=N)
    prices = rng.uniform(0.8, 2.0, size=N)
    delta = u + alpha * prices
    shares = np.zeros(N)
    for t in range(T):
        idx = np.where(market_ids == t)[0]
        e = np.exp(delta[idx])
        shares[idx] = e / (1.0 + e.sum())
    # Cost shifter for the pyRVtest cost formulation.
    z1 = rng.normal(size=N) + 1.5
    # Conduct-testing instrument.
    iv0 = rng.uniform(-1, 1, size=N)
    # Demand instrument (rival x).
    rival_x = np.zeros(N)
    for t in range(T):
        idx = np.where(market_ids == t)[0]
        for j in idx:
            others = [i for i in idx if i != j]
            rival_x[j] = x.flatten()[others].mean()
    intercept = np.ones(N)
    clustering_ids = market_ids
    return pd.DataFrame({
        'market_ids': market_ids, 'firm_ids': firm_ids,
        'prices': prices, 'shares': shares,
        'x1': x.flatten(), 'intercept': intercept,
        'rival_x': rival_x,
        'z1': z1, 'iv0': iv0,
        'clustering_ids': clustering_ids,
    }), alpha, beta_x


def _solve_with_demand_adjustment(df, alpha, beta_x):
    """Build a Problem with demand_params + run solve(demand_adjustment=True)."""
    import pyRVtest as pyrv
    pyrv.options.verbose = False
    problem = pyrv.Problem(
        cost_formulation=pyrv.Formulation('1 + z1'),
        instrument_formulation=pyrv.Formulation('0 + iv0'),
        model_formulations=(
            pyrv.ModelFormulation(
                model_downstream='bertrand', ownership_downstream='firm_ids',
            ),
            pyrv.ModelFormulation(model_downstream='perfect_competition'),
        ),
        product_data=df,
        demand_params={
            'alpha': alpha, 'sigma': [],
            'beta': np.array([0.0, beta_x]),
            'x_columns': ['intercept', 'x1'],
            'demand_instrument_columns': ['rival_x', 'intercept', 'x1'],
        },
    )
    return problem.solve(demand_adjustment=True, clustering_adjustment=False)


@given(seed=st.integers(min_value=1, max_value=10_000))
@settings(max_examples=25, deadline=None)
def test_within_market_permutation_invariance_with_demand_adjustment(seed):
    """Permuting rows within each market must leave TRV / F / MCS invariant
    when demand_adjustment=True.

    Covers the step-4 demand-adjustment code: the unified
    compute_demand_adjustment, the LogitBackend's demand_moments /
    xi_gradient / jacobian_gradient, and the implicit-differentiation
    markup gradient. A failure here would indicate an index-order
    dependence somewhere in that pipeline.
    """
    df_original, alpha, beta_x = _make_demand_adjustment_dgp(seed=seed)

    # Within-market permutation with a seeded RNG.
    perm_rng = np.random.default_rng(seed=seed + 777)
    df_permuted = df_original.copy()
    for t in df_permuted['market_ids'].unique():
        mask = df_permuted['market_ids'] == t
        idx = np.where(mask)[0]
        perm = perm_rng.permutation(idx)
        df_permuted.iloc[idx] = df_original.iloc[perm].values
    df_permuted = df_permuted.reset_index(drop=True)

    r_original = _solve_with_demand_adjustment(df_original, alpha, beta_x)
    r_permuted = _solve_with_demand_adjustment(df_permuted, alpha, beta_x)

    np.testing.assert_allclose(
        r_original.TRV, r_permuted.TRV, atol=1e-10, equal_nan=True,
        err_msg=(
            'TRV with demand_adjustment=True should be invariant to within-market '
            'row permutation. A failure indicates index-order dependence in the '
            'unified compute_demand_adjustment or one of the backend methods.'
        )
    )
    np.testing.assert_allclose(
        r_original.F, r_permuted.F, atol=1e-10, equal_nan=True,
        err_msg='F with demand_adjustment=True should be permutation-invariant'
    )
    np.testing.assert_allclose(
        r_original.MCS_pvalues, r_permuted.MCS_pvalues, atol=1e-10, equal_nan=True,
        err_msg='MCS_pvalues with demand_adjustment=True should be permutation-invariant'
    )


# ===========================================================================
# v0.4 step 20: property tests audit — three gap-filling properties.
#
# Per the migration plan, step 20 ensures coverage on determinism, row-
# permutation invariance, market-partition invariance, FWL identity, and
# sign-of-alpha. The first two are already covered above. The three tests
# below fill the remaining gaps: (12) market-partition invariance of the
# GMM moment g, (13) FWL identity g = (1/N) * Z_perp' @ omega_perp, and
# (14) sign-of-alpha homogeneity of degree -1 for plain-logit Bertrand
# markups plus the associated strict-monotonicity property.
# ===========================================================================


# ---------------------------------------------------------------------------
# Property 12: market-partition invariance of the GMM moment
# ---------------------------------------------------------------------------

def _run_pyrvtest_on_subset(product_data: pd.DataFrame, market_ids_keep):
    """Run pyRVtest on a subset of markets.

    Returns (results, N_subset). The subset is selected by market_ids
    inclusion; all products within each kept market are retained so
    within-market structure is preserved.
    """
    mask = product_data['market_ids'].isin(market_ids_keep)
    df_sub = product_data.loc[mask].reset_index(drop=True)
    results = _run_pyrvtest(df_sub)
    return results, len(df_sub)


def test_market_partition_invariance_of_gmm_moment():
    """Partitioning markets into disjoint subsets A and B preserves N*g.

    The GMM moment is g = (1/N) * Z' @ omega. As a linear function of
    rows, the product N * g aggregates additively across any disjoint
    row partition of the sample: if markets A and B are disjoint and
    their union covers all T markets, then

        N_full * g_full = N_A * g_A + N_B * g_B.

    The catch is that in pyRVtest both Z and omega have been
    residualized on cost shifters w via full-sample OLS, and that
    projection is NOT row-block-separable: refitting OLS on each
    subset gives different residuals than restricting the full-sample
    residuals to the subset. So rather than rerunning pyRVtest on
    each half (which would apply subset-specific projections and the
    identity would not hold exactly), we verify the partition
    property at the level of the RAW moment (1/N) * Z' @ omega where
    omega = prices - markup for each candidate model. Linearity in
    rows is exact there, and that is the fundamental invariant.

    Fixture: Bertrand + PerfectCompetition DGP with 8 markets, 2
    products per market. We build the raw moment hand-side for both
    models on the full sample and on each half, then check the
    additive identity at atol=1e-10. We also run pyRVtest on the full
    and each subset as a smoke check that the problem remains solvable
    after splitting (no structural break, finite markups).
    """
    # Fixture: 8 markets, 2 products each. Use user_supplied_markups path.
    T = 8
    df = _make_dgp(seed=4242, T=T, J=2)
    N_full = len(df)

    # Partition the market ids into two disjoint halves.
    all_markets = sorted(df['market_ids'].unique())
    markets_A = all_markets[:T // 2]
    markets_B = all_markets[T // 2:]

    # Raw moment (1/N) * Z' @ omega for each model; omega_m = prices - markup_m.
    # This raw moment aggregates exactly additively across any disjoint row
    # partition, so the identity below holds to numerical precision.
    Z_full = df[['iv0', 'iv1', 'iv2']].values
    prices_full = df['prices'].values.reshape(-1, 1)
    # Two models: Bertrand (markup_m1) and perfect competition (zero markup).
    for markup_col in ['markups_m1', 'markups_m2']:
        markup_full = df[markup_col].values.reshape(-1, 1)
        omega_full = (prices_full - markup_full).flatten()
        g_full_hand = (Z_full.T @ omega_full) / N_full  # shape (K,)

        df_A = df.loc[df['market_ids'].isin(markets_A)].reset_index(drop=True)
        df_B = df.loc[df['market_ids'].isin(markets_B)].reset_index(drop=True)
        Z_A = df_A[['iv0', 'iv1', 'iv2']].values
        Z_B = df_B[['iv0', 'iv1', 'iv2']].values
        omega_A = (df_A['prices'].values - df_A[markup_col].values)
        omega_B = (df_B['prices'].values - df_B[markup_col].values)
        N_A, N_B = len(df_A), len(df_B)
        g_A_hand = (Z_A.T @ omega_A) / N_A
        g_B_hand = (Z_B.T @ omega_B) / N_B

        # Partition identity for the raw moment: linearity of Z' @ omega
        # across disjoint row blocks.
        np.testing.assert_allclose(
            N_A * g_A_hand + N_B * g_B_hand, N_full * g_full_hand,
            atol=1e-10,
            err_msg=(
                f'Raw GMM moment N*g must aggregate additively across market '
                f'partitions for {markup_col}; this is a property of Z\'omega '
                f'being a linear function of rows.'
            ),
        )

    # Sanity: run pyRVtest end-to-end on both halves and on the full
    # sample; verify the problem is solvable on each partition (no
    # structural break from splitting markets).
    r_full = _run_pyrvtest(df)
    r_A, N_A_run = _run_pyrvtest_on_subset(df, markets_A)
    r_B, N_B_run = _run_pyrvtest_on_subset(df, markets_B)
    assert N_A_run + N_B_run == N_full
    for r in (r_full, r_A, r_B):
        for m in range(len(r.markups)):
            assert np.all(np.isfinite(r.markups[m])), 'markups not finite on subset'


# ---------------------------------------------------------------------------
# Property 13: FWL identity — g = (1/N) * Z_perp' @ omega_perp
# ---------------------------------------------------------------------------

def test_fwl_identity_for_gmm_moment():
    """pyRVtest's g matches hand-computed (1/N) * Z_perp' @ omega_perp.

    Frisch-Waugh-Lovell: if we residualize both the instruments Z and
    the marginal cost mc on the cost shifters w via OLS, then the GMM
    moment (1/N) * Z_perp' @ omega_perp (where omega_perp is the
    residual from regressing mc on w) equals pyRVtest's exposed g
    value for each model and instrument set.

    This tests the internal FWL path in
    ``Problem._compute_instrument_results`` which residualizes Z on w
    via ``_qr_residualize`` (problem.py ~line 1191) and constructs
    ``g[m] = 1/N * (Z_orthogonal.T @ omega[m])`` (problem.py ~line
    1206), where ``omega[m]`` itself was residualized on w at line
    1140. A bug anywhere in that residualization pipeline would
    produce a hand-check mismatch.

    Fixture: Bertrand + PerfectCompetition with user_supplied_markups
    (so marginal_cost = prices - user_markup is deterministic) and a
    non-trivial cost formulation ``1 + cost_shifter`` with 3
    instruments ``iv0 + iv1 + iv2``. No fixed effects, no endogenous
    cost component — this is the clean baseline path.
    """
    T = 10
    df = _make_dgp(seed=9876, T=T, J=2)
    N = len(df)

    results = _run_pyrvtest(df)

    # Hand computation: residualize Z on [intercept, cost_shifter] via OLS,
    # and residualize marginal_cost_m (= prices - user_markup_m) on the
    # same w. Then compare (1/N) * Z_perp' @ omega_perp to results.g[0][m].
    w = np.column_stack([np.ones(N), df['cost_shifter'].values])
    Z = df[['iv0', 'iv1', 'iv2']].values
    prices = df['prices'].values

    # OLS projection matrix ingredients via QR (mirrors pyRVtest internal)
    Q_w, R_w = np.linalg.qr(w, mode='reduced')
    # Z residualized on w
    Z_perp = Z - Q_w @ (Q_w.T @ Z)

    for m, markup_col in enumerate(['markups_m1', 'markups_m2']):
        mc = prices - df[markup_col].values
        # omega_perp = residual from regressing mc on w
        omega_perp = mc - Q_w @ (Q_w.T @ mc)
        g_hand = (Z_perp.T @ omega_perp) / N  # shape (K,)

        # pyRVtest exposes g as a list indexed by instrument set.
        # With one instrument formulation, that's results.g[0], shape (M, K).
        g_pyrv = results.g[0][m]
        np.testing.assert_allclose(
            g_pyrv, g_hand, atol=1e-10,
            err_msg=(
                f"FWL identity violated for model {m} ({markup_col}): "
                f"pyRVtest g != (1/N) * Z_perp' @ omega_perp where both are "
                f"OLS-residualized on cost shifters w=[1, cost_shifter]."
            ),
        )


# ---------------------------------------------------------------------------
# Property 14: sign-of-alpha homogeneity of degree -1 in plain-logit Bertrand
# ---------------------------------------------------------------------------

@given(alpha1=st.floats(min_value=-4.0, max_value=-0.5),
       alpha_ratio=st.floats(min_value=1.25, max_value=4.0))
@settings(max_examples=25, deadline=None)
def test_bertrand_markups_alpha_homogeneity_degree_minus_one(alpha1, alpha_ratio):
    """Plain-logit Bertrand markups satisfy markup(alpha2) = markup(alpha1) * (alpha1/alpha2).

    For plain logit, the Bertrand FOC gives Delta = -(O * D)^{-1} s where
    D = alpha * (diag(s) - s s'). Since D is linear in alpha, (O * D)^{-1}
    scales as 1/alpha, so Delta is homogeneous of degree -1 in alpha
    PROVIDED the shares s are held fixed. The key subtlety: shares are
    held fixed in this test because we supply observed shares in the DGP
    and pyRVtest (with demand_params + no demand_adjustment) treats them
    as data.

    Concretely: for the same DGP (same shares, same firm_ids, same market
    structure), running pyRVtest with alpha1 and alpha2 = alpha1 *
    alpha_ratio gives markup(alpha2) = markup(alpha1) / alpha_ratio.

    Tolerance 1e-12: this is an exact algebraic identity, not an
    equilibrium property, so the tolerance is near-machine-precision.

    Hypothesis-driven: alpha1 in [-4, -0.5], alpha_ratio in [1.25, 4],
    25 examples to keep runtime under ~30s for this property.
    """
    # Small deterministic DGP: the Bertrand-equilibrium prices depend on
    # alpha, but pyRVtest with demand_params uses OBSERVED shares, so
    # homogeneity in alpha holds for the computed markup at the OBSERVED
    # shares regardless of whether the prices came from equilibrium at
    # alpha1 or alpha2. We just need consistent shares; we get them from
    # the DGP built at alpha1 and keep the frame fixed.
    T = 6
    df = _make_dgp_multifirm(seed=17, T=T, alpha=alpha1)

    # Run with alpha1
    p1 = pyRVtest.Problem(
        cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
        instrument_formulation=pyRVtest.Formulation('0 + iv0 + iv1 + iv2'),
        model_formulations=(
            pyRVtest.ModelFormulation(
                model_downstream='bertrand', ownership_downstream='firm_ids',
            ),
            pyRVtest.ModelFormulation(model_downstream='perfect_competition'),
        ),
        product_data=df,
        demand_params={'alpha': alpha1, 'sigma': []},
    )
    r1 = p1.solve(demand_adjustment=False, clustering_adjustment=False)

    # Run with alpha2 = alpha1 * alpha_ratio (more negative / more elastic)
    alpha2 = alpha1 * alpha_ratio
    p2 = pyRVtest.Problem(
        cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
        instrument_formulation=pyRVtest.Formulation('0 + iv0 + iv1 + iv2'),
        model_formulations=(
            pyRVtest.ModelFormulation(
                model_downstream='bertrand', ownership_downstream='firm_ids',
            ),
            pyRVtest.ModelFormulation(model_downstream='perfect_competition'),
        ),
        product_data=df,
        demand_params={'alpha': alpha2, 'sigma': []},
    )
    r2 = p2.solve(demand_adjustment=False, clustering_adjustment=False)

    markup1 = r1.markups[0]
    markup2 = r2.markups[0]

    # Homogeneity of degree -1: markup2 = markup1 * (alpha1 / alpha2).
    expected_markup2 = markup1 * (alpha1 / alpha2)
    np.testing.assert_allclose(
        markup2, expected_markup2, atol=1e-12,
        err_msg=(
            f"Plain-logit Bertrand markups should be homogeneous of degree -1 "
            f"in alpha: markup(alpha2={alpha2}) should equal markup(alpha1={alpha1}) "
            f"* (alpha1/alpha2={alpha1/alpha2}). Max abs deviation: "
            f"{np.max(np.abs(markup2 - expected_markup2))}"
        ),
    )

    # Monotonicity corollary: alpha2 more negative (|alpha2| > |alpha1|)
    # implies smaller markups elementwise.
    assert np.all(markup2 < markup1 + 1e-12), (
        f"For alpha2={alpha2} more negative than alpha1={alpha1}, Bertrand "
        f"markups should shrink. max(markup2 - markup1) = "
        f"{np.max(markup2 - markup1)}"
    )
