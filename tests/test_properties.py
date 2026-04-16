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
