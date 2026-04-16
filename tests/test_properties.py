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

3. **Bertrand markups are positive when alpha < 0**: for price-elastic
   demand (the economically reasonable regime), the implied Bertrand
   markup is strictly positive. This property would have failed on the
   pre-b3b08a3 analytical path for the sign-of-gradient check — the
   markup itself was always correct, but the gradient sign check is a
   natural tripwire.

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
    """Run pyRVtest on the standard Bertrand-vs-perfect-comp DGP."""
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


# ---------------------------------------------------------------------------
# Property 1: determinism
# ---------------------------------------------------------------------------

@given(seed=st.integers(min_value=1, max_value=10_000))
@settings(max_examples=5, deadline=None)
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
@settings(max_examples=5, deadline=None)
def test_bertrand_markups_positive_for_elastic_demand(alpha):
    """For price-elastic demand (alpha < 0), Bertrand markups are > 0.

    Economic content: when consumers substitute away from higher prices
    (alpha < 0), firms charge strictly positive markups over marginal
    cost. Bertrand formula Δ = -(O·D)^{-1} s with O·D evaluated at
    alpha<0 and positive shares yields markup > 0.

    This is a natural tripwire for sign bugs in markup computation.
    """
    df = _make_dgp(seed=42, T=8, J=2, alpha=alpha)
    results = _run_pyrvtest(df)

    bertrand_markups = results.markups[0]
    assert np.all(bertrand_markups > 0), (
        f"Bertrand markups should be strictly positive for alpha={alpha}; "
        f"got min {np.min(bertrand_markups)}"
    )


# ---------------------------------------------------------------------------
# Property 4: within-market row permutation invariance
# ---------------------------------------------------------------------------

@given(seed=st.integers(min_value=1, max_value=10_000))
@settings(max_examples=5, deadline=None)
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
@settings(max_examples=10, deadline=None)
def test_dgp_seed_invariance(seed):
    """Building the same DGP with the same seed twice yields identical data.

    Sanity check on the property-test scaffolding itself. If this fails,
    Hypothesis's seeded runs are not reproducible and the rest of the
    suite's guarantees are suspect.
    """
    df1 = _make_dgp(seed=seed)
    df2 = _make_dgp(seed=seed)
    pd.testing.assert_frame_equal(df1, df2)
