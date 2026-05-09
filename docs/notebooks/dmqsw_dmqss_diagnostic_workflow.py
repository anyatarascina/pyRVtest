# DMQSW + DMQSS Diagnostic Workflow
#
# Demonstrates the v0.4 final pre-test diagnostic suite end-to-end:
#
# 1. Constant marginal cost (DMQSW): the (Cournot, PerfectCompetition)
#    pair under logit demand has structurally degenerate pass-through
#    (offdiag_ratio = 0). passthrough_summary identifies this ex-ante;
#    reliability_summary confirms it post-solve (F = 0); switching to a
#    different IV type breaks the degeneracy.
#
# 2. Non-constant marginal cost with multi-column endogenous variables
#    (DMQSS A.4): quadratic cost c = γ_1 q + γ_2 q² requires K_endog = 2
#    endogenous columns and K_inst > K_endog instruments. The unified
#    diagnostic in instrument_channels uses DMQSS Appendix B z^e
#    residualization and reflects both the DMQSW instrument-relevance
#    condition and the DMQSS economic distinctness condition in one
#    magnitude.
#
# Run with: python3 dmqsw_dmqss_diagnostic_workflow.py

import os
import sys
import warnings

import numpy as np
import pandas as pd

# Use the in-source pyRVtest (worktree / repo checkout) rather than any
# stale site-packages install. Matches the convention in the other
# scripts in this directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

warnings.filterwarnings('ignore')

import pyRVtest

print("=" * 78)
print("PART 1 — DMQSW pre-test framework on the synthetic example (constant MC)")
print("=" * 78)

# %% Load the shipped synthetic dataset (3000 markets, duopoly, logit DGP, PC truth).
data = pyRVtest.data.load_example()
print(f"Loaded {len(data)} product-market observations from "
      f"{data['market_ids'].nunique()} markets.")

# %% Build the Problem with four candidate models. DON'T solve yet — pass-through
#    framework reasoning happens BEFORE running the test, so post-selection
#    inference issues don't contaminate the choice of IV bundle.
problem = pyRVtest.Problem(
    cost_formulation=pyRVtest.Formulation('1 + z1 + z2'),
    instrument_formulation=pyRVtest.Formulation('0 + rival_z1 + rival_z2'),
    models=[
        pyRVtest.Bertrand(ownership='firm_ids'),
        pyRVtest.Cournot(ownership='firm_ids'),
        pyRVtest.Monopoly(),
        pyRVtest.PerfectCompetition(),
    ],
    product_data=data,
    demand_params={
        'estimate': 'logit',
        'formulation_X': pyRVtest.Formulation('1 + x1'),
        'formulation_Z': pyRVtest.Formulation('0 + z1'),
    },
)

# %% Pre-solve framework view: γ-free pair-by-pair structural distances.
#    Four DMQSW-keyed metrics target four primitive instrument types:
#      * offdiag_ratio (Remark 1): rival cost shifters
#      * full_pass     (Remark 2/4): own+rival cost or product chars under linear-index demand
#      * row_sum       (Remark 5):  per-unit tax
#      * level_adj     (Remark 5):  ad-valorem tax
print()
print(problem.passthrough_summary())

# %% Read off the structural degeneracy. The (Cournot, PerfectCompetition) row
#    has offdiag_ratio ≈ 1.3e-9 — numerical zero. Under logit demand both
#    candidates have diagonal pass-through, so rival cost shifters cannot
#    distinguish them at any sample size. The other three metrics are
#    nonzero, so own+rival cost / product characteristics / per-unit taxes /
#    ad-valorem taxes WOULD provide identifying variation.

# %% Solve the problem and read the post-solve empirical view.
results = problem.solve(demand_adjustment=False)

print()
print("Post-solve reliability_summary (subset of columns):")
df = results.reliability_summary()
keep = ['model_i_label', 'model_j_label', 'F', 'rho_squared', 'strongest_claim_size']
print(df[keep].to_string(index=False))

# %% Cross-read the framework view (passthrough_summary) and the empirical
#    view (reliability_summary). For (Cournot, PerfectCompetition):
#      * Pre-solve:  offdiag_ratio ≈ 0
#      * Post-solve: F ≈ 0
#    The two agree: the pair is structurally degenerate under rival cost
#    shifters. Switching to a different IV type — say a per-unit tax instead
#    — would distinguish the pair (level_adj is nonzero).

# %% Per-pair channel decomposition for one IV column.
print()
print(results.instrument_channels(column='rival_z2', instrument='rival_cost'))


# %% --------------------------------------------------------------------
print()
print("=" * 78)
print("PART 2 — Multi-column endogenous cost (DMQSS A.4): quadratic cost")
print("=" * 78)

# %% Construct a synthetic dataset with non-constant marginal cost. We'll
#    take the existing example data, add a "quantity" column derived from
#    market_size * shares, and a derived q_sq = q² column. Both q and q_sq
#    are endogenous (q_sq is a deterministic function of q, but their
#    correlations with instruments are not proportional in general — see
#    DMQSS A.4 paragraph on quadratic cost).
data_qq = data.copy()
market_size = 1000.0
data_qq['q'] = market_size * data_qq['shares']
data_qq['q_sq'] = data_qq['q'] ** 2

# DMQSS A.4 quadratic-cost requires at least K_endog + 1 = 3 testing
# instruments. The shipped example has rival_z1 and rival_z2; we add
# rival_z1 * rival_z2 as a third (mildly nonlinear in the original
# instruments — generates variation that's economically distinct enough to
# satisfy the rank condition on this fixture).
data_qq['rival_z12'] = data_qq['rival_z1'] * data_qq['rival_z2']

# %% Build the multi-column endog Problem. K_endog = 2, K_inst = 3.
problem_qq = pyRVtest.Problem(
    cost_formulation=pyRVtest.Formulation('1 + z1 + q + q_sq'),
    instrument_formulation=pyRVtest.Formulation(
        '0 + rival_z1 + rival_z2 + rival_z12'
    ),
    models=[
        pyRVtest.Bertrand(ownership='firm_ids'),
        pyRVtest.PerfectCompetition(),
    ],
    product_data=data_qq,
    demand_params={
        'estimate': 'logit',
        'formulation_X': pyRVtest.Formulation('1 + x1'),
        'formulation_Z': pyRVtest.Formulation('0 + z1'),
    },
    endogenous_cost_component=['q', 'q_sq'],   # multi-column!
)

print(f"\nendog cols normalized to: {problem_qq._endogenous_cost_columns}")

# %% Solve. The IV correction's first stage simultaneously instruments q
#    and q_sq using all three IV columns; the second stage estimates a
#    length-K_endog gamma vector per candidate.
results_qq = problem_qq.solve(demand_adjustment=False)

print()
print("Per-candidate gamma vector (γ_1 = coefficient on q, γ_2 = coefficient on q²):")
for m, label in enumerate(['Bertrand', 'PerfectCompetition']):
    gamma_block = np.asarray(results_qq.endogenous_cost_coefficient[0, m])
    print(f"  {label:20s}  γ = {gamma_block}")

# %% Post-solve diagnostics. Under non-constant MC, instrument_channels uses
#    DMQSS Appendix B z^e residualization automatically (the data-side
#    regression projects on (q̃, w_exog) instead of raw (q, w_exog)). The
#    methodology footer documents this, and the magnitude of dp_0/dz_obs +
#    direct β_m + structural ‖P_m^{-1} - P_m'^{-1}‖_F simultaneously
#    reflects the DMQSW instrument-relevance condition and the DMQSS
#    economic distinctness condition.
print()
print(results_qq.instrument_channels(column='rival_z2', instrument='rival_cost'))


# %% --------------------------------------------------------------------
print()
print("=" * 78)
print("PART 3 — Multi-column endogenous cost: scale + scope (own + rival q)")
print("=" * 78)

# %% Same synthetic example, but the cost regression now includes own
#    quantity AND a constructed "rival quantity sum" (a stand-in for the
#    DMQSS scale + scope example, where Q_minus is total production by the
#    same firm on the same platform). Both q and rival_q are endogenous.
#
#    Note: DMQSS A.4 states this case in log-log form
#    (log(c) = γ_1 log(q) + γ_2 log(Q⁻) + w'τ + ω), and the multi-column
#    code path supports costs_type='log' identically — the API is the same.
#    Here we demo with linear cost on the synthetic example because Bertrand
#    on this fixture produces some implied markups that exceed price (the
#    in-package logit alpha estimate gives Bertrand markup ~ 1/(α(1-s)),
#    which inflates when shares are small), making the log transform
#    undefined. The multi-column machinery is identical either way; for a
#    log-cost example with controlled markups see
#    tests/test_log_cost_demand_adjustment_runs_end_to_end.
data_ss = data.copy()
data_ss['q'] = market_size * data_ss['shares']
# Rival-firm quantity sum within market.
rival_q_sum = []
for mid in data_ss['market_ids'].unique():
    market = data_ss[data_ss['market_ids'] == mid]
    total_q = market['q'].sum()
    for j, row in market.iterrows():
        rival_q_sum.append(total_q - row['q'])
data_ss['rival_q'] = rival_q_sum
data_ss['rival_z12'] = data_ss['rival_z1'] * data_ss['rival_z2']

# %% Build the scale + scope Problem.
problem_ss = pyRVtest.Problem(
    cost_formulation=pyRVtest.Formulation('1 + z1 + q + rival_q'),
    instrument_formulation=pyRVtest.Formulation(
        '0 + rival_z1 + rival_z2 + rival_z12'
    ),
    models=[
        pyRVtest.Bertrand(ownership='firm_ids'),
        pyRVtest.PerfectCompetition(),
    ],
    product_data=data_ss,
    demand_params={
        'estimate': 'logit',
        'formulation_X': pyRVtest.Formulation('1 + x1'),
        'formulation_Z': pyRVtest.Formulation('0 + z1'),
    },
    endogenous_cost_component=['q', 'rival_q'],
)

results_ss = problem_ss.solve(demand_adjustment=False)

print()
print("Scale + scope gamma estimates (γ_1 = q coefficient, γ_2 = rival_q coefficient):")
for m, label in enumerate(['Bertrand', 'PerfectCompetition']):
    gamma_block = np.asarray(results_ss.endogenous_cost_coefficient[0, m])
    print(f"  {label:20s}  γ = {gamma_block}")

# %% --------------------------------------------------------------------
print()
print("=" * 78)
print("Summary")
print("=" * 78)
print("""
- DMQSW passthrough_summary identifies structurally degenerate
  candidate-IV combinations BEFORE solving. Under logit demand,
  (Cournot, PerfectCompetition).offdiag_ratio = 0 — rival cost shifters
  cannot distinguish this pair at any sample size.

- DMSS reliability_summary confirms the structural finding empirically
  post-solve: F ≈ 0 for the same pair.

- DMQSS multi-column endogenous_cost_component handles q + q² (quadratic
  cost) and scale + scope. The IV correction's first stage instruments
  all K_endog columns jointly; the second stage produces a length-K_endog
  gamma vector. K_inst > K_endog is enforced.

- Under non-constant MC, instrument_channels uses DMQSS Appendix B z^e
  residualization automatically. A single magnitude simultaneously
  reflects the DMQSW instrument-relevance condition and the DMQSS
  economic distinctness condition.

See docs/advanced_features.rst, docs/math.rst, and docs/faq.rst for the
full pre-test framework reasoning.
""")
