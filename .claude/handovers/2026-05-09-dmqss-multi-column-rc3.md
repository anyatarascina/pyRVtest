# Handover — 2026-05-08/09 — DMQSW Phase 5 + DMQSS A.4 multi-column (rc3 tagged)

**Branch:** `v0.4-refactor` at `607a909` (pushed; tagged `v0.4.0rc3`).
**Plan:** [`.claude/plans/multi-column-endog-cost-transforms.md`](../plans/multi-column-endog-cost-transforms.md)

## Session goals

Landed two distinct work streams.

**Stream A** — finish DMQSW Phase 5 (documentation pass for the
pre-existing DMQSW pass-through diagnostic suite that shipped through
rc2). Also closed two latent issues surfaced during the docs review:
the stale `endogenous_cost_component + demand_adjustment` caveat and
the missing K_inst > K_endog validation gate.

**Stream B** — implement Phases 1–5 of a new plan
([`.claude/plans/multi-column-endog-cost-transforms.md`](../plans/multi-column-endog-cost-transforms.md))
generalizing endogenous-cost machinery to the K_endog > 1 case and
fixing the long-standing log + demand_adjustment hole. Paper-supported
per DMQSS Appendix A.4 (identification with named examples) and B
(variance derivation generalized to vector γ). Tagged `v0.4.0rc3`
when complete.

## What landed

### Stream A — Phase 5 docs + cleanup (commits prior to rc3)

- **Phase 5 documentation** (`ac0562a`, merged via `18ea86b`).
  Three new method docstrings as the single source of truth for
  methodology text (`passthrough_summary`, `instrument_channels`,
  `reliability_summary`). math.rst gains a worst-rho CV bound section
  and a "Pass-through framework for instrument relevance" section
  with the chain-rule indirect/direct decomposition and FWL
  conditional-regression derivation. advanced_features.rst Pass-through
  section rewritten as a three-method walkthrough on the synthetic
  example with captured outputs (the (Cournot, PerfectCompetition)
  `offdiag_ratio ≈ 1.3e-9` degeneracy demonstration). FAQ entries,
  migration notes, README headline-features bullet, tutorial pointer,
  CHANGELOG entry. Methodology footers in printed output now reference
  `passthrough_summary()` / `instrument_channels()` docstrings.

- **`#3` stale-doc cleanup** (`9cceefd`). FAQ + advanced_features note
  rewritten — `endogenous_cost_component + demand_adjustment=True` is
  already correctly implemented (step 4d/4e), the docs just hadn't
  caught up. Stale xfail-strict regression test
  `test_memo1_section42_adjustment_interaction.py` deleted; correctness
  coverage moved to the existing cross-path parity test.

- **`K_inst > K_endog` validation gate** (`c84f7b6`, merged via
  `07cb152`). Replaces a silent `ZeroDivisionError` in
  `test_engine.compute` with a clear `ValueError` from
  `_validate_solve_args`. Names every offending instrument set so a
  user with multiple bundles only needs to fix the bad ones. Cites
  DMQSS Remark 1. Forward-compatible with the multi-column refactor:
  `K_endog` derived from `len(...)` semantics. Four tests in
  `TestKInstValidationWithEndogenousCost`.

- **Plan file** (`c851958`). `.claude/plans/multi-column-endog-cost-transforms.md`
  consolidates the DMQSS A.4 multi-column work as a 5-phase plan with
  the rank-(K+1) decomposition argument, the parametric-linearity
  argument for why the moment factors cleanly, and the coauthor ask.

### Stream B — DMQSS A.4 multi-column generalization (Phases 1–5)

Branch: `feat/multi-column-endog` (retained locally), merged into
`v0.4-refactor` as `b138e0d`. 6 commits:

- **Phase 1: Multi-column `endogenous_cost_component` core** (`bf971b8`).
  - `Optional[str]` widens to `Optional[Union[str, Sequence[str]]]` on
    `Problem.__init__`. Validation accepts non-empty list of distinct
    str column names; rejects empty list, duplicates, non-str entries,
    and missing columns with the canonical Expected/Received/Fix message
    structure.
  - Internal `Problem._endogenous_cost_columns` holds the normalized
    tuple of names; every consumer of
    `problem.endogenous_cost_component` uses this for uniform
    handling.
  - `iv_correct` (endogenous_cost.py) vectorized: 2SLS first stage and
    second-stage gamma block of length K_endog;
    `mc_correction[m] = -endog_cols_raw @ gamma_m` matrix-vector form.
  - `_compute_gamma_gradient` (demand_adjustment.py) finite-diffs the
    last K_endog elements of `cost_param`; output shape preserves
    `(M, n_theta)` for `K_endog == 1` and widens to
    `(M, n_theta, K_endog)` for `K > 1`.
  - `compute_instrument_results` (test_engine.py): `K_effective =
    K_inst − K_endog`; Appendix B influence-function correction
    rewritten with `Lambda_q ∈ ℝ^{K × K_endog}` and the variance
    contraction generalized. K_endog = 1 case bit-identical to prior
    (verified algebraically and in tests).
  - Demand-adjustment block reshapes `gradient_gamma[m]` to
    `(K_endog, n_theta)` so `endog_col_for_grad @ gg_m_T` produces
    `(N, n_theta)` uniformly.
  - `orthogonalize.residualize` and
    `_residualize_grad_on_cost_shifters` updated with set-based
    exog filtering.
  - `endogenous_cost_coefficient` storage stays `(L, M)` scalar for
    K_endog = 1 (back-compat) and widens to `(L, M, K_endog)` for K>1.
  - 6 new tests in `TestMultiColumnEndogenousCostRoundTrip`.

- **Phase 2: log + demand_adjustment chain rule** (`6af5c88`).
  - Drop the `not demand_adjustment` gate on the log transform.
  - Save `marginal_cost_level` (pre-log) for the chain rule.
  - After `compute_demand_adjustment` returns, when
    `costs_type='log'`, rescale `gradient_markups[m]` elementwise by
    `1/marginal_cost_level[m]`. The endogenous-cost gamma channel
    does not need rescaling: gamma is already in the LOG cost
    regression so `∂ω_m/∂γ = -q` in both linear and log cases.
  - Removed the stale "costs_type='log' is ignored when
    demand_adjustment=True" UserWarning.
  - 1 new test in `test_log_cost_demand_adjustment_runs_end_to_end`
    (cross-cost-transform parity on the `_build_scale_dgp` fixture).

- **Phase 3: `instrument_channels` z^e residualization** (`4861907`).
  - When `endogenous_cost_component` is set, the data-side regression
    and FWL partialling project on `(g(q̃), w_exog)` rather than raw
    `(q, w_exog)`. This is DMQSS Appendix B z^e.
  - The first stage uses the **FULL declared instrument set** (union
    over all L test bundles) plus w_exog, NOT the single diagnostic
    column. Using the diagnostic column alone would put it in the span
    of `(q_tilde, w_exog)` and degenerate the residualization to zero
    (verified: prior implementation hit this and was caught by tests).
  - Methodology footer updated conditionally. Final wording (after
    coauthor review feedback in `a73e158`):

    > Data-side ‖dp_0/dz‖_obs and direct channel β_m use z residualized
    > on (g(q̃), w_exog) — DMQSS z^e — which simultaneously reflects the
    > instrument-relevance condition (DMQSW) and the economic
    > distinctness condition (DMQSS Appendix A.4). Magnitudes are
    > finite-sample sample-regression estimates; small-but-nonzero
    > values may reflect noise rather than population identifying
    > variation.

  - 3 new tests in `TestZeResidualizationUnderNonConstantMC`.

- **Phase 4: documentation** (rolled into `4861907` commit).
  - math.rst: new "Pass-through diagnostic under non-constant
    marginal cost" section stating the rank-(K+1) condition and its
    decomposition into first-stage rank K + Cov(z^e, f-difference).
  - advanced_features.rst: new "Multiple endogenous variables"
    subsection with the q + q² and scale + scope examples.
  - faq.rst: `costs_type='log' + demand_adjustment` and "What if I
    have q + q² or scale + scope?" entries.
  - migrating_to_v0.4.rst: "Multi-endogenous-variable cost regressions"
    and `costs_type='log' + demand_adjustment=True` subsections.
  - README.rst: headline-features bullet updated.
  - CHANGELOG.md: "Added (Phase 6: multi-column endogenous cost +
    log-cost demand-adjustment + z^e diagnostic)" subsection.
  - .claude/plans/v0.5-followups.md: items 1 (log + demand_adjustment)
    and 3 (demand_adjustment + endogenous_cost) marked CLOSED.

- **Phase 5: coauthor memo** (`ad0e21b`).
  `.claude/handovers/MEMO_dmqss_appendix_b_vectorization_2026-05-08.md`
  drafts an ask to Marco / Lorenzo / Daniel / Mikkel for a one-
  paragraph remark in DMQSS Appendix B explicitly stating the
  vector-γ generalization of Lemma B.1 and the rank-(K+1)
  decomposition. Suggested wording, list of code locations, asks,
  and timing options. **Status: DRAFT — to be reviewed before
  sending.**

- **Methodology footer wording tighten** (`a73e158`). Per coauthor
  review feedback, three changes: use "instrument-relevance condition
  (DMQSW)" instead of "Dearing pass-through-feature condition" or
  "Dearing condition"; cite as DMQSW / DMQSS rather than "Dearing et
  al. (2026)"; replace "the diagnostic collapses [conditions] into
  one" with "simultaneously reflects" and add a finite-sample caveat
  to avoid overclaiming that the rank condition holds from a sample
  magnitude.

- **Version bump** (`607a909`). `__version__` and setup.py from
  '0.4.0rc1' to '0.4.0rc3'.

- **`v0.4.0rc3` tag** (annotated). Pushed to origin.

## Test progression

| State | Passing |
|---|---|
| Pre-Phase-5 baseline (rc2) | 767 |
| + Phase 5 docs | 767 (no test changes) |
| + #3 stale-doc cleanup | 767 (1 stale xfail-strict deleted) |
| + K_inst > K_endog gate | 771 (+4) |
| + Multi-column endog (Phase 1) | 777 (+6) |
| + log + demand_adjustment (Phase 2) | 778 (+1) |
| + z^e diagnostic (Phase 3) | **781 (+3)** |

mypy `--strict` clean throughout. sphinx `-W` clean modulo two
pre-existing unrelated docstring warnings (`Models` recarray and
`MixCournotBertrand`).

## What's left

### Path to v0.4.0 final

1. **Coauthor confirmation on Appendix B remark** (Phase 5 ask).
   Send `.claude/handovers/MEMO_dmqss_appendix_b_vectorization_2026-05-08.md`
   to Marco / Mikkel / Lorenzo / Daniel after Chris reviews. The
   suggested paragraph captures the vector-γ generalization and the
   rank-(K+1) decomposition we already implement; coauthors confirm
   or correct the math.

2. **Wait period** (~1 week typical coauthor turnaround).

3. **Tag `v0.4.0`** when the paper-side remark is in place. Bump
   `__version__` from `0.4.0rc3` to `0.4.0`. Update CHANGELOG to
   convert `[0.4.0] — unreleased` to a proper release header.

### Optional / deferred

- **Phase 6 of the original DMQSW plan**:
  `docs/notebooks/replication_DMQSW_marijuana.py`. Not implemented
  in this session. Could ship in v0.4.0 final or v0.4.1 patch.

- **v0.5-followups items still open** (`v0.5-followups.md`):
  - Item 2: no-cost-formulation support (`Problem.__init__`).
  - Item 4: §4.1 F-stat rank-adjustment regression test rewrite.
  - Item 5: F-stat collapse on identically-zero-markup candidates.

## Notable design decisions

- **Multi-column endog as a list of column names; transforms are
  user-precomputed.** API stays declarative — user provides `q_sq`,
  `log_q`, etc. as observed columns. Package never evaluates
  user-supplied callables for the cost regression's basis. The cost
  transform `f` (linear / log) is curated; no user-callable interface
  in v0.4.

- **z^e residualization fires only under non-constant MC.** Constant-
  MC users see no behavior change in `instrument_channels`. The
  unified diagnostic is opt-in via `endogenous_cost_component`.

- **First stage uses the FULL declared IV bundle, not the diagnostic
  column.** Using only the diagnostic column collapses the
  residualization. Matches DMQSS's construction: q_tilde is the
  projection of g(q^p) onto the column space of all available
  exogenous variation.

- **K_endog = 1 path is bit-identical to pre-refactor.** Verified
  via existing snapshot suite (706+ tests including
  `analytical_base`, `analytical_scale`, `first_stage_*`,
  `nested_logit_vertical`). Multi-column users opt in by passing a
  list; everyone else sees no change.

- **Methodology footer is the user-facing entry point to the
  rank-(K+1) decomposition argument.** Footer says what the magnitude
  measures, finite-sample caveat, points at docstring + math.rst for
  the formal statement.

## Pickup notes for next session

1. **Read this handover** + the plan file at
   `.claude/plans/multi-column-endog-cost-transforms.md`.

2. **Verify env**: `python3 -c "import pyRVtest; print(pyRVtest.__version__)"`
   should print `0.4.0rc3`. Run
   `python3 -m pytest tests/ -q --tb=line` to confirm 781 passing on
   pickup.

3. **Decide on coauthor memo timing**. The memo at
   `.claude/handovers/MEMO_dmqss_appendix_b_vectorization_2026-05-08.md`
   is DRAFT. Chris reviews; send to Marco / Mikkel / Lorenzo / Daniel
   (or have Chris send directly). Or: tag `v0.4.0` first with
   "vector-γ generalization pending paper remark" CHANGELOG note and
   add the cross-reference in v0.4.1.

4. **Don't relaunch agents for completed phase work** — Phases 1–4
   are merged. The plan file's Phase 5 is the coauthor coordination
   step (external).

## File tree pointers

- Plan: [`.claude/plans/multi-column-endog-cost-transforms.md`](../plans/multi-column-endog-cost-transforms.md)
- Coauthor memo: [`.claude/handovers/MEMO_dmqss_appendix_b_vectorization_2026-05-08.md`](MEMO_dmqss_appendix_b_vectorization_2026-05-08.md)
- Multi-column iv_correct: [`pyRVtest/solve/endogenous_cost.py`](../../pyRVtest/solve/endogenous_cost.py)
- Vectorized gamma gradient:
  [`pyRVtest/solve/demand_adjustment.py`](../../pyRVtest/solve/demand_adjustment.py)
  (`_compute_gamma_gradient`)
- Vector-γ Appendix B variance:
  [`pyRVtest/solve/test_engine.py`](../../pyRVtest/solve/test_engine.py)
  (Lambda_q computation + term1/term2 contractions)
- z^e residualization:
  [`pyRVtest/solve/passthrough.py`](../../pyRVtest/solve/passthrough.py)
  (`compute_instrument_channels`,
  `_build_instrument_channels_methodology_line`)
- K_inst > K_endog validation gate:
  [`pyRVtest/problem.py`](../../pyRVtest/problem.py)
  (`_validate_solve_args`)
- Multi-column tests: [`tests/test_analytical.py`](../../tests/test_analytical.py)
  (`TestMultiColumnEndogenousCostRoundTrip`,
  `TestKInstValidationWithEndogenousCost`)
- Log + demand_adjustment test:
  [`tests/test_demand_adjustment.py`](../../tests/test_demand_adjustment.py)
  (`test_log_cost_demand_adjustment_runs_end_to_end`)
- z^e diagnostic tests:
  [`tests/test_instrument_channels.py`](../../tests/test_instrument_channels.py)
  (`TestZeResidualizationUnderNonConstantMC`)
- Math: [`docs/math.rst`](../../docs/math.rst)
  ("Pass-through diagnostic under non-constant marginal cost"
  section)
- User-facing walkthrough: [`docs/advanced_features.rst`](../../docs/advanced_features.rst)
  ("Multiple endogenous variables" subsection)

## Worktrees

- `/Users/christophersullivan/Dropbox/Economics/claude/pyrvtest` — main
  checkout, on `v0.4-refactor` at `607a909`.
- `/Users/christophersullivan/Dropbox/Economics/claude/pyrvtest/.claude/worktrees/goofy-chaplygin-b18b51` —
  this session's worktree, on `feat/multi-column-endog` at `a73e158`
  (one commit behind v0.4-refactor — does not have the version bump).
  Reachable from v0.4-refactor via the merge commit `b138e0d`. Can be
  deleted whenever.

## Coauthor coordination status

- **DMQSW (Dearing, Magnolfi, Quint, Sullivan, Waldfogel)** — pre-
  test framework paper. No coordination needed for this session's
  work; rc2 already shipped the diagnostic suite that paper
  motivates. The methodology footer cites DMQSW for the
  instrument-relevance condition.

- **DMQSS (Duarte, Magnolfi, Quint, Sølvsten, Sullivan)** — non-
  constant-cost paper. **DRAFT memo asks for a one-paragraph remark
  in Appendix B** explicitly stating the vector-γ generalization of
  Lemma B.1 and the rank-(K+1) decomposition. The pyRVtest
  implementation is paper-supported per A.4 (identification with
  named examples for q + q² and scale + scope) and the framing of
  Appendix B (arbitrary cost transforms), but Lemma B.1 as written
  is for K_endog = 1 with scalar γ_m. Generalization is uncontroversial
  multi-endog IV but not stated. Asking for it before tagging v0.4.0
  final keeps the package and paper aligned.
