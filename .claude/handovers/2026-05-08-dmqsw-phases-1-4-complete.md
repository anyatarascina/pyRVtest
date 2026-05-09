# Handover — 2026-05-08 — DMQSW pass-through diagnostic Phases 1–4 complete (rc2 tagged)

**Branch:** `v0.4-refactor` at `04bc73e` (pushed; tagged `v0.4.0rc2`).
**Plan:** [`.claude/plans/dmqsw-passthrough-diagnostic.md`](../plans/dmqsw-passthrough-diagnostic.md)

## Session goals

Implement the v0.4 DMQSW pass-through diagnostic suite per the plan
written at session start. Did Phases 1, 2, 3, 4 + the cross-validation
test that quantifies approach-A-vs-B agreement. Phase 5 (docs) and
Phase 6 (optional replication) remain.

## What landed

### Phase 4 — rc1 → final cleanup (`ce56a3b` → merged)

Mechanical renames and removals between v0.4 rc1 and v0.4 final. No
deprecation aliases since rc1 was internal/coauthor-only.

- `F_reliability_summary` → `reliability_summary` on `ProblemResults`.
- Removed `F_se` / `F_ci_low` / `F_ci_high` (asymptotic SE/CI not part of
  canonical DMSS reliability; computation removed in test_engine.py too).
- Removed `verdict` column from `reliability_summary` DataFrame (kept
  the internal attribute for `print(results)` symbol annotations).
- CV columns reshaped to (axis, level) grid: `size_cv_075` /
  `size_cv_100` / `size_cv_125`, `power_cv_050` / `power_cv_075` /
  `power_cv_095`. Plus empirical-rho variants `*_emp`.
- Removed `passthrough_comparison` method (superseded by
  `passthrough_summary` in Phase 2). `_PASSTHROUGH_METRICS` and
  `_passthrough_distance` helpers gone.
- `RELIABILITY_CI_LEVEL` constant removed.
- `migrating_to_v0.4.rst` updated; CHANGELOG split into `[0.4.0rc1]` /
  `[0.4.0]`.

### Phase 1 — Numerical pass-through for all conducts (`6550609` → merged)

Generalized the Vertical-only `build_passthrough` to dispatch across
every conduct class.

- `compute_passthrough_numerical(conduct_model, ownership, response,
  hessian, shares, markups, delta=1e-7)` core: central-difference
  perturbation through the markup function. Linearly perturbs `(D, s)`
  by their first-order responses `(δH[:,:,k], δD[:,k])`, evaluates the
  candidate's markup at the perturbed demand state, finite-differences,
  inverts `I − ∂Δ/∂p`. No equilibrium re-solve.
- `build_passthrough` dispatches: Vertical → Villas-Boas analytical fast
  path (existing `_construct_passthrough_from_hessian`, unchanged);
  trivial conducts (PC, ConstantMarkup, UserSuppliedMarkups,
  RuleOfThumb) → identity / `φI` short-circuit; everything else →
  numerical core.
- `docs/math.rst` gained "Pass-through by conduct class" section with
  per-conduct analytical references.
- `tests/test_passthrough_numerical.py` (18 tests): trivial closed-form
  bit-exact agreement, paper Example 2 hand-derived 2×2 logit cases
  (Bertrand off-diag nonzero; Cournot diagonal — the headline DMQSW
  result), smoke test on synthetic example.

**Bug fixes I had to apply** after the agent's initial pass: (a) safe
model-field membership check that handles both recarray and
dict-shaped `problem.models`; (b) early short-circuit for trivial
conducts in the per-market loop so PC etc. don't trip on missing
`ownership_downstream`; (c) seven mypy-strict fixes (Dict generic
params, np.asarray casts, type:ignore on legacy untyped function calls).

### Phase 1 cross-validation (`2cd321b` → merged; tightened to atol=1e-7 in `9e8fc37`)

Self-coded random-coefficient logit demand (Gauss-Hermite quadrature)
+ scipy.optimize.root fixed-point equilibrium solver. Confirms approach
A (our linear-perturbation method) agrees with approach B (textbook
"shock mc and re-solve equilibrium prices") to ~1e-9 absolute under RC
logit at δ=1e-5. Test tolerance set to atol=1e-7 / rtol=1e-6 — 2 orders
of magnitude looser than empirical noise floor, tight enough to catch
real regressions.

### Phase 2 — `passthrough_summary` (`eb9fa94` → merged)

Pair × pass-through-feature distance summary with methodology line.

- `Problem.passthrough_summary([with_models=True], [detail='median'])` →
  `PassthroughSummary` with `__repr__` and `to_dataframe()`.
- Four DMQSW-keyed metrics: `offdiag_ratio` (Remark 1), `full_pass`
  (Remark 2 / Remark 4), `row_sum` (Remark 5 unit tax), `level_adj`
  (Remark 5 ad valorem tax).
- `with_models=True` adds per-model structural block (median diagonal,
  signed-max off-diagonal, median row sum).
- `detail='full'` returns one row per (market, pair) for inspection.
- Per-feature notes (one short line each).
- Methodology line conditional on candidate-set composition (numerical /
  analytical / short-circuit).
- `ProblemResults.passthrough_summary` thin wrapper.
- `tests/test_passthrough_summary.py` (21 tests): headline DMQSW result
  confirmed — `(Cournot, PC).offdiag_ratio < 1e-6` (degenerate under
  rival cost shifters) but `row_sum`, `level_adj`, `full_pass` all >
  0.1 (other instrument types break the degeneracy).

### Phase 3 — `instrument_channels` (`01bc452` → merged)

Per-pair channel components for one chosen IV column.

- `Problem.instrument_channels(column, instrument=None)` →
  `InstrumentChannels` with `__repr__` and `to_dataframe()`.
- Reports building blocks separately rather than collapse to "indirect"
  (which requires instrument-specific projection of `dp_0/dz`):
  - **Data-side**: `‖dp_0/dz‖_obs` (sample regression slope of prices
    on z with cost-formulation controls), `SD(z)`, range.
  - **Direct channel** per candidate: `β_m` from OLS of `Δ_m` on z with
    `p` as control (FWL). Exactly zero for PerfectCompetition.
  - **Per-pair structural**: `‖P_m^{-1} − P_m'^{-1}‖_F` (γ-free).
  - **Per-pair direct**: `|β_m − β_m'|`.
- Methodology line covers pass-through computation method + the
  regression-based components.
- Naming chosen after iteration: `causal_effects` →
  `instrument_decomposition` → `instrument_channels`.
- `tests/test_instrument_channels.py` (21 tests): structural shapes,
  PC β_m exactly zero, ProblemResults wrapper match, methodology line
  contents, edge cases.

### Other session housekeeping

- `feat/rc1-cleanup` and `feat/numerical-passthrough` branches pushed
  to origin; not deleted yet.
- jinja2 upgraded 3.0.3 → 3.1.6 in the local Python env (was blocking
  8 `to_latex` tests with environmental ImportError; not a code issue).
- pypandoc_binary installed in the local Python env so sphinx-build
  can run on this machine (Homebrew not present here).
- README.rst feature bullet updated by Phase 4 agent to remove the
  rc1 `passthrough_comparison` reference.
- The `verdict` column in `reliability_summary` was dropped per Phase 4
  but the `verdict` *attribute* on `ProblemResults` is preserved
  because `_format_results_tables` uses it for the `⚠` symbol
  annotations in `print(results)`.

## Test suite progression

| State | Passing |
|---|---|
| Pre-Phase-1 baseline | 706 |
| + Phase 1 (numerical pass-through) | 724 (+18) |
| + Cross-validation (approach A vs B) | 725 (+1) |
| + Phase 2 (`passthrough_summary`) | 746 (+21) |
| + Phase 3 (`instrument_channels`) | **767 (+21)** |

mypy --strict clean throughout. 8 environmental TestToLatex failures
were resolved mid-session by the jinja2 upgrade (not a code issue).

## What's left

### Phase 5 — Documentation (1–2 sessions)

Per the plan, this is the remaining substantive work:

- Rewrite `docs/advanced_features.rst` Pass-through section: replace
  the "Vertical models only, v0.4" caveat with the three-method
  walkthrough (`passthrough_matrix`, `passthrough_summary`,
  `instrument_channels`) on the synthetic example.
- Expand `docs/math.rst` further: Phase 1 added per-conduct closed-form
  pass-through formulas; Phase 5 adds the channel-decomposition
  framework explanation (chain rule, indirect/direct splits, conditional
  regression for the direct channel, worst-rho CV bound notation).
- New tutorial section / notebook: "Pre-test framework reasoning per
  DMQSW (2026)" — pass-through inspection on synthetic example;
  identify (Cournot, PC) degeneracy via `offdiag_ratio = 0`; switching
  to unit tax instruments breaks degeneracy; post-solve
  `instrument_channels` shows channel components.
- 4 FAQ entries:
  1. "How do I check whether my IVs will distinguish my candidates
     before running the test?" → `passthrough_summary`
  2. "Why doesn't `passthrough_summary` give me a verdict?" → γ-free
     structural feature distance, framework can't predict power
  3. "How do I tell if a weak F-stat means structural degeneracy or
     limited identifying variation?" → cross-read
     `reliability_summary` against `passthrough_summary`
  4. "Why isn't there a `moment_relevance` or empirical pre-screening
     method?" → post-selection inference; see plan's design principles
- Update `migrating_to_v0.4.rst` with the new methods + the v0.4 final
  API surface.
- README headline-features bullet: add the DMQSW diagnostic suite
  (`passthrough_summary`, `instrument_channels`).
- Docstring methodology text for the three new methods (single source of
  truth — printed-output footers point to docstrings).
- CHANGELOG entry under `[0.4.0]` summarizing Phases 1–4 + 5.

### Phase 6 — Replication (optional, 1 session)

`docs/notebooks/replication_DMQSW_marijuana.py` reproducing the
Bertrand-vs-Keystone test on the Washington marijuana data using ad
valorem tax instruments. Demonstrates the three-method workflow.
Deferred / optional per the plan.

### Outstanding from earlier work (not blocking final tag, may pick up)

Per the prior 2026-05-06 handover's v0.5-followup tracker:

1. Hard-reject `costs_type='log' + demand_adjustment=True` (vs current
   soft warning + linear fallback). Branch
   `origin/fix/log-costs-with-demand-adjustment` retained.
2. Add no-cost-formulation support to `Problem.__init__`. Currently
   raises TypeError.
3. Add the `demand_adjustment + endogenous_cost_component` gate.
   Currently silently allowed but produces biased variance.
4. Rewrite the §4.1 F-stat rank-adjustment regression test against a
   non-degenerate fixture.
5. F-stat collapse under identically-zero-markup candidate models.
   PC-vs-anything pairs show suspiciously small F-stats even when
   TRV is sharp.

## Notable design decisions made this session

- **Numerical-first pass-through, not analytical-per-conduct.** Decided
  early — single numerical routine handles every conduct uniformly;
  per-conduct closed forms documented in math.rst as reference. Saved
  ~4 sessions of per-conduct derivation work. Approach A's truncation
  empirically validated to ~1e-9 vs textbook approach B.
- **No `moment_relevance` method.** Suppressed entirely. Empirical
  pre-screening of instruments invalidates F-stat / RV-test inference;
  the framework view (`passthrough_summary`) is the right ex-ante
  reasoning, and `reliability_summary` (post-solve) is the
  inference-honest empirical channel.
- **`reliability_summary` stays pure statistical.** The framework
  cross-tab annotation was dropped mid-session — separating concerns
  cleanly, the user reads `reliability_summary` next to
  `passthrough_summary` with FAQ guidance instead of the package
  computing a cross-tab classification.
- **Numbers, not flags, in `passthrough_summary` output.** No DEGENERATE
  asterisks; user reads the values and decides per their (α, β)
  application.
- **CV grid in `reliability_summary` instead of categorical verdict.**
  3 size α + 3 power β CVs at worst-case ρ + a CV_worst_ρ column;
  user sees magnitude vs threshold instead of a one-of-three label.
- **Methodology line in every diagnostic output.** Tells the user
  whether pass-through was computed numerically / analytically /
  via short-circuit; for `instrument_channels` also documents the
  regression-based provenance of `dp_0/dz_obs` and `β_m`.
- **`instrument_channels` reports components, not collapsed indirect
  channel.** The full indirect formula `P_m·(P_m^{-1}-P_m'^{-1})·(dp_0/dz)`
  needs a projection of `dp_0/dz` (column ℓ for rival cost shifter,
  row sums for unit tax, etc.). Reporting structural-side and
  direct-side magnitudes separately lets the user apply the right
  projection per their instrument type.

## Pickup notes for next session

1. **Read this handover** + the plan file at
   `.claude/plans/dmqsw-passthrough-diagnostic.md`.
2. **Pull `v0.4-refactor`** to ensure local is current with `04bc73e`.
3. **Verify env**: `python3 -c "import pyRVtest; print(pyRVtest.__version__)"`
   should print `0.4.0rc1` (the version string hasn't been bumped yet
   — the `v0.4.0rc2` tag is on the same `__version__='0.4.0rc1'`
   string; bump to `'0.4.0'` when tagging final). Run
   `python3 -m pytest tests/ --tb=line -q` to confirm 767 passing on
   pickup.
4. **Phase 5 doc work.** Touch the files listed under "Phase 5" above;
   keep the methodology footer text consistent with what `__repr__`
   already prints. The docstring becomes the single source of truth
   for the per-method methodology explanation.
5. **Don't relaunch agents for Phase 1 or Phase 2 work** — those are
   merged. Phase 5 is text-heavy and best done in-session.
6. **Phase 6 (replication) is independent** and can be picked up any
   time after Phase 5.

## File tree pointers

- Plan: `.claude/plans/dmqsw-passthrough-diagnostic.md`
- Diagnostic implementation: `pyRVtest/solve/passthrough.py` (numerical
  core, `PassthroughSummary`, `compute_passthrough_summary`,
  `InstrumentChannels`, `compute_instrument_channels`)
- Methods: `pyRVtest/problem.py` (`passthrough_summary`,
  `instrument_channels`); `pyRVtest/results/results.py`
  (delegating wrappers + updated `passthrough_matrix` docstring)
- Tests: `tests/test_passthrough_numerical.py`,
  `tests/test_passthrough_summary.py`,
  `tests/test_instrument_channels.py`
- Math: `docs/math.rst` ("Pass-through by conduct class" section
  added by Phase 1)

## Worktrees

- `/Users/christophersullivan/Dropbox/Economics/claude/pyrvtest` — main
  checkout, on `v0.4-refactor` at `04bc73e`.
- `/Users/christophersullivan/Dropbox/Economics/claude/pyrvtest/.claude/worktrees/suspicious-shockley-866a8a` — this session's worktree, on `feat/numerical-passthrough` at the same tip as v0.4-refactor (after the merges). Can be deleted whenever.
