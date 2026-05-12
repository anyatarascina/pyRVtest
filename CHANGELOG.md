# Changelog

All notable changes to pyRVtest are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the
project roughly follows [Semantic Versioning](https://semver.org/).

## [0.4.0] — unreleased

Final tag for v0.4. Supersedes the internal / coauthor-only
`[0.4.0rc1]` snapshot below. The rc1-tagged code shipped a draft of the
F-stat reliability diagnostic and the Dearing pass-through helpers; v0.4
final cleans those up. There is no deprecation alias for the renamed /
removed names because rc1 was not a public release.

### Fixed (rc4 → rc5)

- **Log-cost positivity guard now rejects the boundary case.** Pre-rc5
  the `costs_type='log'` guard at `problem.py:1853` used `< 0`, so the
  case `price - markup == 0` slipped through and `np.log(0) = -inf`
  propagated silently to `NaN` TRV / F. rc5 tightens the guard to
  `<= 0` and raises a clean `ValueError`. New regression
  `test_log_cost_zero_marginal_cost_raises` covers the boundary.
- **`solve()` docstring no longer contradicts the code.** The rc3
  docstring still said `costs_type='log'` "emits a UserWarning and
  falls back to linear costs when combined with `demand_adjustment=True`."
  The code has supported the combination since rc3 via the chain-rule
  rescaling; the docstring is updated to match.
- **Dearing citation sweep, take 2.** rc4 swept most "Dearing 2026"
  instances to "Dearing 2024" but missed the README BibTeX entry
  (`dmqsw2026` → `dmqsw2024`, `year={2026}` → `year={2024}`), the
  bare "Dearing 2026" form in `AGENTS.md`, two changelog entries,
  `pyRVtest/formulation.py`, `docs/tutorial.rst`,
  `docs/migrating_to_v0.4.rst`, and the line-wrapped
  `S. Waldfogel (2026)` in two `pyRVtest/problem.py` docstrings. rc5
  catches all of these (verified with the auditor's grep).
- **`docs/notebooks/speed_test.py` renamed to `speed_benchmark.py`.**
  Pytest's default `*_test.py` collection pattern was picking up the
  file and executing a PyBLP simulation at import time, which crashed
  collection on stress stacks. Rename takes it out of collection
  without breaking the standalone-script use.
- **FAQ Python claim aligned with `setup.py`.** FAQ said "Python 3.7+"
  while rc4's `setup.py` declared `>=3.9`. FAQ updated to state the
  3.9 floor and the actual CI matrix (Python 3.11 × {numpy<2 +
  pyblp<1.2, numpy>=2 + pyblp>=1.2}).
- **Dependency-envelope gate at import time.** `requirements.txt`
  cannot express "if numpy >= 2 then pyblp >= 1.2" via PEP 508
  markers, so a user installing fresh under Python 3.12+ could
  pip-resolve to `numpy 2 + pyblp 1.1.x` (the combination produces
  silent NaN / wrong-sign diagnostics from older pyblp LAPACK paths).
  `pyRVtest/__init__.py` now detects the bad combination and raises
  a clear `ImportError` with the fix.
- **README install section.** Reordered to lead with the GitHub-tag
  install command (the canonical v0.4 path until PyPI alignment),
  with the older v0.3 PyPI install kept as a fallback for users on
  the prior API. Python 3.9+ floor stated explicitly.

### Changed (rc1 → final)

- **`F_reliability_summary` → `reliability_summary`.** The reliability
  diagnostic on `ProblemResults` is renamed to
  `ProblemResults.reliability_summary()`; the column structure of the
  returned DataFrame is reshaped (see below). No alias: rc1 callers see
  `AttributeError`.
- **CV columns split by axis and level.** The rc1
  `worst_case_cv_size` / `worst_case_cv_power` columns (each an object
  cell holding a length-3 vector) are replaced by six scalar columns
  reporting the worst-ρ CVs at the published-table levels: `size_cv_075`,
  `size_cv_100`, `size_cv_125` (size at 7.5% / 10% / 12.5%) and
  `power_cv_050`, `power_cv_075`, `power_cv_095` (power at 50% / 75% /
  95%). Six matching empirical-ρ columns (`size_cv_075_emp` etc.) report
  the plug-in CVs the verdict uses internally. `strongest_claim_size`
  and `strongest_claim_power` are unchanged.
- **Per-cell array attributes simplified.** `worst_case_cv_size` /
  `worst_case_cv_power` remain on `ProblemResults` as `(M, M)` object
  arrays per instrument set (used by the printed footer and exposed for
  inspection); they are now exposed as scalar columns in
  `reliability_summary()` rather than nested object cells.

### Removed (rc1 → final)

- **`F_se`, `F_ci_low`, `F_ci_high`.** The asymptotic-SE and 95%-CI
  arrays for F are removed from both `ProblemResults` (per-cell array
  attributes) and `reliability_summary()` (DataFrame columns). The
  plug-in CV check the verdict already runs is the principled
  robustness signal; the SE / CI was redundant inspection-only
  metadata.
- **`verdict` column from `reliability_summary()`'s DataFrame.** The
  internal classification still drives the `⚠` warning glyph in the
  printed output and remains accessible as
  `ProblemResults.verdict[instrument_set]` (an `(M, M)` object array
  per instrument set), but the diagnostic frame no longer carries a
  `verdict` column.
- **`ProblemResults.passthrough_comparison`.** Removed entirely along
  with the private `_PASSTHROUGH_METRICS` and `_passthrough_distance`
  helpers and `tests/test_passthrough_comparison.py`. The Dearing et al.
  (2024) pass-through diagnostic suite in v0.4 final is provided by
  `passthrough_summary` (γ-free structural-feature distance, ex-ante)
  and `instrument_channels` (post-solve channel decomposition); see
  the new "Added" subsection below and `docs/advanced_features.rst`.
  The rc1 method's `offdiag_frobenius` metric was also paper-
  renumbered: the off-diagonal-to-diagonal feature is Remark 1 in the
  current DMQSW draft, not Remark 4.
- **`RELIABILITY_CI_LEVEL` constant** in
  `pyRVtest.solve.test_engine`. Used only by the removed `F_se` /
  `F_ci` computation. `RELIABILITY_LAMBDA_THRESHOLD` (the lambda
  informational threshold) remains.

### Added (rc1 → final)

The Dearing, Magnolfi, Quint, Sullivan, and Waldfogel (2024, "DMQSW")
pass-through diagnostic suite is now first-class on `Problem` and
`ProblemResults`. Three coordinated methods:

- **`Problem.passthrough_summary(with_models=False, detail='median')`.**
  Pre-solve γ-free pair-by-pair structural feature distances across
  every unordered candidate pair. Reports four DMQSW-keyed metrics:
  `offdiag_ratio` (Remark 1, rival cost shifters), `full_pass`
  (Remark 2 / Remark 4, own+rival cost; product chars under linear-
  index demand), `row_sum` (Remark 5, per-unit tax), and `level_adj`
  (Remark 5, ad valorem tax). `with_models=True` adds a per-model
  structural block (median diagonal, signed-max off-diagonal, median
  row sum). `detail='full'` returns one row per (pair, market). Also
  callable on `ProblemResults` post-solve. Each printed view ends
  with a methodology footer documenting how the underlying pass-
  through matrices were computed (numerical / analytical / short-
  circuit, conditional on the candidate-set composition).
- **`Problem.instrument_channels(column, instrument=None)`.** Post-
  solve per-pair channel decomposition for one chosen IV column.
  Reports the data-side empirical magnitude `‖dp_0/dz‖_obs` (sample
  regression slope of observed prices on z controlling for the cost
  formulation), the per-candidate direct-channel coefficient `β_m`
  (FWL conditional regression of model-implied Δ_m on z given p),
  the structural-side magnitude `‖P_m^{-1} − P_m'^{-1}‖_F`
  aggregated by median across markets, and the per-pair
  `|β_m − β_m'|`. The optional `instrument=` kwarg labels the
  declared primitive instrument type ('rival_cost', 'unit_tax',
  'advalorem_tax', 'rival_product_char', etc.) in the methodology
  footer; it does not change the computation. Also accessible on
  `ProblemResults`.
- **`ProblemResults.passthrough_matrix(model_index, market_id=None)`
  generalized.** Now returns the per-candidate pass-through matrix
  for *every* conduct class via numerical central-difference
  perturbation of prices through the candidate's first-order
  condition. The existing `Vertical` / Villas-Boas (2007) analytical
  fast path is preserved and dispatched automatically; trivial
  conducts (`PerfectCompetition`, `ConstantMarkup`,
  `UserSuppliedMarkups`, `RuleOfThumb`) short-circuit to identity /
  `φI`. The v0.4 rc1 `Vertical`-only restriction is removed.

Internal computation:

- **`compute_passthrough_numerical(conduct_model, ownership, response,
  hessian, shares, markups, delta=1e-7)`** in
  `pyRVtest/solve/passthrough.py`. Single numerical core handling
  every conduct uniformly via central-difference perturbation through
  the candidate's markup function; closed-form expressions per conduct
  documented in `docs/math.rst` as reference for what the numerics
  approximate.
- **Pass-through dispatch** in `pyRVtest.solve.passthrough.build_passthrough`
  routes `Vertical` to the existing analytical fast path, trivial
  conducts to short-circuit identity / `φI`, and everything else to
  the numerical core.
- **Cross-validation against textbook approach** in
  `tests/test_passthrough_numerical.py::test_numerical_matches_rc_logit_resolve`:
  self-coded random-coefficient logit demand (Gauss-Hermite quadrature)
  with a scipy fixed-point equilibrium price solver confirms approach
  A (linear-perturbation) agrees with approach B (textbook "shock mc
  and re-solve equilibrium prices") to ~1e-9 absolute under RC logit
  at delta=1e-5.

Documentation:

- **`docs/advanced_features.rst`** Pass-through diagnostics section
  rewritten as a three-method walkthrough on the synthetic example,
  including the (Cournot, PerfectCompetition) `offdiag_ratio = 0`
  degeneracy demonstration and the cross-read pattern between
  `passthrough_summary` and `reliability_summary`.
- **`docs/math.rst`** expanded with per-conduct closed-form pass-
  through formulas, the indirect/direct channel decomposition
  `dp/dz = P · (∂Δ/∂z + ∂c̄/∂z)`, the FWL conditional-regression
  identification of the direct channel, and worst-rho CV bound
  notation alongside the existing F-statistic section.
- **FAQ entries** in `docs/faq.rst` covering: how to check IV-
  candidate distinguishability ex-ante (`passthrough_summary`); why
  the framework view does not return a verdict; how to diagnose
  weak F-stats as structural vs. empirical; and why there is no
  `moment_relevance` / empirical IV pre-screening method.
- **Tutorial section** in `docs/tutorial.rst` ("Pre-test framework
  reasoning per DMQSW") pointing to the advanced-features walkthrough.
- **Migration notes** in `docs/migrating_to_v0.4.rst` covering the
  new methods.
- **Method docstrings** as the single source of truth for methodology
  text; methodology footers in printed output now reference
  `passthrough_summary()` / `instrument_channels()` docstrings.
- **Endogenous-cost + demand-adjustment documentation corrected.**
  Earlier docs (FAQ, advanced_features) carried a stale caveat
  claiming the combination silently produced biased variance. The
  unified-demand-adjustment refactor (step 4d/4e) actually fixed
  this: `_compute_gamma_gradient` finite-differences the per-
  instrument-set IV correction at perturbed demand parameters and
  the `∂γ_m / ∂θ` channel is folded into the test_engine variance
  term (`G_k -= 1/N · Z' · (endog_resid · gradient_gamma[m])`).
  Cross-path parity between `demand_results` and `demand_params`
  is pinned to `atol=5e-9` on a 30-market endogenous-cost fixture
  (`tests/test_demand_adjustment.py::test_option_a_demand_params_matches_demand_results_with_endogenous_cost`).
  The stale `tests/regressions/test_memo1_section42_adjustment_interaction.py`
  xfail-strict gate (asserted `ValueError` for the combination) is
  removed; correctness coverage moved to the cross-path parity test.

Tests:

- **`tests/test_passthrough_numerical.py`** (18 tests): trivial
  closed-form bit-exact agreement (PC, ConstantMarkup,
  UserSuppliedMarkups, RuleOfThumb), paper Example 2 hand-derived
  2×2 logit cases (Bertrand off-diagonal nonzero; Cournot diagonal —
  the headline DMQSW result), and smoke tests across every conduct
  class on the synthetic example.
- **`tests/test_passthrough_summary.py`** (21 tests): headline DMQSW
  result confirmed — `(Cournot, PerfectCompetition).offdiag_ratio <
  1e-6` (degenerate under rival cost shifters) but `row_sum`,
  `level_adj`, `full_pass` all > 0.1 (other instrument types break
  the degeneracy). Plus per-feature notes, methodology line
  composition, and edge-case coverage.
- **`tests/test_instrument_channels.py`** (21 tests): structural
  shapes, PC `β_m` exactly zero, ProblemResults wrapper match,
  methodology line contents, and edge cases.

Test suite progression: 706 (pre-Phase-1 baseline) → 767 (+61) tests
passing. mypy `--strict` clean throughout.

### Added (Phase 6: multi-column endogenous cost + log-cost
demand-adjustment + z^e diagnostic, 2026-05-08)

DMQSS (2026) Appendix A.4 + B generalization, paper-supported per
identification arguments and the parametric-linearity argument
written into `docs/math.rst`'s "Pass-through diagnostic under
non-constant marginal cost" section.

- **Multi-column `endogenous_cost_component`.** The kwarg widens
  from `Optional[str]` to `Optional[Union[str, Sequence[str]]]`.
  Examples from the paper that now slot into the API directly:
  quadratic cost (`['q', 'q_sq']`), scale + scope
  (`['log_q', 'log_Q_minus']`), and any linear-in-basis-columns
  cost regression with `K_endog >= 1` endogenous variables. The
  IV correction's first stage projects all `K_endog` endogenous
  columns simultaneously; the second stage estimates a length-
  `K_endog` gamma vector. Demand-adjustment gradient and Appendix
  B influence-function correction generalize via a `(K, K_endog)`
  Lambda_q matrix. `K_endog == 1` (string or length-one list) is
  bit-identical to the prior single-column behavior.
- **`costs_type='log' + demand_adjustment=True`.** Implemented the
  proper chain rule: `gradient_markups[m]` rescales by
  `f'(p − Δ_m) = 1/(p − Δ_m)` so the demand-adjusted variance
  reflects the log-cost moment derivative. Replaces the prior
  silent fallback (and the never-merged hard-reject branch
  `fix/log-costs-with-demand-adjustment`).
- **`instrument_channels` z^e residualization under non-constant MC.**
  When `endogenous_cost_component` is set, the data-side regression
  and FWL partialling project on `(g(q̃), w_exog)` rather than on
  raw `(q, w)`. The residualization absorbs the K_endog cost-
  parameter-identifying dimensions of `z` and leaves the surviving
  testing dimension; the diagnostic then unifies the Dearing
  pass-through condition (LHS of DMQSS Eq 10) with the A.4
  rank-`K + 1` distinctness check. Constant-MC behavior (no
  endogenous cost) is unchanged. Methodology footer updated
  conditionally.

Test additions:

- **`tests/test_analytical.py::TestMultiColumnEndogenousCostRoundTrip`**
  (6 tests): single-string vs list-of-one bit-identical round trip;
  K_endog == 2 end-to-end smoke; rejection of empty list, duplicate
  columns, non-string entries, and missing columns.
- **`tests/test_demand_adjustment.py::test_log_cost_demand_adjustment_runs_end_to_end`**:
  costs_type='log' + demand_adjustment=True runs on the
  `_build_scale_dgp` fixture; linear and log produce different TRV/F
  (the chain-rule rescaling has effect rather than no-op'ing).
- **`tests/test_instrument_channels.py::TestZeResidualizationUnderNonConstantMC`**
  (3 tests): data-side regression matches the manually-computed z^e
  slope; differs from the raw-(q, w) projection; methodology footer
  mentions z^e under non-constant MC, doesn't mention it under
  constant MC.

Closes v0.5-followups item 1 (log + demand_adjustment hard-reject).

## [0.4.0rc1] — 2026-04-20

Release candidate for v0.4.0. Version 0.4 is a substantial refactor of
the package architecture. The primary motivations are (a) encapsulating
PyBLP private-attribute access behind a `DemandBackend` protocol, (b)
merging the two parallel demand-adjustment paths (PyBLP results vs.
`demand_params`) into a single dispatch, and (c) setting up hooks for
labor-side conduct testing. See `.claude/plans/v0.4-refactor.md` for
the full design document.

Step 16 (AFSSZ dogfood on a real 910-market-year panel) is still
outstanding and may introduce additional changes before v0.4.0 final is
tagged. Step 16 is data-blocked (~1 week lead time on the AFSSZ panel).

### rc1 fixes (coauthor break-it pass, 2026-04-18)

- **`UserSuppliedMarkups` class.** Pre-computed markup columns are now a
  first-class conduct model:
  ```python
  pyRVtest.UserSuppliedMarkups(markups='mkup_col', ownership='firm_ids')
  ```
  The legacy pattern
  `ModelFormulation(user_supplied_markups='col', ownership_downstream='firm_ids')`
  (no `model_downstream`) used to crash with an `AssertionError` in the
  adapter; it now translates to `UserSuppliedMarkups` and emits the
  standard `ModelFormulation` deprecation warning. Fixes Lorenzo's P0
  regression against carRV's production `conduct_test.py`.
- **K > 30 critical-value warning.** Instrument counts above 30 previously
  fell back to the K=30 critical values silently. A `UserWarning` is now
  emitted once per instrument set when K exceeds the tabulated range.
- **`options.digits` wired through result formatters.** The global
  `pyRVtest.options.digits` setting now controls numeric precision in
  `to_markdown()` / `to_latex()` / `summary_df()` output (previously
  hardcoded to 6 significant figures). Default value changed from 7 to
  6 to match the prior `_dataframe_to_github_markdown` hardcoded format
  (no user-visible change vs. pre-rc1 output at the default setting).
- **`options.verbose` deprecated.** Reading the attribute emits a
  `DeprecationWarning` pointing at the `logging.getLogger('pyRVtest')`
  API that superseded it in v0.4. Assignment stays silent so the
  widely-used `pyRVtest.options.verbose = False` pattern keeps working.
  Removal scheduled for v0.6.
- **`pyproject.toml` added.** Minimal build-system declaration so
  `pip install -e .` works under modern pip (>=23) without falling back
  to legacy setuptools.

### numpy 2.x compatibility (post-rc1 tag)

Eight of the nine failures Lorenzo reported on numpy 2.x are now
resolved in the post-rc1 branch:

- **`jinja2` added to `requirements.txt`.** pandas 2.3 routes
  `DataFrame.to_latex` through its Styler API, which requires jinja2
  unconditionally; without it, `ProblemResults.to_latex` /
  `PanelResults.to_latex` raise `ImportError` on a fresh install
  (7 test failures). Not strictly numpy-related but surfaced by
  the same cross-environment audit.
- **Critical-values lookup hardened** against NaN rho at
  `pyRVtest/solve/test_engine.py`. Two model pairs with identical
  markups (e.g. a salience test where the opt-out produces the same
  raw markup as the default path) push the F-stat denominator to
  zero, yielding NaN rho. `np.where(rho == NaN)` is always empty and
  the lookup `[0][0]` previously raised `IndexError`; numpy 1.x
  happened to sidestep this via slightly different numerical values
  on the degenerate pair, numpy 2.x exposed the latent bug. rc1+
  returns NaN critical values and a blank significance symbol for
  NaN rho, which is semantically correct for this regime.

The remaining numpy 2.x item:

- **`test_snapshot_analytical_scale` F-shift** (`xfail` under
  numpy >= 2.0). The endogenous_cost_component IV-correction path
  produces `F[0][0][1] ≈ 0.998` on numpy 2.x vs the
  `F[0][0][1] = 1.032` snapshot captured on numpy 1.x — a ~3% shift
  that is too large for BLAS noise. **Root-caused (2026-04-20):**
  fixture-level numerical sensitivity, not a pyRVtest math bug.
  The `_build_scale_dgp` fixture produces a near-degenerate conduct
  pair where the three sigma values feeding
  `F_denominator = sigma[0]*sigma[1] - sigma[2]**2` are nearly
  equal (~2.07e-3), putting the denominator (~4e-8) in the
  catastrophic-cancellation regime. numpy 2's LAPACK QR returns a
  slightly different — but equally orthonormal — basis for the
  `controls = hstack([w, endog_hat])` matrix, which is itself
  moderately ill-conditioned (cond=3582 due to near-collinearity
  between `w` and `endog_hat`). The different Q produces `omega`
  residuals that differ by ~1e-13; those tiny shifts amplify
  through catastrophic cancellation into a 3% F shift. Both numpy
  versions compute F correctly to machine precision; the
  `atol=1e-10` snapshot tolerance is simply too tight for this
  fixture across different-but-valid QR bases.

  Candidate fixes for v0.4.0 final: (a) revise the scale DGP
  seed/parameters so the three sigma values differ enough to avoid
  the catastrophic-cancellation regime; (b) switch `qr_residualize`
  to an SVD-based projection that is numerically stable under the
  near-collinearity; (c) widen the snapshot tolerance for this
  specific field. All other snapshots
  (`analytical_base`, `analytical_base_fe`, `first_stage_*_path`,
  `nested_logit_vertical`, etc.) remain bit-identical across
  numpy 1 and numpy 2. The `xfail` reason string captures the full
  diagnostic for future readers.

### Removed (post-rc1)

- **`Keystone()` shorthand class.** The `phi=2` shorthand was dropped in
  favor of writing `RuleOfThumb(phi=2)` directly; the indirection added
  no math and was a maintenance footgun across the adapter, model
  registry, and docs. Migration is a one-line search-and-replace:
  `Keystone()` → `RuleOfThumb(phi=2)`.
- **`mc_correction` argument to `Problem.solve()`.** Now emits a
  `DeprecationWarning` pointing at the more general
  `endogenous_cost_component` argument on `Problem`. Removal scheduled
  for v0.6.

### Deferred to v0.4.0 final (tracked, not rc1-blocking)

- Pick a resolution for the `analytical_scale` F-shift (fixture
  revision, SVD-based projection, or per-field tolerance widening).
- `PanelResults` roster-hash validation (audit B1).
- `Problem(demand_backend=...)` public kwarg (audit B3; already flagged
  as future work in `docs/custom_demand.rst`).
- Per-model tax `DeprecationWarning` firing at construction time.

### Migration from v0.3.x

Four user-visible break points. Most emit a `DeprecationWarning` for
one release (slated for removal in v0.6). The per-model tax kwargs get
two releases (removed in v0.7) because they appear in user code much
more frequently. See `docs/migrating_to_v0.4.rst` for the full
deprecation timeline.

1. **Conduct-model specification.** Prefer the new class-based API over
   `ModelFormulation`:
   ```python
   # v0.3
   from pyRVtest import ModelFormulation
   models = [
       ModelFormulation(model_downstream='bertrand', ownership_downstream='firm_id'),
       ModelFormulation(model_downstream='cournot',  ownership_downstream='firm_id'),
   ]

   # v0.4
   from pyRVtest import Bertrand, Cournot
   models = [
       Bertrand(ownership='firm_id'),
       Cournot(ownership='firm_id'),
   ]
   ```
   `ModelFormulation(...)` still works but emits `DeprecationWarning`.
   See `docs/migrating_to_v0.4.rst` for per-model mappings, including
   `Vertical`, `Monopoly`, `PerfectCompetition`, and
   `MixedCournotBertrand`.

2. **`demand_params` parameter name.** `rho` is the canonical nesting
   parameter key; `sigma` is retained as a deprecated alias:
   ```python
   # v0.3
   problem.solve(demand_params=dict(alpha=-1.2, sigma=0.5))

   # v0.4
   problem.solve(demand_params=dict(alpha=-1.2, rho=0.5))
   ```
   Passing `sigma=` raises `DeprecationWarning` and continues to work.

3. **Custom-demand users.** If you previously monkey-patched PyBLP
   internals to plug in a non-PyBLP demand system, subclass
   `pyRVtest.backends.UserSuppliedBackend` instead. Supply
   `compute_jacobian`, `compute_hessian`, and (optionally) the
   `SupportsDemandAdjustment` hooks. See `docs/custom_demand.rst` for a
   worked linear-demand example.

4. **Tax specification: model-level → Problem-level.** Per-model
   `unit_tax='col'` / `advalorem_tax='col'` / `advalorem_payer=...`
   kwargs on `ConductModel` / `ModelFormulation` / `Vertical` are
   deprecated. Specify the tax once on `Problem`; opt individual
   models out via `unit_tax_salient=False` / `advalorem_tax_salient=False`
   (salience-testing pattern). Removal scheduled for **v0.7** (one
   release later than other v0.4 deprecations, since the per-model tax
   pattern is common in existing user code).

Existing v0.3 scripts without custom demand code should run unchanged on
v0.4 modulo one-line deprecation warnings.

### Added

- **`DemandBackend` protocol and implementations** (steps 1–4). Core
  protocol in `backends/base.py` with `PyBLPBackend`, `LogitBackend`,
  `NestedLogitBackend`, and `UserSuppliedBackend` implementations.
  Optional `SupportsDemandAdjustment` mixin declares which backends can
  supply first-stage correction inputs (`xi`, `Z_D`, `W_D`, gradients).
  Encapsulates previously-scattered `pyblp_results._*` accesses behind a
  single documented interface.
- **Class-based `ConductModel` API** (step 5). `Bertrand`, `Cournot`,
  `Monopoly`, `PerfectCompetition`, `MixedCournotBertrand`, and
  `Vertical` classes each own their `compute_markup()`. `ModelFormulation`
  kept as a thin backward-compat bridge that raises `DeprecationWarning`.
  Full migration guide in `docs/migrating_to_v0.4.rst` with one-to-one
  recipes for every `ModelFormulation` shape.
- **Dearing et al. (2024) simple-markup conduct models** (step 12).
  `RuleOfThumb(phi)` implements the Dearing Example 1 rule
  :math:`p = \varphi \cdot mc` as an ergonomic wrapper over the
  existing `cost_scaling` machinery, which v0.4 step 12a extends to
  accept a numeric scalar in addition to a column name.
  `ConstantMarkup(markup)` implements Example 7's fixed per-product
  dollar markup via a new additive-markup plumbing path threaded
  through the `Models` recarray and `evaluate_first_order_conditions`.
  Both classes re-exported at the package level
  (`pyRVtest.RuleOfThumb`, `pyRVtest.ConstantMarkup`).
  Backward compatibility: the legacy `PerfectCompetition(cost_scaling='lmbda_col')`
  pattern still works unchanged.

  *Note:* an earlier draft of this changelog also mentioned a
  `Keystone()` alias for `RuleOfThumb(phi=2)`; that alias was dropped
  before release (commit `e7ea1e3`). Use `RuleOfThumb(phi=2)` directly.
- **Analytical nested-logit Hessian** (step 7). Closed-form
  `compute_analytical_hessian` in `backends/logit.py` for plain logit
  and single-scalar-rho nested logit. Per-nest rho (Cardell-Nevo),
  multi-level nesting, and BLP continue to use the pyblp
  finite-difference path. AFSSZ-style specifications with per-nest rho
  therefore do not benefit from the O(ε²) Hessian improvement; plan
  accordingly when routing analytical vs finite-difference.
  Validated against finite-diff, PyBLP's own `compute_demand_hessians`, and
  Clairaut symmetry (8 parametrized tests plus a nested-logit-vertical
  snapshot).
- **`pyRVtest.build_passthrough`** (step 11). Stand-alone helper that
  returns the Villas-Boas passthrough matrix per vertical model, either
  for a single market or as a dict across all markets. Clear errors for
  non-vertical models, invalid `market_id`, and missing `hessian_fn` on
  `UserSuppliedBackend`.
- **`ProblemResults.passthrough_comparison`** and
  **`ProblemResults.passthrough_matrix`** (OQ 15). Dearing-style
  pass-through diagnostics surfaced on `ProblemResults`.
  `passthrough_comparison` returns a pandas DataFrame with one row per
  `(market_id, unordered model pair)` and a scalar pairwise distance
  between pass-through matrices; three metrics are supported —
  `'frobenius'` (default), `'offdiag_frobenius'` (implements the Dearing
  et al. (2024) Remark 4 distinguishability condition — invariant to
  diagonal-only differences in pass-through), and `'max_abs'`. The
  chosen metric is recorded on `frame.attrs['metric']`.
  `passthrough_matrix` is a thin ergonomic wrapper over
  `build_passthrough`. Both methods currently require every candidate
  model to be `Vertical`; a non-Vertical candidate raises
  `NotImplementedError` with a pointer to the v0.5 scope item for
  per-model closed-form pass-through (Bertrand / Cournot / RuleOfThumb /
  ConstantMarkup / PerfectCompetition).
- **Instrument construction helpers** (step 13). New
  `pyRVtest.instruments.product` module with `rival_sums`,
  `differentiation_ivs`, `blp_instruments`. New
  `pyRVtest.instruments.labor` with `hausman` and `bartik`. All five
  accept DataFrame, structured recarray, or dict-like `product_data`.
  A labor-side `concentration_hhi` helper was prototyped and then
  deliberately removed before release: labor-market HHI is endogenous
  in wages (shares respond to the variable being tested), so it is not
  a valid wage instrument even though the product-side analogue is
  sometimes defensible. See `pyRVtest/instruments/labor.py` for the
  rationale and references.
- **Worked `UserSuppliedBackend` example** (step 15). New
  `docs/custom_demand.rst` with an end-to-end linear-demand DGP and
  accompanying test.
- **`ProblemResults` export methods** (step 9). `to_dataframe()`,
  `summary_df(alpha=0.05)`, `to_latex(...)`, and `to_markdown(...)` on
  `ProblemResults` for friction-free reporting.
- **Stable `reject` column + `alpha` in `DataFrame.attrs`**
  (Open-Question-11 resolution). `ProblemResults.summary_df` emits a
  constant `reject` column name (not `reject_at_{alpha:g}`) and records
  the critical level on `DataFrame.attrs['alpha']`. Downstream
  aggregators can read alpha without re-deriving the column name;
  `PanelResults.rejection_rates` and `PanelResults.summary_df` consume
  and propagate it accordingly.
- **`PanelResults`** (step 10). Multi-problem aggregation class for
  panels of market-years. Mapping-like API (`keys`, `__getitem__`,
  `__iter__`, `__len__`, `__contains__`), plus `to_dataframe()`,
  `rejection_rates(alpha)`, `summary_df`, `to_latex`, `to_markdown`.
  Constructor validates non-empty mapping, `ProblemResults` values, and
  homogeneous candidate-model count across the panel.
- **Structured logging** (step 18). Per-module loggers
  (`pyRVtest.problem`, `pyRVtest.backends.logit`, …) replace the
  in-house `output()` / `print()` sites. Users can silence a subsystem
  with `logging.getLogger("pyRVtest.problem").setLevel(logging.WARNING)`;
  no handlers are installed at import time. `tests/test_logging.py`
  covers emission, silencing, and the shim's deprecation warning. See
  the new "Logging layout" section of `docs/agent_guide.rst`.
- **Custom exception hierarchy** (step 19). New `pyRVtest/exceptions.py`
  with `PyRVTestError`, `ValidationError` (+ `InstrumentDataError`),
  and `BackendError` (+ `DemandBackendError`, `HessianUnavailableError`).
  Every class multi-inherits from the appropriate built-in
  (`ValueError` / `RuntimeError`) so existing `except ValueError:` and
  `pytest.raises(ValueError, ...)` callers work unchanged. Re-exported
  from `pyRVtest.__init__`; documented in `docs/api.rst` under
  "Exceptions".
- **Doctest coverage on public API** (step 21). 47 runnable docstring
  examples across 28 modules, with 9 intentionally skipped blocks that
  require a fitted PyBLP results object. `pytest --doctest-modules`
  wired into CI; `tox -e doctest` available for local runs.
- **Property-based tests** (step 20). Market-partition invariance of the
  raw moment, FWL identity for the GMM moment, Bertrand markup
  homogeneity of degree −1 in α (Hypothesis-driven).
- **Public API pin** (step 22). Dynamic `__all__` audit across every
  pyRVtest module (65 parametrized assertions across 32 modules).
- **`AGENTS.md` + `docs/agent_guide.rst` + `pyRVtest.show_agent_guide()`**
  (step 23). Architecture tour, deprecation policy, and pointers for
  AI-assisted contributors and users.
- **Snapshot regression suite** (step 0b) + golden DMSS-yogurt
  scaffolding (step 0d, data pending Lorenzo).
- **Minimal CI** (step 24.5). `.github/workflows/ci.yml` runs pytest on
  Ubuntu + Python 3.11 on every push and PR. `mypy --strict` and
  `pytest --doctest-modules` steps are active following steps 17 and 21.
- **Problem-level taxes + per-model salience flags** (OQ 14). Pass
  `unit_tax='col'`, `advalorem_tax='col'`, and
  `advalorem_payer='firm'|'consumer'` directly on `Problem(...)` so the
  tax lives on the DGP (where it belongs) rather than being repeated on
  every candidate model. Individual models opt out via
  `Bertrand(..., unit_tax_salient=False)` /
  `advalorem_tax_salient=False` — the mechanism for salience tests
  (e.g., comparing a salient-tax Bertrand to a non-salient-tax Bertrand
  under the same Problem-level tax). Salience flags default to `True`;
  if no Problem-level tax is set the flag is a no-op. The legacy
  per-model `unit_tax` / `advalorem_tax` path still works and wins by
  precedence when both are set (each emits a once-per-session
  `DeprecationWarning` plus a separate conflict-warning when Problem-
  level and model-level values disagree). 18 tests in
  `tests/test_problem_level_taxes.py`.
- **Known-coefficient cost shifters on `Formulation`** (OQ 14). The
  `cost_formulation` now accepts a
  `known_coefficients={'col': gamma, ...}` dict of cost shifters with
  researcher-supplied (non-estimated) coefficients. They enter the
  effective-price line in `Problem.solve`:
  `prices_effective = advalorem_tax_adj * p / (1 + cost_scaling) -
  unit_tax - sum(gamma_k * x_k)`, applied uniformly to every model
  (these are DGP-level primitives, not behavioral choices). Per-unit
  taxes are the leading special case; Dearing et al. (2024) work with
  a broader class of such shifters. Validation at Formulation
  construction time (dict type, finite numeric coefficients, no
  overlap with the formula); column-existence check at
  `Problem.__init__`. 16 tests in `tests/test_known_coefficients.py`.

### Changed

- **Unified demand-adjustment path** (step 4). The `demand_params` branch
  and the PyBLP-results branch now share a single
  `compute_demand_adjustment` function. Two ~200-line duplicate methods
  on `Problem` deleted; auto-routing sends plain logit and
  single-scalar-rho nested-logit cases to the analytical path by
  default. Per-nest rho (Cardell-Nevo), multi-level nesting, and BLP
  continue through the pyblp finite-difference fallback.
- **`sigma` → `rho` in `demand_params`** (step 6b). `rho` is the
  canonical key; `sigma` still accepted as a deprecated alias for one
  release (raises `DeprecationWarning`).
- **Error messages follow an expected / received / fix structure**
  (step 19). All 120 `raise` sites in `pyRVtest/` rewritten. User-facing
  validation errors now state what was checked, what was actually
  received, and a concrete fix. For example, from `Products`:
  ```
  Expected the 'market_ids' column to be one-dimensional. Received
  shape (200, 2). Fix: pass a single vector of market identifiers, not
  a multi-column array.
  ```
  Internal-invariant failures are prefixed `"pyRVtest internal error:"`
  and kept terse.
- **Labor-side conduct testing hooks** (step 14). New `market_side`
  parameter on `Problem` (default `'product'`; set to `'labor'` for
  monopsony / wage-setting conduct tests). Four new labor conduct model
  classes in `pyRVtest.models.labor`: `Monopsony`, `BertrandWages`,
  `CournotEmployment` (all with real sign-flipped markup formulas),
  plus `NashBargaining` as a v0.5 stub. Skeleton `LaborSupplyBackend`
  in `pyRVtest.backends.labor.nested_logit_labor` honors the
  `DemandBackend` protocol; real math deferred to v0.5 when labor data
  arrives. Labor-mode column-name defaults are `'wages'` (in place of
  `'prices'`) and `'employment_share'` (in place of `'shares'`); the
  default advertises units because the canonical `shares` column is
  treated as a share in `[0, 1]`, so users with raw employment counts
  must normalize first. Sign-convention validation rejects non-positive
  wages or employment shares with rich error messages.
  `ProblemResults.__str__` swaps the header banner under labor mode
  ("markdown / MRP / wage" instead of "markup / MC / price"). Labor-
  side models cannot be mixed with product-side models. `PerfectCompetition`
  stays side-neutral (zero markup has no sign convention).
  `CustomConductModel` now requires an explicit `side='labor'` opt-in
  when used under labor mode (and rejects `side='labor'` under the
  default product mode) because the user-supplied `markup_fn`
  implicitly picks a sign convention and silent acceptance on either
  side would let a product-side formula leak into a labor problem
  unnoticed. The symmetric cross-side validator also now rejects
  labor-side conduct classes (Monopsony, BertrandWages, etc.) under the
  default `market_side='product'`. 39 tests in
  `tests/test_labor_mode.py`.
- **`Problem.solve` split into staged pipeline** (step 8). The ~200-line
  monolithic `solve()` method is now a thin orchestrator that calls
  staged modules under `pyRVtest/solve/`: `markups.compute`,
  `orthogonalize.residualize`, `endogenous_cost.iv_correct`,
  `demand_adjustment.apply`, and `test_engine.compute`. `problem.py`
  shrank from 1733 to 1328 lines (−23%). Each stage is a pure function
  with its own logger (`pyRVtest.solve.markups`,
  `pyRVtest.solve.orthogonalize`, etc.). User-facing behavior is
  bit-identical: 471 tests + snapshot suite pass unchanged.
- **Module split of `problem.py`** (internal). `Products` extracted to
  `pyRVtest/products.py` (step 2). `ModelFormulation` bridge lives in
  `pyRVtest/models/_adapter.py`; standard model classes live in
  `pyRVtest/models/standard.py`.
- **`mypy --strict` coverage** (step 17, internal). Eight modules
  strict-clean: `output`, `data`, `formulation`, `models._adapter`,
  `models.standard`, `results`, `solve.demand_adjustment`,
  `solve.passthrough`. `problem.py` and `markups.py` remain lax with
  narrow `disable_error_code` lists pending the step-8 split.

### Fixed

- **`Dict_K` / `Dict_Z_formulation` shared class state** (step 6a).
  Previously class attributes, so two concurrent `Problem` instances
  could accumulate each other's state. Now per-instance. Three
  regression tests pin the fix.
- **First-stage correction weight-matrix and sign bugs** (`b3b08a3`,
  pre-v0.4). Carried over from CClean-fixes.

### Deprecated

- `ModelFormulation(...)` now raises `DeprecationWarning`; use the
  class-based `ConductModel` API (`Bertrand`, `Cournot`, `Vertical`, …)
  directly. See the "Migration from v0.3.x" section above.
- `demand_params=dict(sigma=…)` raises `DeprecationWarning`; use
  `demand_params=dict(rho=…)` instead.
- `pyRVtest.output.output()` is a logging-backed compatibility shim and
  emits a once-per-session `DeprecationWarning`. Use
  `logging.getLogger("your.module").info(...)` in new code.
- **Per-model `unit_tax` / `advalorem_tax` / `advalorem_payer`** on
  `ConductModel`, `Vertical`, and `ModelFormulation` are deprecated in
  favor of the Problem-level kwargs. The model-level fields still work
  (and win by legacy precedence when both are set) but emit a once-
  per-session `DeprecationWarning`. **Removal scheduled for v0.7**
  (one release later than other v0.4 deprecations — the per-model tax
  pattern is common enough in existing user code to warrant an extra
  release of runway). Migrate by moving the tax column to
  `Problem(..., unit_tax='col', ...)` and using `unit_tax_salient=False`
  on individual models for salience-test opt-outs.

### Notes for coauthors

- See `.claude/handovers/MEMO_coauthor_updates.md` for a running,
  behavior-focused ledger of v0.4 changes that affect downstream code.
- The v0.4 test suite is at 619 passed + 3 skipped as of v0.4.0rc1 and
  continues to grow.
- Data-dependent regression tests for DMSS yogurt (`step 0d`) and the
  Dearing `LearningFirmConduct` reference (`step 12`) remain blocked on
  external inputs.

## [0.3.2] — prior

See `git log v0.3.2` for pre-v0.4 history. Notable line items from the
CClean-fixes branch merged before the v0.4 refactor started:

- 12 correctness fixes (first-stage correction, sign conventions,
  ownership handling).
- 900× clustering speedup.
- `demand_params` feature for passing demand parameters directly.
- `endogenous_cost_component` support for non-constant marginal cost.
