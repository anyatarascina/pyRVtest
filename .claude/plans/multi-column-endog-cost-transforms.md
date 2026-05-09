# Plan — Multi-column endogenous cost + log-cost demand_adjustment + z^e diagnostic

> **Note on scope.** Earlier work tracked log-cost-with-demand-adjustment as
> a v0.5 followup (`v0.5-followups.md` item 1). The 2026-05-08 review
> reframed both that item and the broader "what about q + q² or scale +
> scope?" question through the DMQSS (2026, "Conduct and Scale Economies")
> Appendix A.4. The conclusion was that these are textbook generalizations
> of explicit paper results and that pyRVtest can ship them in v0.4 final
> as a coordinated feature — paired with a coauthor ask for a one-paragraph
> remark in Appendix B vectorizing Lemma B.1.

## Goal

Generalize pyRVtest's endogenous-cost machinery from a single endogenous
column under linear / log cost transforms to:

1. **Multi-column endogenous cost.** A user can pass
   `endogenous_cost_component=['q', 'q_sq']` (quadratic) or
   `['log_q', 'log_Q_minus']` (scale + scope) and the package handles the
   IV correction, demand-adjustment chain rule, and post-solve diagnostics
   without any further user-side scaffolding.
2. **`costs_type='log' + demand_adjustment=True` properly supported.**
   Replace the existing hard-reject (commit `d6d8b07` on
   `fix/log-costs-with-demand-adjustment`, never merged into v0.4-refactor)
   with the correct chain-rule scaling of `gradient_markups` by
   `f'(p − Δ_m) = 1/(p − Δ_m)`.
3. **`instrument_channels` uses z^e.** Residualize the data-side regression
   on `(g(q̃), w_exog)` instead of raw `(q, w)` when
   `endogenous_cost_component` is set, collapsing the Dearing condition and
   the DMQSS A.4 distinctness check into a single unified diagnostic.

Together, these expose what the DMQSS paper's Appendix A.4 explicitly says
the framework supports — quadratic cost, scale + scope, and arbitrary
linear-in-basis-columns cost regressions — as a first-class API.

## Design principles

1. **Multi-column is a list of column names; transforms are user-precomputed
   columns.** The API stays declarative: the user precomputes `q_sq`,
   `log_q`, `q * x`, etc., as columns and names them. The package never
   evaluates user-supplied callables for `g`. (Cost transform `f` stays
   curated: `'linear' | 'log'`, with an internal `CostTransform` protocol
   to slot future transforms in without API churn — see Appendix A below.)

2. **Linear-in-basis-columns is the parametric scope.** All math
   generalization (multi-column variance lemma, Dearing-on-z^e collapse)
   relies on the cost regression being linear in pre-supplied columns.
   This is the same parametric structure pyRVtest already imposes for
   inference. The diagnostic and the test inherit the same scope.

3. **Single unified diagnostic, not two parallel ones.** `instrument_channels`
   with z^e residualization handles both the constant-MC (Dearing only)
   and non-constant-MC (Dearing + distinctness) cases by absorbing the
   K_endog "consumed" dimensions into the residualization. No separate
   "DMQSS distinctness" method.

4. **Pre-existing K_inst > K_endog gate generalizes automatically.** The
   `_validate_solve_args` check committed in `c84f7b6` derives `K_endog`
   from `len(endogenous_cost_component)` when the type is a list. No
   change needed to the validation gate.

5. **Coauthor confirmation is part of the deliverable.** The variance
   derivation in DMQSS Appendix B is stated for K_endog = 1 (scalar γ_m,
   λ_q ∈ ℝ^{dz}). Generalization to vector γ_m and Λ_q ∈ ℝ^{dz × K} is
   standard multi-endogenous IV but not written out. Coordinate with
   Marco / Mikkel / Lorenzo / Daniel to add a one-paragraph remark in
   Appendix B before tagging v0.4.0 final.

## Out of scope (for this plan)

- **User-supplied cost transforms** via callable. Argued against in 2026-05-08
  discussion: the validation surface is too large for the realistic transform
  set. Internal `CostTransform` protocol provides forward-compatibility for
  v0.5+ if demand emerges (Box-Cox, sqrt, inverse).
- **Non-constant cost with PyBLP-fit demand_results path.** Already supported
  via the unified `compute_demand_adjustment` (step 4d/4e); no new work.
- **Independent v0.5 followup items** in `v0.5-followups.md` not directly
  related to this scope (no-cost-formulation, F-stat collapse on zero-markup
  candidates, §4.1 fixture rewrite).

## Final API surface

```python
# str: existing single-column behavior (unchanged)
Problem(..., endogenous_cost_component='log_quantity')

# list of str: new multi-column behavior
Problem(..., endogenous_cost_component=['log_q', 'log_Q_minus'])
Problem(..., endogenous_cost_component=['q', 'q_sq'])

# costs_type='log' + demand_adjustment=True now works correctly
Problem(...).solve(demand_adjustment=True, costs_type='log')

# instrument_channels uses z^e residualization automatically when
# endogenous_cost_component is set
problem.instrument_channels(column='rival_z2')
# data-side regression now projects on (g(q̃), w_exog), not raw (q, w)
```

## Phases

### Phase 1 — Multi-column `endogenous_cost_component` core (1.5–2 sessions)

Generalize the IV correction and the Problem.__init__ validation to accept
a list of column names.

**Steps:**

1.1 `Problem.__init__` validation
([pyRVtest/problem.py:1316-1343](../../pyRVtest/problem.py:1316)): accept
`Optional[Union[str, List[str]]]`. For each declared column, check it
appears as a single linear term in `cost_formulation`. Reject empty list
with a clear message.

1.2 `pyRVtest/solve/endogenous_cost.py::iv_correct`
([endogenous_cost.py:78-113](../../pyRVtest/solve/endogenous_cost.py:78)):
- `endog_col_idx` (scalar) → `endog_col_indices: List[int]`.
- `endog_col_raw` shape `(N, 1)` → `(N, K_endog)`.
- `endog_hat` shape `(N, 1)` → `(N, K_endog)`.
- `cost_param[m]`: still the K_w-vector of cost-side OLS coefficients,
  but now the last `K_endog` elements are the gamma vector.
- `mc_correction[m]`: `−gamma_m * endog_col_raw` becomes
  `−endog_col_raw @ gamma_m` for `gamma_m ∈ ℝ^{K_endog}` and
  `endog_col_raw ∈ ℝ^{N × K_endog}`.

1.3 `pyRVtest/solve/demand_adjustment.py::_compute_gamma_gradient`
([demand_adjustment.py:483-530](../../pyRVtest/solve/demand_adjustment.py:483)):
- Replace `cp_up[m][-1]` with `cp_up[m][-K_endog:]` (last K_endog
  components are the gamma vector).
- `grad_gamma[inst]` shape `(M, n_theta)` → `(M, n_theta, K_endog)`.

1.4 `pyRVtest/solve/test_engine.py::compute` variance term
([test_engine.py:413](../../pyRVtest/solve/test_engine.py:413)):
- `gradient_gamma[m]` shape `(n_theta,)` → `(n_theta, K_endog)`.
- Variance contraction `endog_col_for_grad @ gradient_gamma[[m], :]`
  becomes `endog_cols_for_grad @ gradient_gamma[m].T` with widened
  shapes; `endog_cols_for_grad` is now `(N, K_endog)`.
- Reconstruct the per-(N, K_endog) endogenous residual matrix,
  applying `_absorb_cost_ids` and `qr_residualize` consistently with
  the single-column case.

1.5 K_effective bookkeeping
([test_engine.py:252](../../pyRVtest/solve/test_engine.py:252)):
- `K_effective = K_inst - 1 if endog_hat is not None else K_inst`
  becomes `K_effective = K_inst - K_endog if endog_hat is not None
  else K_inst`.

1.6 Tests:
- `tests/test_endogenous_cost_multicolumn.py`:
  * Quadratic cost fixture: `c = γ_1 q + γ_2 q² + w'τ + ω`. Two
    instruments rich enough to identify both, third for testing.
    Hand-derive the 2SLS coefficients and TRV.
  * Scale + scope fixture: `log(c) = γ_1 log(q) + γ_2 log(Q_minus)
    + w'τ + ω`. Three instruments. Hand-derive.
  * Cross-validate: K_endog = 1 with a list of length 1 produces
    bit-identical output to K_endog = 1 with a string (no regression).

**Deliverables:** generalized `endogenous_cost_component` API on `Problem`
plus full multi-column IV correction, demand-adjustment gradient, and
variance term; tests on quadratic and scale+scope fixtures; passes existing
single-column snapshots.

### Phase 2 — log-cost + demand_adjustment chain rule (0.5–1 session)

Replace the hard-reject path with the proper chain-rule scaling.

**Steps:**

2.1 `Problem.solve` log transform location
([pyRVtest/problem.py:1724-1733](../../pyRVtest/problem.py:1724)):
- Drop the `not demand_adjustment` gate on the log transform.
- Apply `marginal_cost = np.log(marginal_cost)` whenever
  `costs_type='log'`.

2.2 `Problem.solve` demand_adjustment block
([pyRVtest/problem.py:1798-1808](../../pyRVtest/problem.py:1798)):
- Pass `marginal_cost_level` (the pre-log values) to
  `compute_demand_adjustment` so it can compute `f'`.

2.3 `compute_demand_adjustment` cost-transform scaling:
- After computing `gradient_markups` (which represents `∂Δ_m/∂θ`),
  apply `f'(p − Δ_m)` element-wise:
  * `costs_type='linear'`: `f' = 1`, no rescaling (today's behavior).
  * `costs_type='log'`: `f' = 1/(p − Δ_m)` element-wise; rescale
    `gradient_markups[m]` by `1 / mc_level[m][:, None]`.
- This is the chain rule for `∂ω_m/∂θ = f'(p − Δ_m) · (−∂Δ_m/∂θ)`
  applied at the moment level.

2.4 Remove the hard-reject in
[pyRVtest/problem.py:1724](../../pyRVtest/problem.py:1724) introduced by
`fix/log-costs-with-demand-adjustment`. Delete the `NotImplementedError`
and the `TestLogCostsWithDemandAdjustmentRejected` test it pinned. The
branch can be deleted from origin once this lands.

2.5 Tests:
- `tests/test_log_cost_demand_adjustment.py` (or extend existing
  `test_demand_adjustment.py`):
  * Tiny log-cost fixture, hand-compute TRV and F under
    `demand_adjustment=True`.
  * Cross-path parity (`demand_results` vs `demand_params`) under
    log + demand_adjustment.
  * Compare against the analog `costs_type='linear'` run on an
    equivalent linear-cost fixture (sanity).

**Deliverables:** `costs_type='log' + demand_adjustment=True` produces
correct demand-adjusted variance; old hard-reject branch retired; tests.

### Phase 3 — `instrument_channels` z^e residualization (0.5–1 session)

Use DMQSS-style residualization in the data-side regression and FWL
partialling so the diagnostic unifies Dearing + A.4 distinctness.

**Steps:**

3.1 Compute first-stage prediction `q̃` for the endogenous columns.
Already produced by `iv_correct` as `endog_hat` (shape `(N, K_endog)`).
Surface it through `compute_instrument_channels`'s arguments so the
function can use it.

3.2 `pyRVtest/solve/passthrough.py::compute_instrument_channels`
([passthrough.py:1221](../../pyRVtest/solve/passthrough.py:1221)):
- Change the data-side `_ols_slope_partialled(prices, z, w)` to
  partial on `(g(q̃), w_exog)` when `endogenous_cost_component` is
  set. Add a helper `_residualize_on_first_stage(z, q_hat, w_exog)`
  that handles the K_endog ≥ 1 case uniformly.
- For the FWL `β_m` regression of `Δ_m` on `z` given `p`, use the
  same residualization on `(g(q̃), w_exog)` rather than just `p`.

3.3 Methodology footer text update
([passthrough.py:1105-1122](../../pyRVtest/solve/passthrough.py:1105)):
when endogenous_cost_component is set, mention that the data-side
regression uses z residualized on `(g(q̃), w_exog)` (DMQSS z^e),
collapsing the Dearing and DMQSS distinctness checks into a single
diagnostic.

3.4 Docstring update: add to
`Problem.instrument_channels` docstring a "When endogenous cost is
set" section explaining the unification.

3.5 Tests:
- Constant-MC case (no endogenous_cost_component): output bit-
  identical to the current implementation. Existing tests pass.
- Non-constant-MC case: data-side magnitude differs from the raw
  `(q, w)` projection in expected ways. Cross-validate against a
  hand-derived 2-product 2-market fixture where `q̃` is computable
  in closed form.
- Synthesize a degenerate case (paper A.5: two cost shifters of the
  same rival) and confirm the structural-side magnitude collapses
  to near-zero on z^e even though it would be nonzero on raw z.

**Deliverables:** `instrument_channels` outputs the DMQSS-correct
diagnostic when endogenous_cost_component is set; constant-MC behavior
unchanged; methodology footer documents the residualization.

### Phase 4 — Documentation (0.5–1 session)

4.1 `docs/advanced_features.rst` Pass-through section: add a
"Non-constant marginal cost" subsection explaining the unified
diagnostic. Include a worked example with a synthetic scale-economies
fixture showing how the z^e residualization collapses the Dearing and
distinctness conditions.

4.2 `docs/math.rst`: add a section "Pass-through diagnostic under
non-constant marginal cost" stating the rank-K+1 condition decomposition
into first-stage rank K + Dearing on z^e, with the parametric-linearity
argument for why the moment factors cleanly.

4.3 `docs/faq.rst`: add a "How does the pass-through diagnostic interact
with non-constant cost?" entry pointing at the new advanced_features
subsection. Also add a "What if I have q + q² or scale + scope?" entry
showing the multi-column API.

4.4 `docs/migrating_to_v0.4.rst`: note the multi-column API in the
"new in v0.4 final" section.

4.5 CHANGELOG entry under `[0.4.0]` summarizing Phases 1–3 + 4.

**Deliverables:** documentation reflecting the unified diagnostic and
the multi-column API.

### Phase 5 — Coauthor coordination (external)

5.1 Send Marco / Mikkel / Lorenzo / Daniel a short note covering:
- The natural generalization of Lemma B.1 to vector `γ_m` and
  `Λ_q ∈ ℝ^{dz × K}`.
- The rank-K+1 decomposition: first-stage rank K + Cov(z^e,
  f(p − Δ_m) − f(p − Δ_0)) ≠ 0.
- The argument that the parametric-linearity of the cost regression
  in basis columns makes the constants `γ_{m,k}` come out of the
  moment-level covariance, leaving the residualized-instrument
  Dearing condition.
- Ask: add a one-paragraph remark in Appendix B confirming the
  vector-γ generalization, ideally before tagging v0.4.0 final.

5.2 Coordinate the paper update with the package release.

**Deliverables:** paper-side remark added before final tag; pyRVtest
references the remark in the methodology footers.

## Test strategy

- **Phase 1 multi-column:** quadratic and scale+scope fixtures with
  hand-derived expected values; existing single-column snapshots remain
  bit-identical.
- **Phase 2 log + demand_adjustment:** tiny hand-derived log-cost fixture;
  cross-path parity test (PyBLP results vs `demand_params`).
- **Phase 3 z^e:** redundancy fixture (paper A.5 example) where structural
  magnitude collapses on z^e but not on raw z.
- **End-to-end:** synthetic example with two endogenous cost columns plus
  three testing instruments; full workflow runs to completion.
- **Snapshot regression:** existing `analytical_base`, `analytical_scale`,
  `first_stage_*`, `nested_logit_vertical` snapshots remain within tolerance.

## Open design questions

1. **List uniformity.** When `endogenous_cost_component` is a list, must
   all columns enter the cost formulation as plain linear terms (no
   transforms applied by the package)? Probably yes — the user
   precomputes any transforms, package handles linear-in-columns 2SLS.
   **Resolved (yes).**

2. **Composite IV-set declarations.** When K_endog > 1, should the
   per-instrument-set documentation surface specify which IVs target
   which endogenous? Probably no — the test uses all instruments jointly,
   the rank condition handles which combinations are identifying.
   **Resolved (no).**

3. **Cost transform `f` validation.** Should `costs_type='log'` reject
   negative implied costs across all candidates as today, or warn-and-
   continue? Today's hard-reject is the right behavior. No change.
   **Resolved.**

4. **Methodology footer text.** Should the printed output explicitly
   say "z^e" or paraphrase as "instrument residualized on first-stage
   prediction of endogenous cost variables"? Paraphrase preferred for
   accessibility; mention z^e as the technical term in parentheses.
   **Resolved.**

## Risks

1. **Variance-derivation extension beyond paper.** Lemma B.1 is stated
   for K_endog = 1. Multi-column generalization is standard multi-endog
   IV but not paper-explicit. Mitigation: coordinate with coauthors
   (Phase 5); add a remark to Appendix B before tagging.

2. **Snapshot churn.** The variance pipeline change in Phase 1 could
   produce numerical drift on existing single-column snapshots.
   Mitigation: K_endog = 1 with a list of length 1 should produce
   bit-identical output to K_endog = 1 with a string; verify before
   merge. If drift appears, investigate before relaxing tolerances.

3. **Log-cost catastrophic cancellation.** The `1/(p − Δ_m)` rescaling
   in Phase 2 amplifies noise when implied markups are close to prices
   (small implied costs). This is a known issue with log-cost
   regressions. Mitigation: document the regime in `docs/faq.rst` and
   leave the existing negative-mc rejection in place.

4. **Coauthor coordination delay.** Phase 5 is external and may not
   land before v0.4.0 final tag. Mitigation: document in CHANGELOG
   that the multi-column variance is the standard multi-endog IV
   generalization; ship the code with a docstring note that the
   formal Appendix B remark is pending.

5. **`fix/log-costs-with-demand-adjustment` branch retirement.** The
   branch's `NotImplementedError` is replaced by the proper fix in
   Phase 2. Mitigation: delete the origin branch after Phase 2 lands;
   note in CHANGELOG and v0.5-followups that item 1 is closed.

## Estimated effort

| Phase | Sessions |
|---|---|
| 1 — Multi-column endogenous_cost core | 1.5–2 |
| 2 — log + demand_adjustment chain rule | 0.5–1 |
| 3 — instrument_channels z^e residualization | 0.5–1 |
| 4 — Documentation | 0.5–1 |
| 5 — Coauthor coordination | external |

**Total: 3–5 sessions of code, plus paper-side remark.**

## Incremental shipping plan

Each phase can ship as its own merge into `v0.4-refactor`:

- **rc3 (or appended to current rc2): Phase 1.** Multi-column endogenous
  cost API. Researchers running quadratic / scale-and-scope use cases get
  the new API; constant-cost users see no change.
- **+ Phase 2.** Log-cost demand-adjustment combination newly supported.
- **+ Phase 3.** `instrument_channels` produces the unified diagnostic
  under non-constant cost.
- **v0.4.0 final: + Phase 4.** Documentation lands; coauthor remark
  ideally in place via Phase 5.

## Acceptance criteria for v0.4 final tag

- `endogenous_cost_component` accepts `Optional[Union[str, List[str]]]`.
  Quadratic-cost and scale+scope fixtures produce hand-validated TRV / F.
- `costs_type='log' + demand_adjustment=True` runs with correct
  demand-adjusted variance; cross-path parity ≤ atol = 5e-9 to match
  existing endog-cost cross-path test.
- `instrument_channels` uses z^e residualization under
  endogenous_cost_component; constant-MC behavior unchanged.
- The `K_inst > K_endog` validation (already shipped in `c84f7b6`) auto-
  generalizes to lists via `len(...)`.
- Documentation: advanced_features non-constant-cost subsection;
  math.rst rank decomposition section; FAQ entries; CHANGELOG entry.
- Coauthor remark in Appendix B (Phase 5) — ideal but not blocking the
  code release.
- Existing snapshot tests bit-identical or within tolerance.
- All Phase 1 / 2 / 3 tests pass.

## Coauthor-coordination notes

The paper update is a single-paragraph remark vectorizing Lemma B.1 and
stating the rank-K+1 decomposition. Concretely:

> **Remark (vector γ_m).** Lemma B.1 extends to the multi-endogenous-
> variable case `K_endog ≥ 1` by component-wise replacement of scalar
> quantities with vectors and `(dz × K_endog)` matrices: `(γ_m − γ̂_m)`
> becomes a `(K_endog,)`-vector, `λ_q ∈ ℝ^{dz}` becomes
> `Λ_q ∈ ℝ^{dz × K_endog}`, and `q^e ∈ ℝ` becomes `q^e ∈ ℝ^{K_endog}`.
> The proof steps go through with each component being `O_p(n^{−1/2})`
> and the contraction being `O_p(n^{−1})`. The rank-K+1 condition in
> Appendix A.4 then decomposes into first-stage rank K (instruments
> identify the K endogenous variables) plus
> `Cov(z^e, f(p − Δ_m) − f(p − Δ_0)) ≠ 0`, where
> `z^e = z − Λ̂_q g(q̃) − Λ̂_w w` is the DMQSS-residualized instrument.

This corresponds to the implementation that pyRVtest ships.

## Appendix A — Internal `CostTransform` protocol (forward-compatibility)

To slot future cost transforms (Box-Cox, sqrt, inverse) in v0.5+ without
API churn, refactor the `costs_type` string into an internal
`CostTransform` protocol with `.forward(c)` and `.derivative(c)`. Public
API stays `costs_type: str`; the dispatcher maps strings to private
implementations:

```python
class _CostTransform(Protocol):
    def forward(self, c: Array) -> Array: ...
    def derivative(self, c: Array) -> Array: ...

class _LinearCost(_CostTransform):
    def forward(self, c): return c
    def derivative(self, c): return np.ones_like(c)

class _LogCost(_CostTransform):
    def forward(self, c): return np.log(c)
    def derivative(self, c): return 1.0 / c

_REGISTRY = {'linear': _LinearCost(), 'log': _LogCost()}
```

This makes Phase 2's chain-rule scaling transform-agnostic
(`f_prime = transform.derivative(mc_level)`). Future v0.5+ additions
(`'box_cox'`, etc.) only touch the registry. **Not user-callable.**

## Tracking

- Linked v0.5-followups item 1 (log-cost-with-demand-adjustment): closes
  with Phase 2.
- Linked v0.5-followups item 2 (no-cost-formulation): unchanged, still
  v0.5+.
- Linked v0.5-followups item 3 (demand_adjustment + endogenous_cost):
  already closed by step 4d/4e (commit `9cceefd` on 2026-05-08 corrected
  stale docs). Item should be marked done in `v0.5-followups.md`.
