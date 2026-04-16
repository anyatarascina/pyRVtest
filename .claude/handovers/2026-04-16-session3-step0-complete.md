# Handover: v0.4 review + Step 0 landing + step 1 skeleton

**Date:** 2026-04-16 (third session of the day, continuing from `2026-04-16-v0.4-refactor-design.md`)
**Branches:**
- `CClean-fixes` at `e921649` on origin — Step 0 protection
- `v0.4-refactor` at `7e20ccb` on origin — branched from `CClean-fixes`, step 1 skeleton landed
**Tag:** `v0.3.3-stable` (annotated, pushed) pointing at `47b4457` on `CClean-fixes`
**Status:** v0.4 Step 0 protection landed (except 0d real data) AND v0.4 migration step 1 (module skeleton) committed and pushed. 121 tests pass + 3 skipped on `v0.4-refactor`. Step 2 (extract Products → products.py with type hints + mypy clean) is the next work item.

## TL;DR

1. **Adversarial review of the v0.4 design doc** surfaced a dozen real issues (sequencing, readiness gaps, scope, ambiguities). Review lives at `.claude/plans/review-2026-04-16.md` (also copied to `~/.claude/plans/drifting-wishing-stonebraker.md`). Design doc updated with fixes per Chris's approval.

2. **Step 0 protection built and landed.** 0b (snapshots), 0c (equivalence — already done), 0d (scaffold only), 0e (rollback doc), 0f (property tests with expanded model coverage). 0a (tag `v0.3.3-stable`) pushed to origin.

3. **Test suite grew from 71 to 92** (+ 3 skipped 0d placeholders).

4. **Outstanding:** Lorenzo's input on 0d — data location, pinned paper specification, published TRV/F/MCS values. Scaffold is ready at `tests/replication/test_dmss_yogurt.py`.

5. **Next:** v0.4 migration step 1 — module skeleton per `.claude/plans/v0.4-refactor.md` §5. Chris said "B" ("Start v0.4 step 1") after this handover.

---

## What the review session did

Chris asked for an adversarial review of the v0.4 design doc. The full review is at `.claude/plans/review-2026-04-16.md` — 9 sections covering readiness, sequencing, gaps, scope honest reassessment, and concrete next actions.

Headline findings from the review (all addressed in the updated design doc via commit `f0840c3`):

- **Readiness:** Step 0c was marked DONE but the safety net was hollow without 0b/0d. Made the note prominent.
- **Tag:** Original plan tagged `v0.3.2-stable` at `45bcc4b` — the pre-bug-fix commit. Renamed to `v0.3.3-stable` pointing post-fix.
- **Sequencing — Dearing:** Step 12 (Dearing notation verification) was after step 5 (which implemented `ConstantMarkup`/`RuleOfThumb`/`CostPlus`). Reordered: step 5 ships mechanical classes only; Dearing verification gates the simple-markup classes in step 12.
- **Sequencing — mypy / property tests / `__all__`:** all three moved from late-stage batch steps (17/20/22) to incremental per-step rules.
- **Scope — AFSSZ:** Chris confirmed step 16 is a release blocker for step 25 ("one set of code, once it is ready, we will launch it on AFSSZ and scalable labor").
- **Scope — CI:** Chris decided minimal CI in v0.4 (new step 24.5), one-config GitHub Actions. Full matrix deferred to v0.5.
- **Scope — tiers:** `§2.5` block added listing items as Essential/Strong/Deferable/Cross-cutting. Scope stays maximalist per Chris; tiers are fire-sale order only.
- **Coauthor gates:** Chris chose informal check-ins. No formal blocking gates in `§5`.
- **Decision rule:** Snapshot-update thresholds formalized (≤1e-12 auto / 1e-12 to 1e-7 deliberate / >1e-7 blocks merge).
- **Rollback triggers:** Single-step-breaks / two-step-regression / golden-file-drift triggers spelled out.
- **Labor sign validation:** Added to `§4.5` so upstream sign errors don't produce silent garbage.

Anti-sycophancy honesty note from the review: the prior Claude (two earlier sessions) "flipped three times on scope" by its own admission. Independent re-read concluded the core refactor items (1–10) are load-bearing, but items 17–24 were all added in one flip in response to "is this the best code?" and form a scope-risk cluster. The review proposed trimming 11, 13, 18, 19 to v0.4.1 but Chris kept the maximalist scope. Fine — tiers make the trim order explicit if pressure arises.

---

## What Step 0 landed

### 0a. `v0.3.3-stable` tag (commit `47b4457`, pushed)

Annotated tag describing what's included. Nuclear-revert anchor per plan `§10`. Note: tag was placed while 0d is still a placeholder. If Lorenzo's 0d input changes any numeric output, we may want to create `v0.3.4-stable` rather than moving the tag.

### 0b. Snapshot regression suite (commit `7268925`)

- `tests/_snapshot_helpers.py` — extraction, NaN-aware comparison, REGENERATE_SNAPSHOTS env var.
- `tests/test_snapshots.py` — 6 pinned snapshots:
  - `analytical_base` — user_supplied_markups path
  - `analytical_clustering` — clustering_adjustment=True
  - `analytical_base_fe` — cost absorb='C(firm_ids)'
  - `analytical_scale` — endogenous_cost_component
  - `first_stage_pyblp_path` — the code that had Bug 3 in b3b08a3
  - `first_stage_demand_params_path` — the code that had Bugs 1 and 2
- Mismatch detection verified: a deliberate 1e-5 TRV corruption triggers failure; restore passes.

### 0c. First-stage correction equivalence (prior session, commit `b3b08a3`)

9 tests in `tests/test_first_stage_correction.py`. Already DONE. Caught the three b3b08a3 bugs.

### 0d. DMSS yogurt golden file — **SCAFFOLD ONLY** (commit `47b4457`)

`tests/replication/test_dmss_yogurt.py` with three @pytest.mark.skip placeholders (TRV/F/MCS). Skip reason on each points Lorenzo at the `NEEDED FROM LORENZO` block at the top of the file. Fixture is scoped to module for when it's un-skipped.

### 0e. Rollback procedure

Design doc `§10` was updated during the review to reference `v0.3.3-stable` consistently (was `v0.3.2-stable` in places). Decision rule + rollback triggers added to `§5`.

### 0f. Property tests + expanded model coverage (commits `1bcda1d` + `edcb2e1`)

11 property tests covering all currently-implemented conduct models:

- determinism (Hypothesis)
- perfect_competition markups == 0
- Bertrand markups computed correctly vs DGP (Hypothesis, demand_params path)
- within-market row permutation invariance (Hypothesis)
- DGP seed invariance (Hypothesis)
- dispatch smoke test across all 5 standard models + hand-computed Bertrand/Cournot/Monopoly agreement
- Cournot markups match hand-computed (Hypothesis, max_examples=100)
- Monopoly markups match hand-computed (Hypothesis, max_examples=100)
- profit-weight kappa_specification Bertrand matches hand-computed (Hypothesis, max_examples=30)
- custom 'other' model via custom_model_specification closure
- vertical Bertrand-downstream + Monopoly-upstream smoke

TODO block at end of the file names models that will get property tests when their classes land in v0.4 steps 5 and 12 (RuleOfThumb, ConstantMarkup, CostPlus, NashBargaining, Monopsony, BertrandWages, CournotEmployment, PartialCollusion class).

---

## Commits pushed this session

**Branch `CClean-fixes`** (Step 0 protection):

| Commit | Subject |
|--------|---------|
| `f0840c3` | DOC: incorporate v0.4 review feedback into design doc |
| `1bcda1d` | TEST: v0.4 Step 0f property-based test scaffolding |
| `edcb2e1` | TEST: expand property tests to cover all current conduct models |
| `7268925` | TEST: v0.4 Step 0b snapshot regression suite |
| `47b4457` | TEST: v0.4 Step 0d scaffolding (DMSS yogurt golden file) |
| `e921649` | DOC: session handover + coauthor memo update for Step 0 landing |

Tag `v0.3.3-stable` pushed to origin at `47b4457`.

**Branch `v0.4-refactor`** (branched from `CClean-fixes` at `e921649`):

| Commit | Subject |
|--------|---------|
| `7e20ccb` | REFACTOR: v0.4 step 1 module skeleton + __all__ declarations |

---

## Step 1: DONE (commit `7e20ccb` on `v0.4-refactor`)

Branched from `CClean-fixes` at `e921649`. Delivered:

- 5 new subpackages (`backends/`, `models/`, `instruments/`, `solve/`, plus `backends/labor/`)
- 22 empty skeleton `.py` files, each with `__all__ = []` and a docstring naming the step that will populate it
- `pyRVtest/results.py` → `pyRVtest/results/__init__.py` (verbatim move to resolve the file-vs-directory name collision that would otherwise block step 9)
- `pyRVtest/__init__.py` preserves the full v0.3 public API (paranoid v0.3.3-stable symbol list enforced by tests) and adds namespace re-exports for the new subpackages
- `tests/test_import_roundtrip.py` — 29 tests covering public API importability, v0.3 preservation, parameterized step-1 skeleton coverage (every module imports + has `__all__ == []`), and ProblemResults/Progress accessibility after the subpackage move

Full suite on `v0.4-refactor`: 121 passed + 3 skipped in 2:41 (all 6 Step 0 snapshots still match at atol=1e-10, all 11 property tests still pass — confirms no behavior change from the move).

## What's next — step 2

Per `.claude/plans/v0.4-refactor.md` §5:

> Step 2 | Extract `Products` → `products.py`. No logic change. Add type hints, `mypy --strict` clean for this file. | All step-0 tests + mypy on products.py

**Mechanical tasks:**

1. Create `pyRVtest/products.py` with the `Products` class extracted from `pyRVtest/problem.py` (currently at lines 38-200 approx).
2. Move `_qr_residualize` helper alongside if it's only used by Products (grep to confirm).
3. Update all internal imports: `pyRVtest/__init__.py`, `pyRVtest/problem.py`, any other callers.
4. Add type hints throughout the extracted file. Target `mypy --strict` clean for just this file (PyBLP stubs may be incomplete — fallback is `# type: ignore[attr-defined]` with comments per §7 Open Question 7).
5. Add a mypy check to the test suite or a separate tox env if convenient; per the plan's incremental-mypy rule each new/moved module should ship clean.
6. Run full test suite + import-roundtrip + snapshots.
7. Commit: "REFACTOR: v0.4 step 2 extract Products → products.py with type hints".

**What NOT to do in step 2:**

- Do not change the `Products` API surface. Existing users of `pyRVtest.Products` must still work.
- Do not refactor Products internals or touch its `__new__` logic. The plan calls for extraction only; logic lands in later steps.
- Do not extract anything else from problem.py yet (Models, Problem, etc. come in step 8).

**Incremental mypy note:** this is the first module to get the strict type pass. The cross-cutting rule from §5 of the plan says "every new module added in a step ships with `mypy --strict` clean on that module" — step 2 is the first instance. If PyBLP gaps force `# type: ignore` lines, each should have a short comment explaining the gap so step 17's audit pass knows what to evaluate.

---

## Outstanding items for Chris (end of session)

1. **Send the coauthor memo.** `MEMO_coauthor_updates.md` was updated this session with today's additions. Send to Marco and Lorenzo when convenient.

2. **Lorenzo's 0d inputs.** Without the DMSS yogurt data + pinned values, the strongest single protection (paper-replication golden file) doesn't exist yet. The scaffold is ready to receive his input.

3. **Sanity-check the `v0.3.3-stable` tag placement.** I tagged at the current tip (which has 0d as placeholder). Alternative was to wait for 0d. If you disagree, `git tag -d v0.3.3-stable && git push origin :v0.3.3-stable && git tag v0.3.3-stable <new-commit>` will move it.

---

## Uncertainty flags for the next Claude

1. **Monopoly >= Bertrand element-wise** in the smoke test (tests/test_properties.py) passed on seed=42 alpha=-2.0 but is not theoretically guaranteed for all DGPs. If snapshots or property tests from future migrations fail this assertion, weaken to `np.mean(mono) >= np.mean(bert)`.

2. **Labor sign validation** added to `§4.5` defaults to strict (raises on negative wages). If a real user has legitimately-negative wage-like data (e.g., deviations from a mean), add a `strict_signs=False` opt-out kwarg when step 14 implements this.

3. **`v0.3.3-stable` tag** is at a state where 0d is scaffold only. The plan's strict reading says wait. I tagged now because 0b/0c/0e/0f already give strong protection. If 0d materially changes any output when Lorenzo's inputs arrive, create `v0.3.4-stable` rather than moving the existing tag.

4. **Snapshot files live-diff-friendly but large.** `first_stage_*.json` are ~97KB each. Acceptable now (T=200, J=3 = 600 obs). If a future DGP pushes past T=1000, reconsider format (compressed `.npz`? protobuf?). Today's JSON choice was for diff-review during refactor.

5. **Hypothesis `max_examples=100`** gives coverage but property tests add ~13s to the suite. If the full test suite creeps past 5 minutes during the refactor, drop to 30 examples in a Hypothesis dev profile and keep 100 as a `ci` profile.

6. **Pickle backward compat**: the `pyRVtest.results` move from module → subpackage changes where `ProblemResults` is stored in the pickle header. If any external user has a `.pkl` file created with v0.3.x containing a `ProblemResults`, unpickling under v0.4 may fail with `ModuleNotFoundError: pyRVtest.results` if Python is strict about it. Low-risk (pickled ProblemResults are uncommon, and the class is still reachable at `pyRVtest.results.ProblemResults`), but worth a note in the v0.4 changelog and possibly a compatibility shim in the future.

---

## Files changed this session

**Added:**
- `requirements-dev.txt`
- `tests/test_properties.py`
- `tests/_snapshot_helpers.py`
- `tests/test_snapshots.py`
- `tests/snapshots/analytical_base.json`
- `tests/snapshots/analytical_base_fe.json`
- `tests/snapshots/analytical_clustering.json`
- `tests/snapshots/analytical_scale.json`
- `tests/snapshots/first_stage_pyblp_path.json`
- `tests/snapshots/first_stage_demand_params_path.json`
- `tests/replication/__init__.py`
- `tests/replication/test_dmss_yogurt.py`
- `.claude/plans/review-2026-04-16.md`
- `.claude/handovers/2026-04-16-session3-step0-complete.md` (this file)

**Modified:**
- `.claude/plans/v0.4-refactor.md` — review feedback incorporated
- `.gitignore` — added `.hypothesis/`, `.pytest_cache/`
- `MEMO_coauthor_updates.md` — today's session update

**Tag added:**
- `v0.3.3-stable` pointing at `47b4457`

---

## Key files for the next Claude to read first

1. `.claude/plans/v0.4-refactor.md` — the refactor plan (updated with review feedback). Source of truth for scope, migration steps, decisions, rollback.
2. `.claude/plans/review-2026-04-16.md` — the adversarial review (this session's critique of the plan, with each finding addressed in §VI "Recommended next actions").
3. This handover.
4. `MEMO_coauthor_updates.md` — what coauthors know.
