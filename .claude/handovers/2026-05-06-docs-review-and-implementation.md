# Handover — 2026-05-06 — Docs critical review and full implementation pass

**Branch:** `v0.4-refactor` at `7d7fd3c` (pushed; clean working tree).
**Working directory:** `/Users/christopher.sulli1/Library/CloudStorage/Dropbox/Economics/claude/pyrvtest/`
**Remote:** `git@github.com:anyatarascina/pyRVtest.git`

Predecessor handovers (read in order if context-loading):

1. `.claude/handovers/2026-05-05-merge-wave-comment-cleanup-docs-prep.md`
2. `.claude/handovers/2026-05-05-docs-phase1-quickstart-and-intro.md`

This is the second 2026-05-06 session (the first was Opus 4.7 1M context
which executed Phases 1 final / 2 / 3 / 4 / 5 of the docs overhaul plus
post-audit cleanup; commits `1f06203 → ef3d7a4`). Chris asked me to do a
critical review of the rendered Read the Docs site and then implement
the agreed fixes.

## Session goals

1. Build the rendered docs locally so we can actually look at them.
2. Critical review of the docs site against peer-package conventions.
3. Implement the prioritized fix list.
4. Resolve the local pyblp / numpy environment so the legacy notebooks
   can be re-executed against the v0.4 API.

All four landed.

## Commits this session

```
7d7fd3c DOC: mark v0.5 followup #6 (notebook rewrites) as completed
a283c48 DOC: rewrite cereal-data + model-library notebooks to v0.4 API
f70668c DOC: compress migration guide (703 -> 413 lines)
43e68ba DOC: add advanced features tutorial
2321e5a DOC: enumerate instruments + backends in API; fix conf.py sys.path
52e0515 DOC: drop duplicate Workflow code from introduction.rst
848f687 DOC: tier docs sidebar into User / Developer documentation
49a3403 DOC: fix stale EXAMPLE_SYNTHETIC docstring (Bertrand -> PC)
```

All pushed to `origin/v0.4-refactor`.

## Build env fixes (uncommitted, but persist on the dev box)

* **Pandoc installed via Homebrew** so `nbsphinx` can convert notebooks
  during the docs build. Previously, `sphinx-build` aborted with
  `nbsphinx.NotebookError: PandocMissing`; build now succeeds.
* **pyblp upgraded 1.1.0 → 1.2.0.** pyblp 1.1.0 had `np.unicode_` at
  `pyblp/configurations/formulation.py:381`, which numpy 2.0 removed.
  Local environment had numpy 2.3.4 + pyblp 1.1.0 (the wrong-paired
  combo); CI exercises both supported lanes (`numpy<2 + pyblp<1.2` and
  `numpy>=2 + pyblp>=1.2`). Upgrading pyblp to 1.2 lands us in the
  numpy-2 lane and unblocks notebook execution.
* **pyRVtest re-installed in editable mode.** `pip install -e .` in the
  repo root, after manually removing the leftover
  `/Library/Frameworks/.../site-packages/pyRVtest/` directory that pip's
  uninstall of the old non-editable v0.3.2 release missed (a stale
  `data/` and `__pycache__/` were left behind, turning `pyRVtest` into
  a broken namespace package on import).

## Critical review

I built the docs locally (`sphinx-build -E docs /tmp/sphinx-html`) and
reviewed every user-facing page. Compared against pyblp / sklearn /
statsmodels / linearmodels / pymc via the Explore agent. Wrote a
five-question critical review covering organization, feature coverage,
tutorial sufficiency, API completeness, and best-practice conventions.

Headlines from the review:

1. **Sidebar mixed user and contributor docs.** `agent_guide.rst` (822
   lines) sat directly beside Tutorial and FAQ in the main toctree.
   pyblp uses an explicit Developer Documentation tier; we should too.
2. **End-to-end code triplicated.** introduction.rst's Workflow section
   (~30-line code block) + the README quick-start (which introduction
   `.. include::`s) + tutorial.rst's step-by-step walkthrough = three
   places showing nearly the same example.
3. **migrating_to_v0.4.rst was 703 lines.** pyblp's "Version Notes" for
   major refactors are ~150 lines. Heavy explanatory bloat around tax
   precedence, labor-side backend status, in-package demand estimation
   tradeoffs.
4. **v0.4's headline new features had no working tutorial.** Multi-IV,
   `demand_adjustment`, clustering, `F_reliability_summary`, pass-through
   diagnostics, Problem-level taxes, and `endogenous_cost_component`
   were all documented at the API level only.
5. **API page had two thin spots.** `pyRVtest.instruments` got one
   autosummary line; the actual constructors (Hausman, Bartik, BLP,
   differentiation IVs, rival sums) were not listed. `pyRVtest.backends`
   was unlisted entirely despite FAQ referencing
   `LaborSupplyBackend` as user-visible.
6. **Two notebooks (`testing_firm_conduct.ipynb`, `model_library.ipynb`)
   taught the v0.3 deprecated API.** 17 and 31 hits respectively.
   Banner cells warned readers but executed cells still threw
   DeprecationWarnings. Tracked as v0.5 followup #6, blocked by the
   local pyblp env mismatch.
7. **Stale docstring.** `pyRVtest/data/__init__.py` Attributes-block
   said EXAMPLE_SYNTHETIC was "Bertrand-equilibrium" but the actual
   data is PC truth. Leftover from the 2026-05-05 design iteration.

## What got committed

In execution order (each its own commit):

### `49a3403` — stale docstring fix

One-line rewrite of the EXAMPLE_SYNTHETIC Attributes-block description.

### `848f687` — sidebar tiered

`docs/index.rst` toctree split into "User documentation" (10 pages) and
"Developer documentation" (1 page: agent_guide). agent_guide is
preserved on the rendered site but visually demarcated. No cross-refs
broken.

### `52e0515` — drop intro Workflow code

introduction.rst's 30-line example deleted; the 4-stage prose
description retained, plus a pointer to tutorial.rst for the actual
code. Two code locations remain (README quick-start + tutorial
step-by-step) instead of three.

### `2321e5a` — API enumeration + conf.py path fix

api.rst:

* "Instrument helpers" section: replaced bare `instruments` autosummary
  with five enumerated constructors split product / labor side
  (`rival_sums`, `differentiation_ivs`, `blp_instruments`, `hausman`,
  `bartik`). Notes the deliberate absence of `concentration_hhi` on
  the labor side (HHI is a function of endogenous shares).
* New "Demand backends" section between Demand estimation and Conduct
  models. Lists `DemandBackend` protocol, `SupportsDemandAdjustment`
  mixin, and the four concrete backends (`PyBLPBackend`, `LogitBackend`,
  `NestedLogitBackend`, `UserSuppliedBackend`). Calls out the
  experimental `LaborSupplyBackend` with the v0.4 NotImplementedError
  caveat.

conf.py:

* `sys.path.insert(0, str(source_path.parent))` so autosummary imports
  the dev tree regardless of cwd or whether pyRVtest is installed in
  editable mode. Before this, sphinx running from `docs/` (as
  `tox -e docs` does) was finding the stale `site-packages/pyRVtest/`
  v0.3.2 install which lacks `backends` and crashing the build with
  `ModuleNotFoundError: No module named 'pyRVtest.backends'`.

Build went from 88 warnings to 16 after this fix; the new entries
render with proper signatures.

### `43e68ba` — advanced features tutorial

New page `docs/advanced_features.rst` (373 lines, 6 sections),
inserted in the toctree between in_package_demand and migrating_to_v0.4.
Sections:

1. Multiple instrument sets — list passing + indexed result arrays.
2. Demand adjustment + clustering — both flags, verbatim TRV shift on
   the shipped example (TRV(B,C) 6.894 → 7.013).
3. F-stat reliability inspection — `F_reliability_summary()` DataFrame,
   column-by-column reading guide.
4. Problem-level taxes — `unit_tax` / `advalorem_tax` / `advalorem_payer`
   on Problem; `unit_tax_salient=False` / `advalorem_tax_salient=False`
   per-model opt-outs for salience tests.
5. Pass-through diagnostics — describes API + the v0.4 limitation to
   Vertical models. Numerical pass-through for Bertrand / Cournot /
   etc. is on the v0.5 roadmap; closed forms are in `docs/math.rst`.
6. Endogenous cost components — describes API + DMQSS first-stage
   correction; notes the shipped example doesn't motivate it (linear
   costs in z1/z2 only); points at `tests/test_analytical.py` and
   `tests/test_demand_adjustment.py` for working fixtures with
   `log_quantity` as the endogenous component.

Sections 1–4 were smoke-tested end-to-end on the shipped example and
the prose outputs are verbatim from those runs.

### `f70668c` — migration guide compression (703 → 413 lines, 41% trim)

Editorial pass keeping every Before / After recipe verbatim. Trimmed:

* "Standard oligopoly" merged into a single Before/After block covering
  Bertrand / Cournot / Monopoly / PerfectCompetition (was four
  mini-sections with redundant prose).
* Tax-precedence tiebreaker subsection: ~62 lines → 1 paragraph.
  Deterministic rule covered in 5 lines instead of three example blocks.
* Tax migration "Before / Intermediate / After" trio compressed to
  Before / After.
* Known-coefficient cost shifters: removed the standalone `.. note::`
  about why salience flags don't apply (single sentence in prose now).
* Labor-side: 78 lines → ~30, with column-name conventions and
  backend-status details deferred to FAQ and API.
* In-package demand estimation: kept the pipeline-collapse Before/After,
  pointed at in_package_demand.rst for the standalone-vs-inline
  trade-offs and nested-logit variant.
* Per-section intro paragraphs trimmed (the section header + Before/After
  labels carry the structural signal).

All :ref: anchors preserved (`vertical-migration`, `tax-migration`,
`tax-precedence-tiebreaker`).

### `a283c48` — notebook rewrites to v0.4 API

Both legacy notebooks rewritten and re-executed. Closes v0.5 followup #6
ahead of schedule.

`testing_firm_conduct.ipynb` (Nevo cereal + PyBLP demand):

* Bertrand vs Monopoly comparison: `pyRVtest.Bertrand(ownership='firm_ids')`,
  `pyRVtest.Monopoly(ownership='firm_ids')`, passed via `models=[...]`.
* Five-model run with two Vertical entries: standard
  Bertrand / Cournot / Monopoly + two
  `pyRVtest.Vertical(downstream=Monopoly, upstream=Bertrand|Monopoly)`.
* Tax test: tax columns moved to Problem level. Models 2 and 3 opt out
  via `unit_tax_salient=False`, `advalorem_tax_salient=False` to
  reproduce the pre-v0.4 spelling where models 2/3 simply omitted
  per-model tax kwargs. Model 1 keeps the rule-of-thumb
  `cost_scaling='lambda'`.
* v0.3 deprecation banner cell removed.
* "Detailed inputs" markdown updated to describe `models=` and the
  class-based API.
* "Testing with Taxes" markdown updated to document the v0.4
  Problem-level tax kwargs and per-model salience flags.

`model_library.ipynb` (catalogue of every model class):

* Per-model class examples updated:
  - Bertrand / Cournot / Monopoly: standard `Foo(ownership='firm_ids')`.
  - Bertrand-with-profit-weights → `pyRVtest.PartialCollusion(...)`.
  - Cournot-with-profit-weights → `pyRVtest.Cournot(kappa_specification=...)`
    (the standard class accepts the kwarg).
  - Non-profit conduct → `pyRVtest.PartialCollusion(...)` with custom
    kappa.
  - Marginal-cost pricing → `pyRVtest.PerfectCompetition()`.
  - Rule-of-thumb → `pyRVtest.RuleOfThumb(phi=...)` for the uniform
    case, with `PerfectCompetition(cost_scaling='col')` retained for
    per-product variation.
  - Bertrand with scaled costs → `pyRVtest.Bertrand(cost_scaling='col')`.
  - Constant markup → `pyRVtest.ConstantMarkup(markup=...)`.
  - Vertical → `pyRVtest.Vertical(downstream=..., upstream=...,
    vertical_integration='vi_id')`.
  - Custom conduct → `pyRVtest.CustomConductModel(markup_fn=callable, ...)`.
    Critically, the v0.3 string-formula spelling becomes a Python
    callable; Bertrand recovered as
    `-np.linalg.inv(ownership * response) @ shares`.
  - User-supplied markups → `pyRVtest.UserSuppliedMarkups('col')`.
* v0.3 banner removed.
* Per-model section intros updated to point at the v0.4 class names.
* Custom-models section now distinguishes `CustomConductModel` (callable)
  from `UserSuppliedMarkups` (precomputed column).

Notebook execution: ran via `python3 -m jupyter nbconvert --to notebook
--execute --inplace ...` after the editable install + namespace-package
cleanup. Both notebooks re-executed cleanly with the v0.4 API.

### `7d7fd3c` — v0.5 followup tracker

Marked item 6 (notebook rewrites) as completed in
`.claude/plans/v0.5-followups.md`.

## Open work

### v0.5 followup tracker (5 items left, was 6)

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
   PC-vs-anything pairs show suspiciously small F-stats even when TRV
   is sharp. Tracked since 2026-05-05; the README quick-start narrative
   surfaces this as the canonical example.

### Items in the previous session's tracker that are now resolved

* Item 6 (notebook rewrites) — closed by `a283c48`.

### Possible next docs work

The original 5-question critical review is now fully addressed at the
substantive level. Smaller polish items if you want to keep going:

* **Notebook outputs differ slightly from the previous executions**
  because the underlying math hasn't changed but minor numerical
  details (PyBLP convergence iterations, formatting widths) shift on
  fresh runs. Worth a once-over to confirm nothing surprising landed
  in the saved outputs.
* **Build-time autosummary noise.** Sphinx still emits
  `failed to import pyRVtest.<X>` warnings for the class-based
  autosummary entries (16 warnings, down from 88, mostly cosmetic).
  Root cause is the `:template: class_with_signature.rst` template
  needing per-class metadata pyRVtest doesn't provide. Could be cleaned
  up by writing a custom autosummary template or adding the metadata
  but doesn't affect rendered output.
* **Remove the rstcheck `:ref:` false positives** by either standardizing
  on Sphinx-friendly anchor syntax or adding a `[tool.rstcheck]`
  ignore for `:ref:` directives.
* **References cross-ref case-mismatch warnings.** Six of the build's
  16 warnings are `undefined label: 'references: dearing, magnolfi,
  ...'` — Sphinx is matching labels case-insensitively but emitting
  warnings for the lowercase form. Cosmetic; could be silenced by
  adding `suppress_warnings = ['ref.ref']` to conf.py or fixing the
  cross-ref case in introduction / math / README / advanced_features.

## Notable surprises / discoveries

1. **pyblp 1.1.0 was already broken on numpy 2.3.4** locally. The
   surprise was that a simple version upgrade (1.1.0 → 1.2.0) fixed it,
   and that the test-suite warning count dropped from 560 → 13 in the
   process — many of those warnings were tied to pyblp 1.1.0's own
   deprecation usage, not pyRVtest's.
2. **Stale site-packages/pyRVtest/ directory** persisted after `pip
   install -e .` because pip's uninstall of the old non-editable v0.3.2
   release left behind a `data/` directory and `__pycache__/`. Python
   then loaded `pyRVtest` as a *namespace package* from that stale
   directory, which has no `backends`, `models`, or class attributes
   — so the dev tree was effectively shadowed by an empty stub. Cleared
   by `rm -rf` on the leftover. Worth noting because the same trap
   could bite a future dev who does `pip install -e .` in the same env.
3. **CWD shifted to /tmp** mid-session because of how Bash inherits
   the working directory across persistent shell calls — an early
   `cd /tmp && sphinx-build` left subsequent commands running there
   silently. Caught when an `nbconvert` call reported "no such file."
   Worth using absolute paths or explicit `cd /Users/.../pyrvtest` at
   the start of each command going forward.
4. **`autosectionlabel_prefix_document = True`** in conf.py creates
   labels like `references:Duarte, Magnolfi, ...` (no space after
   colon), but the cross-refs across the codebase use
   `:ref:`references: Duarte, ...`` *with* a space. This pre-existing
   case-mismatch produces warnings in every build. Sphinx falls back
   to a case-insensitive match so the links DO work, but the warnings
   are persistent.
5. **rstcheck does not understand `:ref:`** (Sphinx-specific) — every
   `.. _label:` anchor referenced via `:ref:` produces an "INFO"
   "hyperlink target not referenced" message which rstcheck then exits
   non-zero on. Sphinx itself is happy.
6. **`pyRVtest.passthrough_matrix` and `passthrough_comparison`
   currently require all candidate models to be Vertical** — so the
   advanced_features.rst Pass-through section had to describe the API
   and the limitation rather than show a working example. Numerical
   pass-through for non-Vertical models is on the v0.5 roadmap. The
   README quick-start narrative *does* talk about pass-through (the
   Dearing degeneracy result), but only in terms of theory; no
   diagnostic call demonstrates it on the shipped data.

## Pickup for next session

1. **Read this handover** + the two prior 2026-05-05 handovers if you
   need the full context.
2. **Verify the dev environment.** Quick `python3 -c "import pyRVtest;
   print(pyRVtest.__version__, pyRVtest.__file__)"` should print
   `0.4.0rc1` and a path inside the repo. If it shows
   `site-packages/...` or returns `None` for `__file__`, repeat the
   editable install + stale-dir cleanup from the "Build env fixes"
   section above.
3. **Decide on the v0.5 followups list.** Item 5 (F-stat collapse under
   zero markup) is the most user-visible — it's the surprising cell in
   the README quick-start. Worth looking into now, or formally
   document as a known limitation in the FAQ and defer to v0.5.
4. **Optional polish** items from "Possible next docs work" above —
   the autosummary warnings and references-label case mismatches are
   cosmetic but persistent. ~30 min total to silence both.
5. **Consider the v0.4 release.** The full Phase-1-through-5 docs work
   plus today's review-and-implement pass means the v0.4 docs site is
   in shippable shape. Outstanding releases gates from earlier
   handovers (CI matrix green, RC1 verified, etc.) should be re-checked
   against `7d7fd3c`.
