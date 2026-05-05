# Docs overhaul plan

Status: drafted 2026-05-05, partial answers from Chris recorded below.

## Chris's directives so far (2026-05-05)

| Q | Question | Decision |
|---|---|---|
| 1 | README.md vs README.rst | **No new README.md.** Instead, made the AGENTS.md pointer prominent in README.rst (added a "Reader's guide" section) and added a top-of-file pointer in CLAUDE.md to AGENTS.md. ✓ Done. |
| 2 | Citation: add DMQSW (NBER w32863) + DMQSS non-linear cost | **Yes — cite all three methodology papers.** Updated README citation block and `docs/references.rst` with separate entries for DMSS 2023, DMQSW 2026 (NBER w32863), and DMQSS 2026 (non-linear cost). ✓ Done. **OPEN: title and URL for the DMQSS non-linear cost paper.** |
| 3 | First-time users vs collaborators | **First-time users.** |
| 4 | Conceptual depth | **Tight 3-page summary**, not a long pedagogical intro. |
| 5 | MEMO_*.md at root | Not yet answered. |
| 6 | Notebooks | Not yet answered. |

## Current state (audit)

**Sphinx / RTD** (`docs/`):
- `index.rst` — 25 lines, toctree (introduction, tutorial, migrating_to_v0.4, custom_demand, agent_guide, api, references, legal)
- `introduction.rst` — 9 lines, **stub** (just includes README.rst from `docs-start` marker)
- `tutorial.rst` — 12 lines, **stub** (just toctrees the two notebooks)
- `legal.rst` — 3 lines, includes LICENSE.txt
- `migrating_to_v0.4.rst` — 648 lines, **rich** ✓
- `custom_demand.rst` — 233 lines, **rich** ✓
- `agent_guide.rst` — 825 lines, **comprehensive but maybe too detailed** for most readers
- `api.rst` — 100 lines, autodoc-only
- `references.rst` — 61 lines, bibliography
- Notebooks: `testing_firm_conduct.ipynb`, `model_library.ipynb`

**Top-level** (repo root):
- `README.rst` — 73 lines, the public-facing intro. Cited from PyPI long_description and from `introduction.rst`.
- `CHANGELOG.md` — 456 lines, comprehensive ✓
- `LICENSE.txt`, `MEMO_*.md` (4 memos, design / handover content)
- **No** `README.md` — only RST
- **No** `CONTRIBUTING.md`

**Agent-facing**:
- `CLAUDE.md` — 94 lines, Claude Code project guide (architecture overview, key classes, module layout)
- `AGENTS.md` — 392 lines, larger living contract for coding assistants and contributors
- `docs/agent_guide.rst` — 825 lines, included in RTD; longest of the three
- Programmatic access via `pyRVtest.show_agent_guide()`

**In-package**:
- Docstrings: good coverage on public API; sparser on internals
- The 2026-05-05 comment sweep dropped ~140 editorial markers; substantive docstrings remain

## Audience map

| Audience | Current path | Gap |
|---|---|---|
| Brand-new user, "what is this?" | README.rst → introduction.rst (= same content) | Quick-start with output is missing |
| User running RV test on their data | tutorial.rst → notebooks | Notebook content is good but the .rst index is a stub; no narrative scaffold |
| Theoretical reader (RV / DMSS) | references.rst | No conceptual overview before the API |
| User on v0.3 → v0.4 | migrating_to_v0.4.rst | Already covered |
| User with custom demand | custom_demand.rst | Already covered |
| **In-package demand user (NEW)** | nothing | **Major feature, no docs** |
| User hitting an error | error messages | No FAQ / troubleshooting |
| Contributor | nothing | No dev-setup / how-to-extend guide |
| AI agent | CLAUDE.md / AGENTS.md / agent_guide.rst | Three docs with overlapping content; roles unclear |

## Proposed phases

### Phase 1 — High-leverage gaps (the critical ones)

Tuned for the first-time-user audience (Q3) with a tight conceptual touch (Q4).

1. **Quick-start in README** + an `introduction.rst` that's not just a wrapper.
   - 8-line minimal working example with expected output.
   - Use a synthetic dataset so the snippet is copy-pastable.
   - Why first: every other doc cross-references "see the quick-start" — without it, intros feel abstract.

2. **In-package demand estimation tutorial** — `docs/in_package_demand.rst`.
   - The `LogitEstimator` / `NestedLogitEstimator` / `Problem(demand_params={'estimate':...})` API just landed in v0.4; no user docs exist outside docstrings.
   - Worked example: synthetic logit DGP → `LogitEstimator` → `Problem.solve()` → results.
   - Show both A (standalone) and B (inline shortcut) paths.

3. ~~README.md as MD sibling.~~ **Dropped per Chris's Q1 answer.** Replaced by:

   **3a. Reader's guide in README.rst** ✓ Done 2026-05-05. Audience-keyed pointer block (end users → Tutorial; migrating → migration guide; custom demand → custom_demand.rst; AI agents → AGENTS.md).

   **3b. AGENTS.md pointer at top of CLAUDE.md** ✓ Done 2026-05-05.

   **3c. Methodology citations in README** ✓ Done 2026-05-05. Three-paper block (DMSS 2023, DMQSW 2026, DMQSS 2026) plus matching BibTeX. Mirrored in `docs/references.rst`.

**Effort estimate (remaining):** ~0.5-1 day for items 1 and 2.

### Phase 2 — User journey

4. **Conceptual overview** — `docs/concepts.rst` (~5-7 pages of prose, math via mathjax).
   - RV test statistic and what it means.
   - F-stat / weak-instrument diagnostic.
   - MCS p-values.
   - The DMSS (2024 NBER w32863) instrument-relevance framework — what falsification means, when it works, when it doesn't.
   - Cross-link to `references.rst` for primary sources.

5. **Expand `tutorial.rst`** with narrative around the notebooks.
   - "Read this if you have demand estimates and want to test conduct" — point to `testing_firm_conduct.ipynb`.
   - "Read this if you want to see all built-in models" — point to `model_library.ipynb`.
   - "Read this if you want to estimate logit demand inside pyRVtest" — point to the new in-package demand tutorial.

6. **FAQ / troubleshooting** — `docs/faq.rst`.
   - "PyBLP threw `LinAlgError` during demand estimation" — what to check.
   - "My F-stat is NaN / negative" — degenerate cases, weak instruments.
   - "MCS p-values look identical for many models" — pruning logic.
   - "I'm using `endogenous_cost_component` — why does the demand_adjustment behave differently?" — the v0.5 gate (per `.claude/plans/v0.5-followups.md`).
   - Labor-side experimental status.

**Effort estimate:** 2-3 days.

### Phase 3 — Reference / contributor

7. **Hand-written API narrative** — augment `api.rst`.
   - Group by user task: "constructing a Problem", "solving and reading results", "passthrough diagnostics", "instrument helpers", "demand estimators", "exception types".
   - Currently `api.rst` is just autodoc directives; add brief intro paragraphs per section.

8. **`CONTRIBUTING.md`** at repo root.
   - Dev environment setup (Python 3.9 framework on Mac, requirements-dev).
   - How to run tests, run flake8, build docs locally.
   - How to add a new `ConductModel` subclass (link to `models/standard.py` as the reference).
   - How to add a new demand backend (link to `custom_demand.rst`).
   - Branch / PR conventions.

9. **Math appendix** — `docs/math.rst` (or section within `concepts.rst`).
   - DMSS eq (3) and the marginal-cost formula under rule-of-thumb pricing (the cost_scaling derivation we worked through today).
   - Villas-Boas (2007) passthrough matrix derivation summary.
   - F-stat formula and rank adjustment under `endogenous_cost_component`.

**Effort estimate:** 2 days.

### Phase 4 — Agent docs consolidation

10. **Clarify the three agent docs**:
    - `CLAUDE.md` (current: 94 lines) — keep as the Claude Code session-start file. Architecture sketch + module layout + commands. Used to load every Claude Code session.
    - `AGENTS.md` (current: 392 lines) — keep as the comprehensive coding-assistant contract: code state, layout, conventions, deprecations, supported operations. Read at start of any agent session that's going to touch code.
    - `docs/agent_guide.rst` (current: 825 lines) — keep as the RTD-rendered version of agent guidance. Surfaced to users via `pyRVtest.show_agent_guide()`.
    - **Audit for content drift** — the three docs probably have overlapping text; pick a "source of truth" doc per topic and have the others link to it. My guess: AGENTS.md is the deepest, agent_guide.rst should be a structured rendering for RTD, CLAUDE.md is the entry point.

11. **Top-level pointer table** — small section in README pointing readers to the right entry doc:
    - Users → tutorial / notebooks
    - Migrating → migrating_to_v0.4
    - Custom demand → custom_demand
    - Contributing → CONTRIBUTING.md
    - AI agents → AGENTS.md (then agent_guide.rst for depth)

**Effort estimate:** 1 day.

### Phase 5 — Polish

12. Cross-reference `:class:` / `:meth:` directives consistently across all RST.
13. Verify RTD builds cleanly with the new files (`tox -e docs`).
14. Update `index.rst` toctree to reflect new files (concepts, in_package_demand, faq).
15. Drop the one-character "stub" `legal.rst` — fold into `index.rst` or move to a footer link.
16. Check `setup.py` `long_description` still extracts cleanly from README.rst.

**Effort estimate:** 0.5 day.

## Cross-cutting questions for Chris

| # | Question | Status |
|---|---|---|
| 1 | README.md vs README.rst | **Answered (no MD; pointers added).** ✓ |
| 2 | Add DMQSW + DMQSS citations | **Answered (yes, both).** ✓ Open: URL/title for the DMQSS non-linear cost paper. |
| 3 | First-time users vs collaborators | **Answered (first-time users).** ✓ |
| 4 | Conceptual depth | **Answered (tight ~3 pages).** ✓ |
| 5 | MEMO_*.md at repo root — move to `.claude/` or keep public? | Pending. |
| 6 | Notebooks — refresh against v0.4 API or treat replication `.py` scripts as authoritative? | Pending. (Recommend verifying current notebook state first; ~10 min.) |

## Recommended order

If you want to stop after one phase: **do Phase 1 only**. Quick-start + in-package demand tutorial + README.md plug the three biggest user-facing gaps and are independent of the other phases.

If you want a release-shaped overhaul: **Phases 1 + 2 + 4** before the next push to PyPI. Phase 3 (contributor + math) is best done once the package is stable, and Phase 5 is cleanup that lands incrementally as you touch each file.
