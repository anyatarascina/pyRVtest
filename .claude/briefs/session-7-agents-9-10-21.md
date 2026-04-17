# Session 7 agent briefs — steps 9, 10, 21

**Purpose:** drop-in prompts for three parallel worktree agents.
Recommended launch order: single message with three `Agent` tool calls
in parallel, all with `isolation: "worktree"` and `subagent_type:
"general-purpose"`. See session-6 handover for the pattern.

**Context these agents need before starting:**
- Branch `v0.4-refactor` at `2cdf2d0` (plus any commits added before
  session 7 opens — check `git log` first).
- `.claude/plans/v0.4-refactor.md` is the design doc. §5 has the
  migration table.
- `AGENTS.md` at repo root is the v0.4 architecture tour.
- Test suite: `pytest` from repo root. Target: all tests pass after
  the agent's commit. Currently 388 passed + 3 skipped.
- On darwin, run Python via `python3` (framework Python 3.9).
  The shared project memory `memory/feedback_python_env.md` points at
  Marco's Linux path — ignore it locally.

---

## Agent 9 — Extract `ProblemResults` and add DataFrame/LaTeX exports

**Step:** 9 (from §2 Architecture goal 6).

**Task:** Extract the `ProblemResults` class from
`pyRVtest/results.py` into a new `pyRVtest/results/results.py` module
(or keep the existing `results.py` if it is already module-shaped — check
first). Add four export methods on `ProblemResults`:

1. `to_dataframe()` — long-form pandas DataFrame with columns at minimum:
   `instrument_set`, `model_i`, `model_j`, `TRV`, `F`, `MCS_pvalue`,
   `instrument_set_label` (if available).
2. `summary_df()` — a compact wide-form pandas DataFrame suitable for
   presentation: one row per (instrument_set, model pair) with
   TRV / F / MCS and pass/reject indicators at the 5% level.
3. `to_latex(path=None, caption=None, label=None)` — emits a LaTeX
   tabular environment built from `summary_df()`. Use `pandas.DataFrame.to_latex`
   with `escape=False` so math symbols pass through. If `path` is given,
   write to disk; otherwise return the string.
4. `to_markdown()` — GitHub-flavored markdown rendering of `summary_df()`.

**Hard rules:**
- Do **not** touch `Problem.solve()`. This is step 8's territory and is
  explicitly out of scope here.
- All new code must be `mypy --strict` clean. Match the coverage pattern
  already in `mypy.ini` for `results`.
- Add tests in `tests/test_results_exports.py`: at least one test per new
  method, exercised against an existing fixture from the snapshot suite
  (see `tests/test_snapshot_regression.py` for the fixture pattern).
- Keep existing `ProblemResults` attributes (`TRV`, `F`, `MCS_pvalues`,
  `markups`, `marginal_cost`, `taus`) unchanged. New methods are additive.
- Preserve any deprecation aliases already present.

**Deliverables:**
- New file(s) as above.
- Updated `pyRVtest/__init__.py` exports if any public symbol is
  relocated. The public API pin in `tests/test_public_api_pin.py` must
  still pass.
- Commit message: `FEAT: v0.4 step 9 ProblemResults exports (to_dataframe, summary_df, to_latex, to_markdown)`.

**Files likely to touch:** `pyRVtest/results.py` (possibly
`pyRVtest/results/__init__.py` and `pyRVtest/results/results.py`),
`tests/test_results_exports.py` (new), `pyRVtest/__init__.py` (maybe),
`mypy.ini` (probably not).

---

## Agent 10 — `PanelResults` for multi-problem aggregation

**Step:** 10 (from §2 Architecture goal 6).

**Task:** Add a `PanelResults` class that aggregates multiple
`ProblemResults` — one per market-year, one per subsample, etc. — into
a single panel-level view. The motivating use case is AFSSZ scalable
testing with 910 market-years (step 16).

**Minimum API:**

```python
from pyRVtest import PanelResults

panel = PanelResults(
    results={key: problem_results for key, problem_results in ...},
    # key can be anything hashable — e.g. (market_id, year) or a string
)

panel.to_dataframe()       # long-form, panel_key column added
panel.summary_df()         # aggregation: rejection rates per model pair
                           # across panel keys, at 5% level by default
panel.rejection_rates(alpha=0.05)  # explicit helper returning a DataFrame
                                    # of rejection shares per model pair
panel.to_latex(...)        # reuses ProblemResults.to_latex
panel.to_markdown(...)     # reuses ProblemResults.to_markdown
panel.keys()               # iterator over panel keys
panel[key]                 # returns the underlying ProblemResults
```

**Hard rules:**
- Requires agent 9's `to_dataframe` / `summary_df`. Both agents touch
  `pyRVtest/results.py` (or its successor). **Coordination:** agent 10
  should read the latest `git log` before opening `results.py`; if
  agent 9's commit is already there, merge from it; if not, design
  `PanelResults` so it only calls `to_dataframe` / `summary_df` on
  `ProblemResults` and does not depend on their internals. Prefer the
  second approach — it lets the two agents land in either order.
- `mypy --strict` clean.
- Tests in `tests/test_panel_results.py` covering: happy path with
  2-3 keys, rejection-rate computation against a hand-computed
  expected value, empty-panel error, mismatched-model-sets error.
- Do not break `test_public_api_pin.py` — add `PanelResults` to the
  top-level `pyRVtest.__all__`.

**Deliverables:**
- `pyRVtest/results/panel.py` (new) or sibling file to agent 9's work.
- `tests/test_panel_results.py` (new).
- Updated `pyRVtest/__init__.py` export for `PanelResults`.
- Commit message: `FEAT: v0.4 step 10 PanelResults for multi-problem aggregation`.

**Files likely to touch:** `pyRVtest/results/panel.py` (new),
`pyRVtest/__init__.py`, `tests/test_panel_results.py` (new), maybe
`mypy.ini`.

---

## Agent 21 — Doctest coverage on public API

**Step:** 21 (from §2 Developer experience goal 17).

**Task:** Add a runnable doctest example to every public function,
class, and method in the pyRVtest public API. `pytest --doctest-modules`
must pass.

**Scope:**
- Every name in `pyRVtest.__all__` at the top level.
- Every name in `pyRVtest.instruments.product.__all__` and
  `pyRVtest.instruments.labor.__all__`.
- Every name in `pyRVtest.backends` that is publicly re-exported.
- Class methods and attributes listed in class docstrings.

**Hard rules:**
- Use short, self-contained examples. If the example needs a
  `Problem`, construct a minimal 2-model / 2-market one inline. Do not
  import from the test suite.
- Examples must be **fast**. Target <0.1 s per doctest block. Prefer
  synthetic data over the notebook fixtures.
- If an example would need PyBLP, mark it with a `# doctest: +SKIP`
  directive and add a comment explaining why (e.g. "requires fitted
  PyBLP results — see `docs/tutorial.rst`"). Skips are allowed but
  must be justified in-line.
- Do **not** modify function behavior. Docstring changes only.
- Update `pyproject.toml` / `setup.cfg` / `tox.ini` as needed so
  `pytest --doctest-modules` runs by default in CI. The CI workflow
  in `.github/workflows/ci.yml` has a stubbed doctest step to
  uncomment.

**Deliverables:**
- Docstring additions across the public API.
- CI workflow update to enable doctests.
- Commit message: `DOC: v0.4 step 21 doctest coverage on public API`.

**Files likely to touch:** most `.py` files under `pyRVtest/`,
`.github/workflows/ci.yml`, possibly `tox.ini` / `pyproject.toml`.

**Expected conflict surface:** high risk of collision with agents 9
and 10 on `results.py` and `results/panel.py`. **Recommendation:**
agent 21 should skip docstrings on `ProblemResults.to_dataframe`,
`.summary_df`, `.to_latex`, `.to_markdown`, and `PanelResults.*` in
this pass — agents 9 and 10 should ship docstrings with their new
methods directly.

---

## Coordination notes for the launch message

When you kick these off in one message:

1. In each agent's prompt, paste the agent's section verbatim, plus
   the top-of-file "Context these agents need" block.
2. Tell each agent: "Do not run the full test suite more than once;
   target the tests you add plus `tests/test_public_api_pin.py`."
3. Tell each agent: "At the end, commit in the worktree and include
   the commit hash in your final message."
4. After all three complete, do the post-integration sweep flagged in
   the session-6 handover retrospective: merge each worktree branch
   serially, run the full test suite once after all three land.

Estimated wall-clock: 45–75 minutes (session 6 ran similar scope in
~50 min).
