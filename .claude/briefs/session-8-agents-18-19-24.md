# Session 8 agent briefs — steps 18, 19, 24

**Purpose:** drop-in prompts for the next parallel batch after session 7
lands. Steps 18 (logging) and 19 (error-message audit) both touch source
files across the package but on different concerns. Careful coordination
is required — see the "File ownership" subsection before launching.

Step 24 is a CHANGELOG polish pass, mostly doc-only; it can run as the
third parallel slot OR serialize after 18+19 if context pressure is a
concern.

**Context these agents need before starting:**
- Branch `v0.4-refactor` at whatever head session 7 leaves behind. Check
  `git log` first.
- `.claude/plans/v0.4-refactor.md` §2 goals 18, 19, 24; §5 migration
  table.
- `AGENTS.md` — architecture tour.
- `CHANGELOG.md` already exists (created in session 7 prep) with a
  scaffolded Unreleased section.
- Test target: full pytest green. As of session 7 prep, 388 passed + 3
  skipped; session 7 should bump this.
- Python env on darwin: `python3`. Ignore the shared
  `memory/feedback_python_env.md` (coauthor-specific).

---

## File ownership for steps 18 and 19

Both agents edit the same source files but touch **different strings**.
Coordinate by concern, not by file:

| File class | Step 18 (logging) | Step 19 (error messages) |
|------------|-------------------|---------------------------|
| Top-of-module `logger = logging.getLogger(__name__)` | 18 owns | 19 reads |
| Replace `output()` and `print()` calls | 18 owns | 19 reads |
| Replace `raise Exception("…")` / `raise ValueError("…")` message strings | 19 owns | 18 reads |
| Custom exception classes | 19 owns | — |
| Test files | 18 adds `test_logging.py`; 19 adds `test_error_messages.py` | both |

**Merge order:** after both worktrees finish, merge step 19 first, then
step 18, then run `pytest`. If a conflict surfaces on a specific line,
the step 18 agent's logger call and the step 19 agent's exception
message should be straightforward to combine by hand.

---

## Agent 18 — Structured logging

**Step:** 18 (from §2 Developer experience goal 19).

**Task:** Replace all `print()` and pyRVtest's in-house `output()` calls
with Python `logging` module calls, at per-module loggers.

**Concrete steps:**

1. At the top of every pyRVtest module that currently calls `print()` or
   `output()`, add:
   ```python
   import logging
   logger = logging.getLogger(__name__)
   ```
2. Replace each call site with the appropriate level:
   - `logger.info(...)` — normal progress messages (what is currently
     most `output()` calls during `Problem.solve()`).
   - `logger.debug(...)` — verbose diagnostics, per-iteration details.
   - `logger.warning(...)` — things that are unusual but recoverable
     (e.g. fallback to finite-diff when analytical Hessian unavailable).
   - `logger.error(...)` — failure paths *before* an exception is raised
     (if adding context is useful; usually the exception message is
     enough).
3. Keep `pyRVtest/output.py` around as a legacy shim for one release:
   `output(msg)` now calls `logger.info(msg)` via a module logger at
   `pyRVtest.output`. Raise `DeprecationWarning` on import.
4. Document the logger layout in a new section of `AGENTS.md` or
   `docs/agent_guide.rst`: user can silence a subsystem with
   `logging.getLogger("pyRVtest.problem").setLevel(logging.WARNING)`.

**Hard rules:**
- Do **not** change the *content* of any message string. Step 19 owns
  the string content. If a string looks poorly formatted, add a TODO
  comment for step 19 and leave it.
- Do **not** touch `raise` statements. Even if the exception has a bad
  message, that's step 19's territory.
- `mypy --strict` clean.
- Tests in `tests/test_logging.py`: assert that specific loggers emit
  expected records using `caplog` fixture; assert that setting a logger
  level silences output as expected.
- Do not break existing `test_*` that rely on captured stdout.
  `pytest` captures `logger` output by default via `caplog`; if any
  existing test asserts on stdout, convert to a caplog assertion or
  leave that specific call site as `print()` and add a TODO comment
  referencing the test.

**Deliverables:**
- Logger initializers across pyRVtest source files.
- `pyRVtest/output.py` shim behavior updated.
- `tests/test_logging.py` (new).
- Doc section in `AGENTS.md` or `docs/agent_guide.rst`.
- Commit message: `REFACTOR: v0.4 step 18 structured logging at per-module loggers`.

**Files likely to touch:** most `.py` under `pyRVtest/`,
`pyRVtest/output.py`, `tests/test_logging.py` (new), one of `AGENTS.md`
or `docs/agent_guide.rst`.

---

## Agent 19 — Error-message audit

**Step:** 19 (from §2 Developer experience goal 18).

**Task:** Audit every `raise` statement in pyRVtest source and rewrite
the message to follow an "expected / received / fix" structure. Add
custom exception classes where a user-facing category aids debugging.

**Target format for validation errors:**

```
<What the check was looking for>. Received <what was actually there>.
Fix: <concrete, actionable step>.
```

Example rewrite:

```python
# Before
raise ValueError(f"Market {m} has only one product.")

# After
raise InsufficientMarketDataError(
    f"Expected market_id={m!r} to contain 2+ products for markup computation. "
    f"Received 1 product. Fix: drop market_id={m!r} from product_data, "
    f"or verify that market_ids column is correctly labeled."
)
```

**Concrete steps:**

1. Grep for every `raise ` in `pyRVtest/` (excluding tests). For each:
   - Classify: validation error, internal consistency error, or
     genuine runtime failure.
   - Rewrite the message in the expected/received/fix format where it
     improves debuggability. For pure internal-consistency errors
     ("this should never happen"), keep terse but prefix with
     `"pyRVtest internal error: "`.
2. Add a small custom exception hierarchy in
   `pyRVtest/exceptions.py` (new):
   ```
   PyRVTestError(Exception)
       ValidationError
           InsufficientMarketDataError
           InstrumentDataError
           ModelSpecificationError
       BackendError
           DemandBackendError
           HessianUnavailableError
   ```
   Keep the hierarchy minimal — only add subclasses that two or more
   call sites use.
3. Update `pyRVtest/__init__.py` to re-export the exception classes.
4. Do **not** change any logger/`print`/`output` call strings. Step 18
   owns those.

**Hard rules:**
- Preserve every currently-tested exception type. If a test asserts
  `raises(ValueError)`, the new custom exception must subclass
  `ValueError`. Run the test suite to find these.
- `mypy --strict` clean.
- Tests in `tests/test_error_messages.py`: at least one test per new
  custom exception class, asserting the format and the raised type.
- Update `docs/api.rst` with a new "Exceptions" section.
- Public API pin (`tests/test_public_api_pin.py`) must still pass.

**Deliverables:**
- `pyRVtest/exceptions.py` (new).
- Rewrites across pyRVtest source (raise statements only).
- `tests/test_error_messages.py` (new).
- `docs/api.rst` update.
- Commit message: `REFACTOR: v0.4 step 19 error-message audit with expected/received/fix format`.

**Files likely to touch:** most `.py` under `pyRVtest/` (but only
`raise` lines), `pyRVtest/exceptions.py` (new),
`pyRVtest/__init__.py`, `tests/test_error_messages.py` (new),
`docs/api.rst`.

---

## Agent 24 — CHANGELOG polish

**Step:** 24 (from §2 roadmap; also covered in §5 step 24).

**Task:** Take the scaffolded `CHANGELOG.md` from session 7 and polish
it to release-ready quality. The scaffold covers steps 1-7, 11, 13,
15, 17, 20, 22, 23, 24.5. After sessions 7 and 8 land, add entries for
steps 9, 10, 21, 18, 19.

**Concrete steps:**

1. Read `CHANGELOG.md`. Identify any steps committed after `2cdf2d0`
   (session 6 endpoint) that are not yet captured.
2. For each new step, add a bullet in the appropriate Added / Changed /
   Fixed / Deprecated section. Match the existing voice and level of
   detail.
3. Check that every behavior-changing commit between `main` and HEAD
   is either in the CHANGELOG or a deliberate internal-only refactor
   that doesn't warrant a user-facing entry.
4. Add a "Migration from v0.3.x" section at the top of the Unreleased
   block, summarizing the three user-visible break points:
   `ModelFormulation` deprecation, `sigma → rho` in `demand_params`,
   new `DemandBackend` protocol for custom-demand users.
5. Once ALL steps 8, 9, 10, 14, 16, 18, 19, 21 are landed, rename
   "Unreleased — v0.4.0 (work in progress)" to "[0.4.0] — YYYY-MM-DD"
   and bump `pyRVtest/version.py` to `0.4.0`. Do not do this version
   bump if any of those steps is still outstanding.

**Hard rules:**
- Do not touch source files other than `pyRVtest/version.py` (and only
  if the full step list is complete).
- Keep entries focused on user-visible behavior. Internal refactors
  (step 2 Products extraction, step 17 mypy audit) get one-line
  mentions at most.
- Commit message: `DOC: v0.4 step 24 CHANGELOG polish for v0.4.0 release`.

**Files likely to touch:** `CHANGELOG.md`, maybe
`pyRVtest/version.py`.

**Scope note:** this agent is doc-only, so it's a good candidate to
serialize *after* agents 18 and 19 land rather than running in
parallel. That way the CHANGELOG captures their commits in the same
pass. If you do parallelize, have agent 24 target only the
already-landed steps and leave 18/19 entries for a quick follow-up.

---

## Coordination notes for the launch message

1. **Sequence suggestion:** agents 18 and 19 in parallel, then agent
   24 serial after them. This avoids the scenario where agent 24 has
   to guess at 18 and 19's final commit messages.
2. Tell each agent: "Run the full test suite once at the end — do not
   iterate." The post-integration sweep is the handoff's job, not the
   agent's.
3. Tell each agent: "Commit in the worktree and include the commit
   hash in your final message."
4. Post-integration sweep: after 18 + 19 merge serially, run
   `pytest` + `mypy --strict`. Flag any regression on test count or
   type-check count. Then launch agent 24.

Estimated wall-clock: 45-60 min for 18+19 parallel, 15 min for 24
serial after.
