# AGENTS.md

Guide for AI coding assistants (Claude, GPT, others) and new human contributors
working on `pyRVtest`. This file is a living contract: the layout and
conventions it describes match the state of the code on the `v0.4-refactor`
branch and are enforced by the test suite (see §"What NOT to change casually").

If you are a human first-time reader, you can also run
`import pyRVtest; pyRVtest.show_agent_guide()` from a Python session to get
this content on stdout, or open `docs/agent_guide.rst` for a longer narrative
walkthrough.

## What pyRVtest is

`pyRVtest` is a Python package for testing models of firm conduct in industrial
organization. Given a demand estimate and a set of candidate conduct hypotheses
(Bertrand, Cournot, monopoly, perfect competition, vertical models, partial
collusion, user-supplied markup functions, etc.), it runs the Rivers-Vuong (RV)
test and related procedures (F-statistics for weak-instrument diagnostics, the
Model Confidence Set p-values of Hansen-Lunde-Nason) to statistically
discriminate between the models.

The package is consumed by industrial-organization researchers. The most common
upstream dependency is `pyblp` (for BLP / nested-logit demand), but the v0.4
refactor makes the demand interface generic so logit-family, user-supplied
demand systems (via `UserSuppliedBackend`), and eventually labor-supply systems
plug into the same RV pipeline.

This is a beta-status package; the API may change. v0.4 is a large
documentation-and-architecture refactor rather than a feature release.

## Architecture at a glance

```
pyRVtest/
├── __init__.py             # Re-exports the stable public API (see __all__ below)
├── _agent_guide.py         # show_agent_guide() helper (step 23)
├── problem.py              # Problem, Models recarray, Products routing
├── products.py             # Products class (extracted from problem.py in step 2)
├── formulation.py          # Formulation (pyblp re-export), ModelFormulation (deprecated)
├── markups.py              # build_markups (public), _compute_markups (internal),
│                           #   construct_passthrough_matrix, build_ownership,
│                           #   evaluate_first_order_conditions (legacy string dispatch)
├── output.py               # format_table and output helpers
├── options.py              # global dtype / verbose flags
├── data/                   # CSV critical value tables for F-stat size/power
├── backends/
│   ├── base.py             # DemandBackend + SupportsDemandAdjustment protocols
│   ├── pyblp.py            # PyBLPBackend (wraps pyblp.ProblemResults)
│   ├── logit.py            # LogitBackend + analytical jacobian/hessian helpers
│   ├── nested_logit.py     # NestedLogitBackend (analytical nested-logit math)
│   ├── user.py             # UserSuppliedBackend (bring-your-own-Jacobian)
│   └── labor/              # Labor-side backends (LaborSupplyBackend skeleton, step 14b)
├── models/
│   ├── base.py             # ConductModel abstract base
│   ├── standard.py         # Bertrand, Cournot, Monopoly, PerfectCompetition
│   ├── mixed.py            # MixCournotBertrand
│   ├── collusion.py        # PartialCollusion
│   ├── custom.py           # CustomConductModel
│   ├── vertical.py         # Vertical composer (bilateral oligopoly)
│   ├── constant.py         # RuleOfThumb, Keystone, ConstantMarkup (Dearing 2026, step 12)
│   ├── labor.py            # Monopsony, BertrandWages, CournotEmployment, NashBargaining (step 14a)
│   └── _adapter.py         # Legacy ModelFormulation → ConductModel translation (step 5c)
├── solve/
│   ├── demand_adjustment.py # Unified DMSS 2024 eq. 77 first-stage correction (step 4d)
│   ├── passthrough.py       # build_passthrough public helper (step 11)
│   ├── markups.py           # Per-model markup stage (scaffolded, populated step 8)
│   ├── orthogonalize.py     # Orthogonalization stage
│   ├── endogenous_cost.py   # Endogenous-cost gamma correction
│   └── test_engine.py       # Final RV/F/MCS computation
├── instruments/
│   ├── product.py           # BLP / differentiation / rival-sum instrument helpers
│   └── labor.py             # Bartik / Hausman / HHI instrument helpers
└── results/
    └── __init__.py          # ProblemResults + Progress dataclass
```

The canonical data flow:

1. User constructs a backend (typically `PyBLPBackend.from_pyblp_results(...)`,
   `LogitBackend(...)`, or `UserSuppliedBackend(...)`) or passes a pre-existing
   `pyblp.ProblemResults` into `Problem`.
2. User specifies conduct candidates as a list of `ConductModel` instances
   (`Bertrand(...)`, `Cournot(...)`, `Vertical(...)`, etc.) via the `models=`
   kwarg on `Problem`. The legacy `model_formulations=` accepting
   `ModelFormulation` still works but emits a `DeprecationWarning`.
3. `Problem.__init__` validates inputs, builds the `Models` recarray (the
   canonical intermediate), and calls `build_markups` to compute per-model
   implied markups.
4. `Problem.solve()` runs the pipeline: orthogonalize on cost shifters, apply
   demand adjustment (the DMSS 2024 eq. 77 first-stage correction, routed
   through `solve/demand_adjustment.py`), compute GMM fit Q values, RV test
   statistics TRV, scaled F-statistics, and MCS p-values per instrument set.
5. `ProblemResults` holds all outputs and formats them for display.

## Deprecation policy

Four backward-compatibility surfaces exist on the v0.4 branch:

1. **`ModelFormulation`** — the v0.3 string-based conduct specifier
   (`model_downstream='bertrand'`, etc.). Deprecated in v0.4 and v0.5;
   scheduled for removal in v0.6. Use `Bertrand(...)`, `Cournot(...)`,
   `Monopoly(...)`, etc. from `pyRVtest.models` instead. See
   `docs/migrating_to_v0.4.rst` for the per-pattern migration table.

2. **`model_formulations=` kwarg on `Problem`** — passes a sequence of
   `ModelFormulation` objects. Coexists with the new `models=` kwarg;
   passing both raises `TypeError`. Same deprecation timeline
   (removed in v0.6).

3. **`demand_params['sigma']`** — alias for `demand_params['rho']` on the
   nested-logit correlation parameter. Step 6b in the migration. Passing
   both `'sigma'` and `'rho'` in the same dict raises `TypeError`. Note
   the `NestedLogitBackend` class constructor keeps the `sigma=[...]`
   kwarg name because its math follows the AFSSZ L-level convention.

4. **Per-model `unit_tax` / `advalorem_tax` / `advalorem_payer`** on
   `ConductModel` / `Vertical` / `ModelFormulation`. Deprecated in v0.4
   (OQ 14); removed in v0.6. Move the tax column to
   `Problem(..., unit_tax='col', advalorem_tax='col',
   advalorem_payer='firm'|'consumer')`. Use `unit_tax_salient=False` /
   `advalorem_tax_salient=False` on individual models for salience-test
   opt-outs. When both Problem-level and model-level taxes are set, the
   model-level value wins (legacy precedence) and emits a
   `DeprecationWarning` naming the conflict.

**Deprecation-warning hygiene.** Each deprecation site fires once per Python
session, identified by a `(class, field_name)`-style tuple on a class-level
or module-level flag. We explicitly do NOT call
`warnings.simplefilter('once', DeprecationWarning)` at the package level
because that would mutate the user's global filter state.

**Window:** v0.4 and v0.5 both emit deprecation warnings. v0.6 removes the
legacy surfaces. If you are extending the deprecation window, do so only
after discussion with Chris (the maintainer) and update this file, the
migration guide, and the release notes in lockstep.

## Running the test suite

The test suite is the authoritative spec for everything in the package.
There are 357 passing + 3 skipped tests on the `v0.4-refactor` branch.

```bash
# Full suite (~3 min cold)
python -m pytest tests/ -q

# One test file
python -m pytest tests/test_models.py -q

# One test
python -m pytest tests/test_models.py::test_bertrand_ownership_validation -q

# Regenerate snapshots after a *deliberate* numerical change
REGENERATE_SNAPSHOTS=1 python -m pytest tests/test_snapshots.py

# Linting
flake8                        # max-line-length 120, ignores E731 E741 F541 W504

# Docs build (validates RST syntax, cross-refs)
tox -e docs
```

Tests are in `tests/` at the repo root. Each file corresponds to either a
feature (`test_demand_adjustment.py`, `test_passthrough.py`,
`test_first_stage_correction.py`) or a migration invariant
(`test_import_roundtrip.py`, `test_public_api_pin.py`,
`test_models_step_0_parity.py`).

## What NOT to change casually

These are load-bearing invariants. Changing any of them without explicit
coordination with Chris produces silent correctness regressions or breaks
downstream user code.

- **Step 0 snapshots.** The JSON files under `tests/snapshots/` encode the
  exact TRV / F / MCS / markups / g / Q numbers produced by each fixture
  on the v0.3.3-stable baseline. `tests/test_snapshots.py` asserts every
  current run reproduces them. The decision rule is: changes at
  `atol <= 1e-12` are floating-point noise and can be regenerated freely;
  changes above `atol <= 1e-7` require a numerical justification,
  coauthor approval, and a handover note. Anything in between is a grey
  zone — default to investigating first.

- **First-stage correction equivalence tests.** Located in
  `tests/test_first_stage_correction.py`. These directly compare the
  `demand_results` code path (pyblp path) with the `demand_params` code
  path (analytical path) on matched DGP. If you touch
  `solve/demand_adjustment.py`, run these tests first.

- **Deprecation timeline.** Do not remove `ModelFormulation`, the
  `model_formulations=` kwarg, or the `sigma→rho` alias before v0.6. The
  warning emission logic in each of these surfaces is tested; if you
  refactor the emission you must not collapse it to a global filter.

- **Public `__all__` contents.** `tests/test_import_roundtrip.py` and
  `tests/test_public_api_pin.py` lock the top-level and subpackage
  `__all__` lists. If you add a public symbol you must also add it to
  the relevant `__all__` and to the corresponding expected-list in
  `test_import_roundtrip.py`.

- **PyBLP private-attribute coupling.** The only place in the package
  that touches `pyblp.ProblemResults._sigma / _pi / _beta / _rho /
  _delta` is `backends/pyblp.py`. Do not add access sites elsewhere.

## Where to start for common tasks

- **Add a new conduct model.** Create a class under `pyRVtest/models/` that
  inherits from `ConductModel` (see `models/base.py`), implements
  `_compute_markup(O, D, s)` and `_markup_derivative(O, D, dD, s, mu)`,
  and is exported from `pyRVtest/models/__init__.py` (both the `from ...
  import` line and the `__all__` list). Then re-export at the package
  level in `pyRVtest/__init__.py`. Add a unit test to `tests/test_models.py`
  following the existing patterns, plus a snapshot if the model changes
  numerical behavior on fixtures already in `tests/snapshots/`.

- **Add a new backend.** Implement the `DemandBackend` protocol from
  `backends/base.py` (required: `compute_jacobian`, `compute_hessian`,
  `perturbed` context manager, `n_parameters`, `theta_names`). Optionally
  also implement `SupportsDemandAdjustment` (required: `demand_moments`,
  `xi_gradient`, `jacobian_gradient`) if your backend can participate in
  the DMSS eq. 77 first-stage correction. The testing engine uses
  `isinstance(..., SupportsDemandAdjustment)` to decide whether to call
  those methods. `UserSuppliedBackend` is a complete example of the
  opt-out case.

- **Add a new instrument helper.** Place it under
  `pyRVtest/instruments/product.py` (BLP-style / product-side) or
  `pyRVtest/instruments/labor.py` (labor-side). Keep the function
  signature consistent: take a `pandas.DataFrame` of product data plus
  helper-specific kwargs, return an `NDArray[float64]` suitable for
  stacking into a `Formulation`-materialized Z matrix.

- **Add a new test.** Follow the naming convention `tests/test_<target>.py`.
  Keep fixtures fast — prefer `user_supplied_markups` mode where
  applicable so tests don't pay the cost of full markup computation.
  Heavy fixtures (full market panels) belong in `tests/replication/`.

- **Fix a numerical bug.** First reproduce with a focused test under
  `tests/test_<area>.py`. Then compare against the snapshot layer: if
  snapshots move, document why (paper reference, fix is correcting
  behavior vs. changing it, etc.) and regenerate with
  `REGENERATE_SNAPSHOTS=1`. Commit the fix, the regenerated snapshots,
  and a handover note under `.claude/handovers/` all together.

## Style and formatting conventions

- Line length 120 (enforced by `flake8`).
- `from __future__ import annotations` at the top of every `.py` file that
  uses PEP 604 (`X | Y`) or forward references in type hints.
- No f-string prefix on strings without interpolation (F541 is ignored,
  but try to keep f-prefix meaningful).
- Docstrings: brief one-liner summary, a blank line, then a paragraph;
  NumPy-style Parameters/Returns blocks where relevant.
- Commit messages start with a tag: `REFACTOR:`, `TEST:`, `DOC:`, `BUG:`,
  `PERF:`, `CI:`. Include a short handover sentence if the commit touches
  cross-cutting concerns.
- No emojis in source or docs unless the user has explicitly requested them.

## Key references

Recent handovers (read the most recent first):

- `.claude/handovers/2026-04-17-session5-step5-complete.md` — step 5
  class-based API design, architecture hand-off.
- `.claude/handovers/2026-04-16-session4-step4-complete.md` — step 4
  unified demand-adjustment path.
- `.claude/handovers/2026-04-16-session3-step0-complete.md` — step 0
  snapshots + baseline protection.
- `.claude/handovers/2026-04-16-v0.4-refactor-design.md` — initial plan
  discussion, goals, scope tiers.
- `.claude/plans/v0.4-refactor.md` — full 25-step migration plan
  (authoritative source for architectural decisions).
- `docs/migrating_to_v0.4.rst` — user-facing migration guide.
- `MEMO_coauthor_updates.md` — coauthor-facing memo covering behavioral
  changes through the latest step.

## Top-level public API (v0.4)

The canonical list is `pyRVtest.__all__`; the entries below are grouped
by purpose:

- **v0.3 baseline (unchanged):** `data`, `options`, `build_ownership`,
  `build_markups`, `construct_passthrough_matrix`,
  `evaluate_first_order_conditions`, `read_pickle`, `Formulation`,
  `ModelFormulation`, `Problem`, `Models`, `Products`, `ProblemResults`,
  `__version__`.
- **v0.4 subpackages:** `backends`, `instruments`, `models`, `solve`.
- **v0.4 class-based ConductModel API:** `ConductModel`, `Bertrand`,
  `Cournot`, `Monopoly`, `PerfectCompetition`, `MixCournotBertrand`,
  `PartialCollusion`, `CustomConductModel`, `Vertical`.
- **v0.4 Dearing simple-markup models (step 12):** `RuleOfThumb`,
  `Keystone`, `ConstantMarkup` (Dearing, Magnolfi, Quint, Sullivan,
  and Waldfogel 2026, Examples 1 and 7).
- **v0.4 labor-side models (step 14a):** `Monopsony`, `BertrandWages`,
  `CournotEmployment`, `NashBargaining` (raises `NotImplementedError`
  in v0.4; full formula deferred to v0.5).
- **v0.4 diagnostic helper:** `build_passthrough`.
- **v0.4 agent guide exporter:** `show_agent_guide`.

Anything else is internal and may change without notice.

## Labor-side usage (v0.4 step 14)

`Problem(market_side='labor')` switches `pyRVtest` to labor-supply
testing: upward-sloping supply, markdowns instead of markups, and the
four labor conduct classes above. The flip is localized to
`Problem.__init__` — the rest of the pipeline is unchanged.

```python
import pandas as pd
import pyRVtest

df = pd.DataFrame({
    'market_ids':       [0, 0, 1, 1],
    'firm_ids':         [0, 1, 0, 1],
    'wages':            [12.0, 13.5, 11.0, 14.0],   # strictly positive
    'employment_share': [0.30, 0.25, 0.35, 0.20],   # in [0, 1], sums <= 1 per market
    'cost_shifter':     [0.5, 1.2, 0.7, 0.9],
    'iv0':              [1.1, 1.0, 0.9, 1.2],
    'markdown_m1':      [0.1, 0.1, 0.12, 0.12],
    'markdown_m2':      [0.0, 0.0, 0.0, 0.0],
})

problem = pyRVtest.Problem(
    cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
    instrument_formulation=pyRVtest.Formulation('0 + iv0'),
    product_data=df,
    models=[
        pyRVtest.Monopsony(user_supplied_markups='markdown_m1'),
        pyRVtest.PerfectCompetition(user_supplied_markups='markdown_m2'),
    ],
    market_side='labor',
)
```

Column-name defaults for labor-mode are `'wages'` (in place of `'prices'`)
and `'employment_share'` (in place of `'shares'`). The canonical pyRVtest
`shares` column is always a share (values in `[0, 1]`, summing to at
most 1 per market); the default name advertises units rather than naming
a raw quantity, so users with raw employment counts must normalize to a
market-level employment share before passing data in. Override the
defaults with `column_names={'price': 'my_wage_col', 'shares':
'my_emp_share_col'}`. Invalid keys raise `ValidationError` with the
typo surfaced in the message.

**Sign-convention validation.** `wages > 0` and `employment_share > 0`
are checked at `Problem.__init__` on the raw (non-aliased) columns. A
zero-wage or negative-wage row raises `ValidationError` with the
expected / received / fix format; the error names the user's original
column and points at this section. The implied labor-supply Jacobian
sign (`ds/dw > 0`) is a backend-level check and activates when
`LaborSupplyBackend.compute_jacobian` is populated in v0.5.

**Cross-side rejection.** Passing a product-side model (Bertrand,
Cournot, Monopoly, MixCournotBertrand, PartialCollusion) inside a
labor-mode `Problem` raises `ValidationError` at init, and the symmetric
check rejects labor-side models under the default product mode.
`PerfectCompetition` is genuinely side-neutral (zero markup / markdown)
and is accepted on both sides without opt-in. `CustomConductModel`
requires an explicit `side='labor'` opt-in under `market_side='labor'`
(and rejects `side='labor'` under `market_side='product'`) because the
user's `markup_fn` implicitly picks a sign convention.

**Status of `LaborSupplyBackend`.** `pyRVtest/backends/labor/nested_logit_labor.py`
ships a skeleton in v0.4: constructor, `n_parameters` / `theta_names`
honor the protocol, but `compute_jacobian` / `compute_hessian` /
`perturbed` raise `NotImplementedError` with a pointer to v0.5. Users
who need a working labor-supply backend today should wrap their own
with `UserSuppliedBackend` (see `docs/custom_demand.rst`).

**Deferred to v0.5.** Full `LaborSupplyBackend` math, `NashBargaining`
formula, joint product + labor conduct testing (basic_model.tex
framework), and full Almagro-Sood ordered-choice labor supply.
