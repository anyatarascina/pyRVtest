# AGENTS.md

Guide for AI coding assistants (Claude, GPT, others) and new human contributors
working on `pyRVtest`. This file is a living contract: the layout and
conventions it describes match the state of the code on the `v0.4-refactor`
branch and are enforced by the test suite (see §"What NOT to change casually").

If you are a human first-time reader, you can also run
`import pyRVtest; pyRVtest.show_agent_guide()` from a Python session to get
this content on stdout, or open `docs/agent_guide.rst` for a longer narrative
walkthrough.

Other docs you may want:

- `CONTRIBUTING.md` — dev environment setup, testing/linting/docs commands,
  conventions for adding new conduct models or demand backends, branch/PR
  conventions.
- `docs/faq.rst` — user-facing FAQ and troubleshooting (installation
  pitfalls, F-stat anomalies, deprecation behaviors, labor-side caveats).
- `docs/in_package_demand.rst` — walkthrough of `LogitEstimator` and
  `NestedLogitEstimator`.
- `docs/math.rst` — condensed mathematical reference (RV statistic,
  F-stat, MCS, Villas-Boas passthrough, the DMQSW eq (3) FOC).
- `.claude/plans/v0.5-followups.md` — items deferred from v0.4 to v0.5.

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
├── _agent_guide.py         # show_agent_guide() helper
├── problem.py              # Problem, Models recarray, Products routing
├── products.py             # Products class
├── formulation.py          # Formulation (pyblp re-export), ModelFormulation (deprecated)
├── markups.py              # build_markups (public), _compute_markups (internal),
│                           #   build_phi_matrix + build_markup_derivative (precompute),
│                           #   construct_passthrough_matrix, build_ownership,
│                           #   evaluate_first_order_conditions (legacy string dispatch)
├── output.py               # format_table and logging shim
├── options.py              # global dtype / verbose flags
├── exceptions.py           # PyRVTestError hierarchy
├── version.py              # __version__
├── data/                   # CSV critical value tables + synthetic example dataset
├── backends/
│   ├── base.py             # DemandBackend + SupportsDemandAdjustment protocols
│   ├── pyblp.py            # PyBLPBackend (wraps pyblp.ProblemResults)
│   ├── logit.py            # LogitBackend + analytical jacobian/hessian helpers
│   ├── nested_logit.py     # NestedLogitBackend (analytical nested-logit math)
│   ├── user.py             # UserSuppliedBackend (bring-your-own-Jacobian)
│   └── labor/              # Labor-side backends (LaborSupplyBackend skeleton)
├── estimators/
│   ├── _base.py             # Shared 2SLS base class
│   ├── logit.py             # LogitEstimator (in-package plain-logit 2SLS)
│   ├── nested_logit.py      # NestedLogitEstimator (one-level nested-logit 2SLS)
│   └── _within_share.py     # count_in_nest_iv helper for nested-logit auto-IV
├── models/
│   ├── base.py             # ConductModel abstract base
│   ├── standard.py         # Bertrand, Cournot, Monopoly, PerfectCompetition
│   ├── mixed.py            # MixCournotBertrand
│   ├── collusion.py        # PartialCollusion
│   ├── custom.py           # CustomConductModel
│   ├── vertical.py         # Vertical composer (bilateral oligopoly)
│   ├── constant.py         # RuleOfThumb, ConstantMarkup
│   ├── user_supplied.py    # UserSuppliedMarkups (pre-computed markup column wrapper)
│   ├── labor.py            # Monopsony, BertrandWages, CournotEmployment, NashBargaining
│   └── _adapter.py         # Legacy ModelFormulation → ConductModel translation
├── backends/
│   └── factory.py          # make_demand_backend: product_data + demand_results/params
│                           #   -> backend (shared by Problem + the precompute builders)
├── solve/
│   ├── demand_adjustment.py # Unified DMSS 2024 eq. 77 first-stage correction
│   │                        # (split: compute_demand_block + compute_markup_jacobian_raw
│   │                        #  + apply_demand_adjustment_transforms; orchestrated by
│   │                        #  compute_demand_adjustment, reused by build_phi_matrix /
│   │                        #  build_markup_derivative)
│   ├── passthrough.py       # build_passthrough public helper
│   ├── markups.py           # Per-model markup stage
│   ├── orthogonalize.py     # Orthogonalization stage
│   ├── endogenous_cost.py   # Endogenous-cost gamma correction
│   └── test_engine.py       # Final RV/F/MCS computation
├── instruments/
│   ├── product.py           # BLP / differentiation / rival-sum instrument helpers
│   └── labor.py             # Bartik / Hausman instrument helpers (HHI deliberately omitted; see file-level docstring)
└── results/
    ├── results.py           # ProblemResults + Progress dataclass
    ├── panel.py             # PanelResults
    └── _format.py           # Display helpers
```

The canonical data flow:

1. User specifies the demand side. Three options: pass
   `demand_results=<pyblp.ProblemResults>` for pyblp-fit demand;
   `demand_params={...}` for inline analytical logit / nested logit;
   or `demand_backend=<DemandBackend>` for a pre-built backend (e.g.
   `UserSuppliedBackend` wrapping a researcher's custom Jacobian).
   `Problem.__init__` validates that exactly one of the three is set
   (or none, for the user_supplied_markups-only path) and routes through
   `_construct_demand_backend` to a single `self._demand_backend` object.
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
   `ConductModel` / `Vertical` / `ModelFormulation`. Deprecated in v0.4; **removed in v0.7** (one release later than the other v0.4
   deprecations — the per-model tax pattern is common enough in
   existing user code to warrant an extra release of runway). Move the
   tax column to `Problem(..., unit_tax='col', advalorem_tax='col',
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

**Window:** v0.4 and v0.5 both emit deprecation warnings. v0.6 removes
`ModelFormulation`, the `model_formulations=` kwarg, the
`sigma`→`rho` alias, and `pyRVtest.output.output()`. The per-model
tax kwargs get an extra release of runway and are removed in v0.7. If
you are extending the deprecation window, do so only after discussion
with Chris (the maintainer) and update this file, the migration guide,
and the release notes in lockstep.

## Running the test suite

The test suite is the authoritative spec for everything in the package.
There are 619 passing + 3 skipped tests on the `v0.4-refactor` branch as
of v0.4.0rc1.

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
- `.claude/handovers/MEMO_coauthor_updates.md` — coauthor-facing memo
  covering behavioral changes through the latest step.

## Top-level public API (v0.4)

The canonical list is `pyRVtest.__all__`; the entries below are grouped
by purpose:

- **v0.3 baseline (unchanged):** `data`, `options`, `build_ownership`,
  `build_markups`, `construct_passthrough_matrix`,
  `evaluate_first_order_conditions`, `read_pickle`, `Formulation`,
  `ModelFormulation`, `Problem`, `Models`, `Products`, `ProblemResults`,
  `PanelResults`, `__version__`.
- **v0.4 subpackages:** `backends`, `estimators`, `instruments`,
  `models`, `solve`.
- **v0.4 class-based ConductModel API:** `ConductModel`, `Bertrand`,
  `Cournot`, `Monopoly`, `PerfectCompetition`, `MixCournotBertrand`,
  `PartialCollusion`, `CustomConductModel`, `Vertical`.
- **v0.4 in-package demand estimators:** `LogitEstimator`,
  `NestedLogitEstimator`. See `docs/in_package_demand.rst` for the
  walkthrough.
- **v0.4 user-supplied markup wrapper:** `UserSuppliedMarkups` (for
  pre-computed markup columns; alternative to `CustomConductModel` for
  the closed-form-callable case).
- **v0.4 Dearing simple-markup models:** `RuleOfThumb`,
  `ConstantMarkup` (Dearing, Magnolfi, Quint, Sullivan, and Waldfogel
  2024, Examples 1 and 7). Use `RuleOfThumb(phi=2)` for the special
  case (the earlier `Keystone()` alias was dropped in commit `e7ea1e3`).
- **v0.4 labor-side models:** `Monopsony`, `BertrandWages`,
  `CournotEmployment`, `NashBargaining` (raises `NotImplementedError`
  in v0.4; full formula deferred to v0.5).
- **v0.4 pass-through diagnostic suite (DMQSW + DMQSS):**
  `build_passthrough` standalone helper; also exposed on
  `ProblemResults` as `passthrough_matrix`. Plus
  `Problem.passthrough_summary` / `ProblemResults.passthrough_summary`
  (pre- or post-solve γ-free pair-by-pair structural-feature distances
  against the four DMQSW Remarks-keyed metrics);
  `Problem.instrument_channels` / `ProblemResults.instrument_channels`
  (post-solve per-pair channel decomposition for one IV column,
  automatically applies DMQSS Appendix B z^e residualization when
  `endogenous_cost_component` is set);
  `ProblemResults.reliability_summary` (per-cell DMSS F-stat with
  worst-rho and empirical-rho critical values).
- **v0.4 endogenous-cost (DMQSS A.4):** `endogenous_cost_component`
  on `Problem` accepts `Optional[Union[str, Sequence[str]]]` —
  single-column for the classic DMQSS scale-economies case
  (`'log_quantity'`); list-of-strings for the multi-column case
  (`['q', 'q_sq']` for quadratic cost; `['log_q', 'log_Q_minus']`
  for scale + scope). `K_inst > K_endog` enforced per testing-IV
  bundle. Combinable with `costs_type='log'` and
  `demand_adjustment=True`.
- **v0.4 precomputed demand adjustment:** two standalone builders
  mirroring `build_markups` (raw `product_data` + demand results/params,
  no `Problem` needed): `build_phi_matrix(...)` -> `PhiMatrixData` (the
  demand-only block `H`/`H_prime_wd`/`h_i`/`h`, DMSS Appendix C eq. 77)
  and `build_markup_derivative(model_formulations, ...)` ->
  `MarkupDerivativeData` (the RAW per-model `d markup / d theta`, before
  residualize / tax / log). Plug into
  `solve(demand_adjustment=True, phi_matrix=..., markup_derivative=...)`;
  the two kwargs are independently mixable (omit either -> computed
  inline). RAW boundary means one object is valid across `costs_type`
  and tax configs / instrument sets. Each is validated against the
  Problem (dimensions, `market_ids` ordering, backend identity). NOT
  supported with `endogenous_cost_component` (gamma gradient is
  instrument-set specific) — raises. Backend construction is shared via
  `backends/factory.py::make_demand_backend`. See
  `docs/advanced_features.rst` (`advanced-precompute-da`).
- **v0.4 agent guide exporter:** `show_agent_guide`.
- **v0.4 exception hierarchy:** `PyRVTestError`, `ValidationError`,
  `InstrumentDataError`, `BackendError`, `DemandBackendError`,
  `HessianUnavailableError`. Each subclasses a Python built-in
  (`ValueError` or `RuntimeError`) so existing `except ValueError:`
  callers keep working.

Anything else is internal and may change without notice.

## Labor-side usage

**Status: experimental in v0.4.** The labor API ships in v0.4 but is
explicitly marked experimental — the sign convention, column-name
defaults, and validation behavior may adjust based on coauthor review
(Lorenzo's 2026-04-18 review handed the labor-conduct sign convention
to Marco for a cross-check against the labor-market-conduct
manuscript). Scripts written against the v0.4 labor API may need small
adjustments in v0.5 when the full `LaborSupplyBackend` lands. Treat
labor-mode results as indicative until v0.5 signs off the sign
convention.

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
