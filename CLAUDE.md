# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when
working with code in this repository.

> **Coding-agent guidance lives in [`AGENTS.md`](AGENTS.md).** It is
> the deeper, longer contract for any agent that will modify code
> (layout, conventions, deprecations, what NOT to change casually).
> Read it before non-trivial edits. The orientation below is a quick
> summary; `AGENTS.md` is the source of truth.

## What this project does

`pyRVtest` is a Python package for testing firm conduct models in
industrial organization. It implements the Rivers-Vuong (RV) test and
related procedures (F-statistics for instrument-strength diagnostics,
Hansen-Lunde-Nason Model Confidence Set p-values) to statistically
discriminate between candidate models of firm behavior (Bertrand,
Cournot, monopoly, perfect competition, vertical, partial collusion,
rule-of-thumb pricing, ...). Demand can be estimated externally with
[PyBLP](https://github.com/jeffgortmaker/pyblp) or, for plain-logit /
one-level nested-logit cases, in-package via
`pyRVtest.LogitEstimator` / `pyRVtest.NestedLogitEstimator`.

Beta status: the API may change between releases.

## Commands

### Tests
```bash
pytest                              # full suite (~3 min cold)
pytest --ignore=tests/replication   # skip data-dependent suites
pytest tests/test_mypy_strict.py    # mypy --strict gate
```

The `tests/` directory contains 30+ test files including unit tests,
integration tests, snapshot tests, regression guards, and replication
suites. The mypy gate is enforced via the test suite.

### Linting
```bash
flake8
# or via tox
tox -e flake8
```
Lint rules: max line length 120, ignores E731, E741, F541, W504.

### Build docs
```bash
tox -e docs
# or directly
cd docs && sphinx-build -E -d _build/html-doctrees -b html . _build/html
```

### Build and release
```bash
tox -e build          # check + sdist + wheel
tox -e release-test   # upload to test PyPI
tox -e release        # upload to PyPI
```

## Architecture (quick orientation)

### Main workflow

1. **Estimate demand.** Externally with `pyblp` (random-coefficients,
   BLP, micro-moments), or in-package with
   `pyRVtest.LogitEstimator` / `pyRVtest.NestedLogitEstimator` for
   linear-2SLS cases.
2. **Construct a Problem.** Pass the demand specification, product
   data, cost formulation, instrument formulation(s), and the list of
   candidate `ConductModel` instances to `pyRVtest.Problem`.
3. **Solve.** `Problem.solve()` runs the pipeline: orthogonalize on
   cost shifters, optionally apply the demand-adjustment first-stage
   correction, compute GMM moments, RV test statistics, F-stat
   diagnostics, and MCS p-values per instrument set.
4. **Inspect.** `ProblemResults` holds all outputs and formats them
   for display.

### Key classes (top-level imports)

- `Formulation` — R-style formula for cost shifters, instruments, and
  demand-side regressors.
- `ConductModel` hierarchy — `Bertrand`, `Cournot`, `Monopoly`,
  `PerfectCompetition`, `MixCournotBertrand`, `PartialCollusion`,
  `Vertical`, `RuleOfThumb`, `ConstantMarkup`, `UserSuppliedMarkups`,
  `CustomConductModel`. Plus labor-side experimental classes
  (`Monopsony`, `BertrandWages`, `CournotEmployment`,
  `NashBargaining`).
- `LogitEstimator`, `NestedLogitEstimator` — in-package 2SLS
  estimators producing a `demand_params` dict.
- `Problem` — the central orchestrator.
- `ProblemResults` — output (TRV, F, MCS_pvalues, markups,
  marginal_cost, passthrough diagnostics).
- `ModelFormulation` — legacy v0.3 string-based conduct specifier;
  emits `DeprecationWarning`, removal targeted for v0.6.

### Module layout (top level)

```
pyRVtest/
├── __init__.py        # Public API re-exports
├── problem.py         # Problem orchestrator, Models recarray
├── formulation.py     # Formulation, ModelFormulation (legacy)
├── markups.py         # build_markups, passthrough, FOC helpers
├── output.py          # format_table, output helpers
├── options.py         # global runtime options
├── exceptions.py      # PyRVTestError hierarchy
├── _agent_guide.py    # show_agent_guide()
├── data/              # CSV critical value tables, synthetic example
├── backends/          # DemandBackend protocol + impls (pyblp, logit, nested, user, labor)
├── estimators/        # LogitEstimator, NestedLogitEstimator
├── models/            # ConductModel hierarchy
├── solve/             # Solve-stage helpers (orthogonalize, demand_adjustment, etc.)
├── instruments/       # Vectorized instrument constructors
└── results/           # ProblemResults, Progress dataclass
```

The full annotated layout, including subpackage internals and the
data flow through `Problem.solve()`, is in `AGENTS.md`.

### Dependency on PyBLP internals

The package imports directly from `pyblp.utilities.basics`,
`pyblp.utilities.algebra`, and `pyblp.configurations.formulation`.
Changes in PyBLP's internal API can break this package. CI exercises
two pin combinations: `numpy<2 + pyblp<1.2` and
`numpy>=2 + pyblp>=1.2`.
