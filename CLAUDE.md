# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

`pyRVtest` is a Python package for testing firm conduct models in industrial organization. It implements the Rivers-Vuong (RV) test and related procedures (F-statistics, Model Confidence Sets) to statistically discriminate between models of firm behavior (Bertrand, Cournot, monopoly, perfect competition, etc.) using demand estimates from [PyBLP](https://github.com/jeffgortmaker/pyblp).

The package is in beta; the API may change.

## Commands

### Linting
```bash
flake8
# or via tox:
tox -e flake8
```
Lint rules: max line length 120, ignores E731, E741, F541, W504.

### Build docs
```bash
tox -e docs
# or directly:
cd docs && sphinx-build -E -d _build/html-doctrees -b html . _build/html
```

### Build and release
```bash
tox -e build          # check + sdist + wheel
tox -e release-test   # upload to test PyPI
tox -e release        # upload to PyPI
```

There are **no automated tests** in this repository (no `tests/` directory).

## Architecture

### Main workflow

1. **Demand estimation** is done externally using `pyblp`. The `pyblp` results object is passed into `pyRVtest`.
2. **Markup computation** (`build_markups` in `construction.py`) computes implied markups for each candidate model using demand Jacobians/Hessians from `pyblp`.
3. **Problem setup** (`Problem` in `economies/problem.py`) assembles everything: product data, instruments, cost formulation, model formulations, and pre-computed markups.
4. **Testing** (`Problem.solve()`) runs the RV test, computing GMM fit measures, test statistics (TRV), F-statistics, and MCS p-values for all pairwise model comparisons and all instrument sets.
5. **Results** (`ProblemResults` in `results/problem_results.py`) stores and displays all test outputs.

### Key classes

- **`Formulation`** (`configurations/formulation.py`) — R-style formula for cost shifters (`w`) and instruments (`Z`). Wraps PyBLP's `Formulation` with patsy/sympy.
- **`ModelFormulation`** (`configurations/formulation.py`) — Specifies a single conduct model: downstream/upstream model type, ownership columns, taxes, vertical integration, custom markups.
- **`Products`** (`primitives.py`) — Structured record array of product data. Validates and organizes `market_ids`, `shares`, `prices`, cost shifters `w`, and instruments `Z`.
- **`Models`** (`primitives.py`) — Dictionary-like structure holding per-model configurations (ownership matrices, tax vectors, markups, etc.) for all candidate models.
- **`Problem`** (`economies/problem.py`) — The central object. Takes `cost_formulation`, `instrument_formulation` (list of `Formulation`s, one per instrument set), `model_formulations` (list of `ModelFormulation`s), `product_data`, and `pyblp_results`. Calls `build_markups` internally then exposes `.solve()`.
- **`ProblemResults`** (`results/problem_results.py`) — Output of `.solve()`. Key attributes: `TRV` (RV test statistics), `F` (scaled F-statistics), `MCS_pvalues`, `markups`, `marginal_cost`, `taus`.

### Supported conduct models

`model_downstream` / `model_upstream` in `ModelFormulation` can be:
- `'bertrand'` — price-setting with ownership matrix
- `'cournot'` — quantity-setting with ownership matrix
- `'monopoly'` — full collusion
- `'perfect_competition'` — zero markups
- `'mix_cournot_bertrand'` — mixed market (requires `mix_flag`)
- `'other'` — custom formula via `custom_model_specification`

### Vertical integration / bilateral oligopoly

When both `model_downstream` and `model_upstream` are specified, the package computes upstream markups using the Villas-Boas (2007) passthrough formula (`construct_passthrough_matrix` in `construction.py`) and sums them with downstream markups (adjusted for vertical integration via the `vertical_integration` column).

### Multiple instrument sets

`instrument_formulation` accepts a list of `Formulation` objects. Each one defines a separate set of testing instruments (`Z0`, `Z1`, …). The test is run for each instrument set independently, and results arrays are indexed by instrument set.

### Data module

`pyRVtest/data/` contains CSV tables of critical values for the F-statistic size and power diagnostics (loaded via `read_critical_values_tables()` in `primitives.py`).

### Dependency on PyBLP internals

The package imports directly from `pyblp.utilities.basics`, `pyblp.utilities.algebra`, and `pyblp.configurations.formulation`. Changes in PyBLP's internal API can break this package.
