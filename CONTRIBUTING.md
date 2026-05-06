# Contributing to pyRVtest

Thanks for your interest in contributing. This guide covers the bits
you need before submitting a pull request: setting up the dev
environment, running the test suite and linters, building the docs,
and the conventions for adding new conduct models or demand backends.

For an architectural orientation, read [`AGENTS.md`](AGENTS.md) at the
repo root or run `pyRVtest.show_agent_guide()` from a Python session.

## Dev environment

Python 3.7+ is supported at install time; development targets Python
3.9. The package depends on NumPy, pandas, statsmodels, and pyblp;
keep the NumPy / pyblp pin combinations in sync (either `numpy<2` with
`pyblp<1.2` or `numpy>=2` with `pyblp>=1.2`; mixed pins fail because
pyblp 1.1.x uses `np.unicode_` which NumPy 2 removed).

Install everything you need to run the test suite, lint, and build the
docs:

```bash
git clone git@github.com:anyatarascina/pyRVtest.git
cd pyRVtest
pip install -e .
pip install -r requirements-dev.txt
pip install '.[docs]'  # for docs/
```

## Running tests

Full suite:

```bash
pytest
```

The `tests/replication/` suite needs external data fixtures (the DMSS
yogurt dataset) and skips automatically when not installed. Excluding
it on a fresh checkout:

```bash
pytest --ignore=tests/replication
```

Mypy strict typing is enforced by `tests/test_mypy_strict.py`. Run it
explicitly:

```bash
pytest tests/test_mypy_strict.py
```

The `tests/regressions/` directory holds bug-class regression guards.
Some of them are marked `xfail(strict=True)` for known-but-deferred
issues tracked in `.claude/plans/v0.5-followups.md`; if your change
fixes a tracked issue, its xfail will flip to XPASS and you should
remove the marker as part of the same PR.

## Linting

```bash
flake8
# or via tox
tox -e flake8
```

Style is enforced at `max-line-length=120` with a small ignore list
(`E731`, `E741`, `F541`, `W504`); see `setup.cfg`. Don't bypass these
with `# noqa` unless there's a specific reason — comment the reason
inline if you do.

## Building docs

```bash
tox -e docs
# or directly
cd docs && sphinx-build -E -d _build/html-doctrees -b html . _build/html
```

The Sphinx config lives at `docs/conf.py`; the theme is
`sphinx_rtd_theme`. Notebooks are processed via `nbsphinx`; rebuild
times are dominated by the notebook execution step. If you only
changed `.rst` files, you can skip notebook re-execution by setting
`nbsphinx_execute = 'never'` in your local `conf.py` for the
duration.

## Adding a new conduct model

The class-based `ConductModel` API is in `pyRVtest/models/`. To add a
new conduct hypothesis:

1. Subclass `ConductModel` (in `pyRVtest/models/base.py`) in a file
   under `pyRVtest/models/`. Use `pyRVtest/models/standard.py` (where
   Bertrand, Cournot, Monopoly, PerfectCompetition live) as the
   reference for the simplest case.
2. Implement `_compute_markup(self, prices, shares, jacobian, ...)`
   returning the per-product markup vector implied by your
   conduct hypothesis.
3. Implement `_markup_derivative(self, ...)` if the markup depends on
   prices through a closed-form derivative (used for the analytical
   demand-adjustment path; defer to finite differences if not).
4. Re-export the class from `pyRVtest/models/__init__.py` and from
   `pyRVtest/__init__.py` (`__all__`).
5. Add a unit test in `tests/test_<your_model>.py` covering: markup
   computation on a tiny synthetic case, integration with
   `Problem.solve` end-to-end, and any interesting edge cases (zero
   shares, single-product markets, etc.).
6. Add an entry to `docs/api.rst` under the *Conduct models* section
   and a paragraph in the model-library notebook
   (`docs/notebooks/model_library.ipynb`).

If your model has fundamentally different inputs from the standard
hierarchy (vertical structure, non-price strategic variable, etc.),
look at `pyRVtest/models/vertical.py` and
`pyRVtest/models/mixed.py` for examples of more involved patterns.

## Adding a new demand backend

The `DemandBackend` Protocol is in `pyRVtest/backends/base.py`.
Existing backends (`PyBLPBackend`, `LogitBackend`, `NestedLogitBackend`,
`UserSuppliedBackend`) are reference implementations. To add a new
one:

1. Subclass nothing (the protocol is structural per PEP 544); just
   implement the required methods: `compute_jacobian`,
   `compute_hessian`, `perturbed`, `n_parameters`, `theta_names`.
2. If the backend can supply first-stage demand-adjustment inputs
   (`xi`, `Z_D`, `W_D`, gradients), also implement the
   `SupportsDemandAdjustment` mixin so the
   `Problem.solve(demand_adjustment=True)` path can find them.
3. If users will pass the backend through `demand_params=`, decide
   how the backend maps to that dict format.
4. Test in `tests/test_backends.py` (or a new file). The existing
   tests show the parity checks against the analytic backends.

The `docs/custom_demand.rst` page covers the user-facing
`UserSuppliedBackend` path; if your new backend will be available to
end users, add a parallel doc page.

## Branch and PR conventions

- Branch off `v0.4-refactor` (the active development branch); the
  `main` branch lags v0.4 work.
- Keep commits focused and self-describing — multi-paragraph commit
  messages are encouraged when a change is non-trivial.
- Bug fixes that close a tracked v0.5 follow-up: reference the item
  number from `.claude/plans/v0.5-followups.md` in the commit body
  and remove the corresponding entry once merged.
- New features ship with tests and docs in the same PR.
- Run `flake8`, `pytest --ignore=tests/replication`, and
  `tests/test_mypy_strict.py` before pushing.

## Where else to look

- [`AGENTS.md`](AGENTS.md) — full architectural contract.
- `docs/agent_guide.rst` — the longer-form architectural walkthrough
  rendered on Read the Docs.
- `.claude/plans/` — open design plans and v0.5 follow-up tracker.
- `.claude/handovers/` — session-by-session work log.
- `CHANGELOG.md` — user-facing changelog.
