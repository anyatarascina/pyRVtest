# pyRVtest test suite

The bulk of the test suite lives at the top level of `tests/`
(`test_analytical.py`, `test_demand_adjustment.py`,
`test_first_stage_correction.py`, `test_f_reliability.py`, etc.). This
README covers the supporting subdirectories.

## Layout

```
tests/
├── conftest.py            # shared session fixtures (tiny_product_data, tiny_pyblp_results)
├── fixtures/
│   └── tiny_synthetic.py  # deterministic toy datasets
├── regressions/           # bug-class regression guards, one per memo blocker
├── replication/           # external-paper replication suites
├── snapshots/             # frozen-output goldens
└── test_*.py              # primary unit / integration tests
```

## Regressions subdirectory

Each test in `regressions/` is a guard against a specific bug from the
memo-1 audit (`MEMO_pyRVtest_CClean_review_2026-04-14.md`). Tests that
currently fail are marked with `pytest.mark.xfail(strict=True)` and a
reason pointing to the followup that will resolve them; once the fix
lands, the test flips to XPASS and the marker is removed.

## Running

```bash
pytest tests/regressions/ -v   # regression guards only
pytest tests/                   # full suite
```

## Adding tests

New bugs should land here as regression guards before the fix is
written. New features ship with tests in the appropriate top-level
`test_*.py` file.
