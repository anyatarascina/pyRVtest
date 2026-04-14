# pyRVtest test suite

Scaffolding for the regression test strategy described in
`MEMO_pyRVtest_test_strategy_2026-04-14.md` (pyRVtest project folder).

## Layout

```
tests/
├── conftest.py            # shared session fixtures (tiny_product_data, tiny_pyblp_results)
├── fixtures/
│   └── tiny_synthetic.py  # deterministic toy datasets
├── regressions/
│   ├── test_memo1_section41_f_diagnostic.py
│   ├── test_memo1_section42_adjustment_interaction.py
│   └── test_memo1_section43_gradient_no_fes.py
├── unit/                  # to be populated (see memo §2.1)
├── integration/           # to be populated (see memo §2.2)
├── properties/            # to be populated (see memo §2.3)
├── goldens/               # to be populated (see memo §2.4)
└── benchmarks/            # to be populated (see memo §2.5)
```

## Red-green discipline

Each test in `regressions/` is written to **fail on current code** and
**pass after the corresponding fix**. They use `pytest.mark.xfail(strict=True)`
so that:

- on current code, pytest reports `xfail (expected)` and the suite passes
- after the fix, the test unexpectedly passes and pytest reports `XPASS`,
  forcing the author to remove the xfail marker and confirming the fix

This encodes the memo 1 correctness items in the test suite before they are
fixed, making it impossible for any of them to silently regress later.

## Running

```bash
pytest tests/regressions/ -v
```

Expected output today: three `xfail` lines, zero failures.
Expected output after memo 1 §5.1 fixes: three `XPASS` lines, suite fails
until the xfail markers are removed from the fixed items.

## Adding tests

Every new bug found during development should be expressed as a regression
test here before the fix is written. Every new feature should ship with a
test in `unit/`, `integration/`, or `properties/` depending on scope.
