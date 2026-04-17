"""v0.4 step 18: per-module logger emission tests.

The step 18 refactor replaced every ``print()`` / pyblp-``output()``
progress message in pyRVtest with a call on a per-module
:class:`logging.Logger`. This test file pins the resulting behaviour so
future refactors cannot accidentally regress it:

1. ``Problem.__init__`` and ``Problem.solve`` emit at least one
   ``INFO``-level record on the ``pyRVtest.problem`` logger.
2. Setting that logger's level to ``WARNING`` silences the records
   (without touching sibling loggers).
3. The legacy ``pyRVtest.output.output()`` shim still works, emits
   a :class:`DeprecationWarning` on first call, and forwards its
   argument through the ``pyRVtest.output`` logger at INFO level.
4. The :class:`logging.NullHandler` convention — pyRVtest does not
   install its own handlers at import, so users control their own
   handler / level configuration.

The fixture uses ``user_supplied_markups=`` on both models to avoid a
full pyblp demand solve; this test is about logger wiring, not
numerical behaviour. See ``tests/test_snapshots.py`` for numerical
coverage.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import pytest

import pyRVtest


# ---------------------------------------------------------------------------
# Tiny DGP with pre-computed markups.
# ---------------------------------------------------------------------------

def _make_tiny_dgp(seed: int = 1, T: int = 6, J: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed=seed)
    N = T * J
    market_ids = np.repeat(np.arange(T), J)
    firm_ids = np.tile(np.arange(J), T)
    alpha = -2.0
    x1 = rng.normal(size=N)
    prices = rng.uniform(0.8, 2.0, size=N)
    u = 0.4 * x1 + rng.normal(scale=0.2, size=N)
    delta = u + alpha * prices
    shares = np.zeros(N)
    for t in range(T):
        idx = np.where(market_ids == t)[0]
        e = np.exp(delta[idx])
        shares[idx] = e / (1.0 + e.sum())
    rival_x1 = np.zeros(N)
    for t in range(T):
        idx = np.where(market_ids == t)[0]
        for j in idx:
            others = [i for i in idx if i != j]
            rival_x1[j] = x1[others].mean()
    z1 = rng.normal(size=N) + 1.5
    markups_m1 = np.zeros(N)
    for t in range(T):
        idx = np.where(market_ids == t)[0]
        s_t = shares[idx]
        fids = firm_ids[idx]
        O = (fids[:, None] == fids[None, :]).astype(float)
        D = alpha * (np.diag(s_t) - np.outer(s_t, s_t))
        markups_m1[idx] = -np.linalg.solve(O * D.T, s_t).flatten()
    markups_m2 = np.zeros(N)
    return pd.DataFrame({
        'market_ids': market_ids,
        'firm_ids': firm_ids,
        'x1': x1,
        'prices': prices,
        'shares': shares,
        'rival_x1': rival_x1,
        'z1': z1,
        'markups_m1': markups_m1,
        'markups_m2': markups_m2,
        'cost_shifter': rng.uniform(0.5, 1.5, size=N),
    })


def _make_problem(df: pd.DataFrame) -> pyRVtest.Problem:
    return pyRVtest.Problem(
        cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
        instrument_formulation=pyRVtest.Formulation('0 + z1'),
        product_data=df,
        demand_results=None,
        models=[
            pyRVtest.Bertrand(ownership='firm_ids', user_supplied_markups='markups_m1'),
            pyRVtest.PerfectCompetition(user_supplied_markups='markups_m2'),
        ],
    )


# ---------------------------------------------------------------------------
# 1. Problem.__init__ / Problem.solve emit INFO on pyRVtest.problem.
# ---------------------------------------------------------------------------

def test_problem_init_emits_info_on_problem_logger(caplog: pytest.LogCaptureFixture) -> None:
    """``Problem.__init__`` logs at least one INFO record on ``pyRVtest.problem``.

    The specific message asserted — ``"Initializing the problem ..."`` —
    is stable v0.3-baseline progress text that the step 18 refactor
    deliberately preserved. If this assertion ever fails because the
    string was rewritten, that is step 19's territory; bump the
    substring here along with the message rewrite.
    """
    df = _make_tiny_dgp()
    with caplog.at_level(logging.INFO, logger='pyRVtest.problem'):
        _make_problem(df)
    records = [r for r in caplog.records if r.name == 'pyRVtest.problem']
    assert records, (
        "Problem.__init__ emitted no records on the 'pyRVtest.problem' "
        "logger. v0.4 step 18 wired progress messages to "
        "logging.getLogger(__name__); check that the import + logger "
        "assignment at the top of pyRVtest/problem.py survived."
    )
    # At least one INFO-level record matching the init banner.
    init_info = [
        r for r in records
        if r.levelno == logging.INFO and 'Initializing the problem' in r.getMessage()
    ]
    assert init_info, (
        "Expected at least one INFO record containing 'Initializing the "
        "problem' from pyRVtest.problem. Got messages: "
        f"{[r.getMessage() for r in records]}"
    )


def test_problem_solve_emits_info_on_problem_logger(caplog: pytest.LogCaptureFixture) -> None:
    """``Problem.solve`` also logs at INFO on ``pyRVtest.problem``."""
    df = _make_tiny_dgp()
    problem = _make_problem(df)
    with caplog.at_level(logging.INFO, logger='pyRVtest.problem'):
        problem.solve()
    records = [r for r in caplog.records if r.name == 'pyRVtest.problem']
    solve_info = [
        r for r in records
        if r.levelno == logging.INFO and 'Solving the problem' in r.getMessage()
    ]
    assert solve_info, (
        "Expected at least one INFO record containing 'Solving the "
        "problem' from pyRVtest.problem during solve()."
    )


# ---------------------------------------------------------------------------
# 2. Raising the logger's level silences the emission.
# ---------------------------------------------------------------------------

def test_setting_problem_logger_warning_silences_info(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Setting ``pyRVtest.problem`` to WARNING drops the INFO records.

    This is the public contract advertised in docs/agent_guide.rst:
    ``logging.getLogger('pyRVtest.problem').setLevel(logging.WARNING)``
    silences a subsystem without affecting the rest of pyRVtest. Pytest's
    caplog fixture sees records only at or above the configured level.
    """
    df = _make_tiny_dgp()
    problem_logger = logging.getLogger('pyRVtest.problem')
    original_level = problem_logger.level
    try:
        problem_logger.setLevel(logging.WARNING)
        with caplog.at_level(logging.WARNING, logger='pyRVtest.problem'):
            _make_problem(df)
        info_records = [
            r for r in caplog.records
            if r.name == 'pyRVtest.problem' and r.levelno == logging.INFO
        ]
        assert not info_records, (
            "Setting the pyRVtest.problem logger to WARNING should have "
            "dropped all INFO records, but some were still captured: "
            f"{[r.getMessage() for r in info_records]}"
        )
    finally:
        problem_logger.setLevel(original_level)


# ---------------------------------------------------------------------------
# 3. Legacy output() shim: still works, still warns, still routes to logger.
# ---------------------------------------------------------------------------

def test_legacy_output_shim_forwards_to_logger(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``pyRVtest.output.output(msg)`` is the one-release compatibility shim.

    Requirements (v0.4 step 18 brief):
    - forwards ``str(msg)`` to the ``pyRVtest.output`` logger at INFO
    - emits a ``DeprecationWarning`` on the first call in a session

    The module keeps a session-local flag so the warning fires once;
    this test resets that flag so it runs deterministically regardless
    of which other tests in the suite happened to import the shim first.
    """
    from pyRVtest import output as output_module
    output_module._output_shim_deprecation_warned = False  # type: ignore[attr-defined]

    with caplog.at_level(logging.INFO, logger='pyRVtest.output'):
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            output_module.output("hello from the shim")

    # The deprecation warning fires.
    dep_warnings = [w for w in captured if issubclass(w.category, DeprecationWarning)]
    assert dep_warnings, (
        "Expected a DeprecationWarning from pyRVtest.output.output(), "
        "but none was captured."
    )

    # And the message goes through the pyRVtest.output logger at INFO.
    logger_records = [
        r for r in caplog.records
        if r.name == 'pyRVtest.output' and r.levelno == logging.INFO
    ]
    assert any('hello from the shim' in r.getMessage() for r in logger_records), (
        "Expected 'hello from the shim' on the pyRVtest.output logger at "
        f"INFO level. Captured records: {[(r.name, r.levelno, r.getMessage()) for r in caplog.records]}"
    )


def test_legacy_output_shim_warns_once_per_session(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The once-per-session flag matches the rest of pyRVtest's deprecation hygiene."""
    from pyRVtest import output as output_module
    output_module._output_shim_deprecation_warned = False  # type: ignore[attr-defined]

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        output_module.output("first call")
        output_module.output("second call")
        output_module.output("third call")

    dep_warnings = [w for w in captured if issubclass(w.category, DeprecationWarning)]
    assert len(dep_warnings) == 1, (
        "pyRVtest.output.output() should emit its DeprecationWarning "
        f"exactly once per session; got {len(dep_warnings)} warnings."
    )


# ---------------------------------------------------------------------------
# 4. pyRVtest does not install any handlers on import.
# ---------------------------------------------------------------------------

def test_pyrvtest_installs_no_handlers_on_import() -> None:
    """Library-quietude: importing pyRVtest must not attach handlers.

    The standard-library convention is that libraries emit records and
    let the application configure handlers. Installing a StreamHandler
    at import time would double-print when the application also calls
    logging.basicConfig().
    """
    # Top-level pyRVtest logger.
    root_logger = logging.getLogger('pyRVtest')
    # Handlers other than NullHandler are the ones we do not want.
    bad_handlers = [
        h for h in root_logger.handlers
        if not isinstance(h, logging.NullHandler)
    ]
    assert not bad_handlers, (
        "pyRVtest installed non-NullHandler loggers at import time: "
        f"{bad_handlers}. Libraries should not configure handlers; let "
        "the application call logging.basicConfig()."
    )
