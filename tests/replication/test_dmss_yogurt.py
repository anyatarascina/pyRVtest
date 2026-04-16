"""v0.4 Step 0d: DMSS yogurt golden-file replication.

Pinned replication of the DMSS (Duarte, Magnolfi, Solvsten, Sullivan 2024,
QE) yogurt analysis. If pyRVtest produces TRV / F / MCS p-values that
differ from published table values by more than the stated tolerance,
the math is wrong — not just the code. Per the rollback-trigger rule
in .claude/plans/v0.4-refactor.md §5, a >1% drift here is the nuclear
trigger: abandon v0.4-refactor, investigate before restarting.

## Status: PLACEHOLDER

This file is scaffolding only. Each test is marked @pytest.mark.skip
pending inputs from Lorenzo. See the `NEEDED FROM LORENZO` block below
for the specific items required before the tests can be un-skipped.

Once filled in, this file becomes the strongest single protection for
the v0.4 refactor: it anchors pyRVtest output to a real, published
empirical specification that coauthors and reviewers can verify
independently against the paper.

## NEEDED FROM LORENZO

1. **Data location.** Where does the DMSS yogurt product_data live?
   Options:
     - A path in the scalable-testing-markups repo
     - A separate data package on PyPI or conda
     - A file committed to this repo under pyRVtest/data/
   Add the path (or construction script) at `_load_dmss_yogurt_data()`
   below.

2. **Specification.** Which exact specification to pin?
     - Paper table / column number (e.g., "Table 3 column 2")
     - Demand side: logit? nested logit? BLP? With which instruments?
     - Cost side: which w columns? FEs (absorb)?
     - Conduct models: which pair(s) tested against each other?
     - Demand adjustment: True/False? Clustering: True/False?

3. **Expected numeric values from the paper.** Pin TRV, F, MCS-p for
   each instrument set to 4-5 significant figures. Populate the
   `EXPECTED_*` constants below.

4. **Tolerance.** Default is `rtol=1e-4, atol=1e-6` per the plan's
   rollback-trigger discussion (>1% = nuclear). Adjust if Lorenzo's
   published values are rounded to fewer digits.

## How the harness works once populated

- `dmss_yogurt_results` fixture: loads data + runs Problem.solve() per
  the pinned specification. Module-scoped so the expensive demand
  estimation runs once.
- Per-quantity tests assert close to published values. Each test has
  its own skip reason so partial population (e.g., TRV known, F
  unknown) is possible.

## Why this is a separate file from tests/test_snapshots.py

Snapshots (0b) lock _current_ pyRVtest output to prevent silent drift
during the refactor. The golden file (0d) locks _paper-published_
output; any drift from the paper is wrong by definition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Pinned expected values from DMSS (2024) Table __. Replace None with real
# floats once Lorenzo confirms. Each value is a 5-sig-fig number from the
# published paper.
# ---------------------------------------------------------------------------

# TRV[instrument_set][model_i, model_j] — matrix of pairwise RV test stats.
# None until Lorenzo provides.
EXPECTED_TRV: dict[int, list[list[float]]] | None = None

# F[instrument_set][model_i, model_j] — scaled F-statistics.
EXPECTED_F: dict[int, list[list[float]]] | None = None

# MCS_pvalues[instrument_set][model] — Model Confidence Set p-values per model.
EXPECTED_MCS: dict[int, list[float]] | None = None

# Default tolerances (relative | absolute). Override per-test if paper values
# are rounded to fewer digits than the default 5.
DEFAULT_RTOL = 1e-4
DEFAULT_ATOL = 1e-6


def _load_dmss_yogurt_data() -> pd.DataFrame:
    """Load the DMSS yogurt dataset.

    NOT YET IMPLEMENTED. Lorenzo: replace this stub with a loader that
    returns a pandas DataFrame with the columns pyRVtest expects
    (market_ids, firm_ids, shares, prices, plus whatever demand-side
    and instrument columns the specification needs).
    """
    raise NotImplementedError(
        "DMSS yogurt data loader not yet implemented. "
        "See NEEDED FROM LORENZO block at top of this file."
    )


def _solve_dmss_yogurt_specification():
    """Run Problem.solve() with the pinned DMSS yogurt specification.

    NOT YET IMPLEMENTED. Once Lorenzo provides the specification, build
    the Formulation, ModelFormulation tuple, and Problem() here; call
    .solve() with the right flags; and return the ProblemResults.
    """
    raise NotImplementedError(
        "DMSS yogurt specification not yet pinned. "
        "See NEEDED FROM LORENZO block at top of this file."
    )


# ---------------------------------------------------------------------------
# Fixture (skipped until EXPECTED_* constants are populated)
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def dmss_yogurt_results():
    if EXPECTED_TRV is None:
        pytest.skip(
            "DMSS yogurt golden file not yet pinned. "
            "Awaiting Lorenzo's input on data location, specification, and "
            "published expected values. See v0.4 plan §5 Step 0d."
        )
    return _solve_dmss_yogurt_specification()


# ---------------------------------------------------------------------------
# Per-quantity tests (skipped until fixture produces real results)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Awaiting Lorenzo's input on DMSS yogurt expected values — v0.4 plan §5 Step 0d")
def test_dmss_yogurt_trv_matches_published(dmss_yogurt_results):
    """TRV for each instrument set matches DMSS Table __ column __."""
    results = dmss_yogurt_results
    for inst_set_idx, expected_matrix in EXPECTED_TRV.items():
        np.testing.assert_allclose(
            results.TRV[inst_set_idx], np.array(expected_matrix),
            rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL,
            err_msg=f"TRV mismatch for instrument set {inst_set_idx}"
        )


@pytest.mark.skip(reason="Awaiting Lorenzo's input on DMSS yogurt expected values — v0.4 plan §5 Step 0d")
def test_dmss_yogurt_f_matches_published(dmss_yogurt_results):
    """Scaled F-statistics match published values."""
    results = dmss_yogurt_results
    for inst_set_idx, expected_matrix in EXPECTED_F.items():
        np.testing.assert_allclose(
            results.F[inst_set_idx], np.array(expected_matrix),
            rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL,
            err_msg=f"F mismatch for instrument set {inst_set_idx}"
        )


@pytest.mark.skip(reason="Awaiting Lorenzo's input on DMSS yogurt expected values — v0.4 plan §5 Step 0d")
def test_dmss_yogurt_mcs_matches_published(dmss_yogurt_results):
    """MCS p-values match published values."""
    results = dmss_yogurt_results
    for inst_set_idx, expected_pvalues in EXPECTED_MCS.items():
        # MCS_pvalues shape varies; flatten per-model-index for comparison
        actual = np.asarray(results.MCS_pvalues[inst_set_idx]).flatten()
        np.testing.assert_allclose(
            actual, np.array(expected_pvalues),
            rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL,
            err_msg=f"MCS p-values mismatch for instrument set {inst_set_idx}"
        )
