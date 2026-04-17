"""v0.4 step 13: tests for product- and labor-side instrument constructors.

Each helper is exercised on a hand-built fixture where the expected output
is computed by hand so the assertion pins the exact arithmetic. Tolerance
is ``atol=1e-12`` throughout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyRVtest.instruments import (
    bartik,
    blp_instruments,
    differentiation_ivs,
    hausman,
    rival_sums,
)


# ---------------------------------------------------------------------------
# Shared fixture: T=3 markets, J=3 products per market, 2 firms. Firm 1 owns
# the first two products in each market; firm 2 owns the third.
# ---------------------------------------------------------------------------
@pytest.fixture
def toy_panel() -> pd.DataFrame:
    df = pd.DataFrame({
        'market_ids': [0, 0, 0, 1, 1, 1, 2, 2, 2],
        'firm_ids': [1, 1, 2, 1, 1, 2, 1, 1, 2],
        'x': [1.0, 2.0, 4.0, 3.0, 5.0, 7.0, 6.0, 8.0, 10.0],
        'shares': [0.2, 0.3, 0.5, 0.1, 0.4, 0.5, 0.25, 0.25, 0.5],
    })
    return df


ATOL = 1e-12


# ---------------------------------------------------------------------------
# rival_sums
# ---------------------------------------------------------------------------

def test_rival_sums_toy_panel(toy_panel):
    # Market 0: firm1 rows j=0,1 (x=1,2); firm2 row j=2 (x=4).
    #   j=0 rival = x[2] = 4
    #   j=1 rival = x[2] = 4
    #   j=2 rival = x[0]+x[1] = 3
    # Market 1: firm1 rows 3,4 (x=3,5); firm2 row 5 (x=7).
    #   j=3 rival = 7; j=4 rival = 7; j=5 rival = 8
    # Market 2: firm1 rows 6,7 (x=6,8); firm2 row 8 (x=10).
    #   j=6 rival = 10; j=7 rival = 10; j=8 rival = 14
    expected = np.array([4, 4, 3, 7, 7, 8, 10, 10, 14], dtype=float)
    got = rival_sums(toy_panel, 'x')
    assert np.allclose(got, expected, atol=ATOL)


def test_rival_sums_firm_owns_all_market():
    """A product whose firm owns every product in its market gets 0."""
    df = pd.DataFrame({
        'market_ids': [0, 0, 0],
        'firm_ids': [1, 1, 1],
        'x': [1.0, 2.0, 3.0],
    })
    got = rival_sums(df, 'x')
    assert np.allclose(got, np.zeros(3), atol=ATOL)


def test_rival_sums_custom_column_names():
    df = pd.DataFrame({
        'mkt': [0, 0, 0],
        'f': [1, 2, 2],
        'z': [5.0, 3.0, 7.0],
    })
    got = rival_sums(df, 'z', firm_id_column='f', market_id_column='mkt')
    # j=0 (firm 1): rivals = 3+7 = 10
    # j=1 (firm 2): rival = 5 (from j=0)
    # j=2 (firm 2): rival = 5
    assert np.allclose(got, np.array([10.0, 5.0, 5.0]), atol=ATOL)


# ---------------------------------------------------------------------------
# differentiation_ivs
# ---------------------------------------------------------------------------

def test_differentiation_ivs_toy_panel(toy_panel):
    out = differentiation_ivs(toy_panel, 'x')
    assert set(out.keys()) == {'sum_squared_diff_rival', 'sum_squared_diff_same_firm'}

    # Market 0: x=[1,2,4], firms=[1,1,2].
    # j=0: rival={k=2}: (1-4)^2 = 9; same_excl_self={k=1}: (1-2)^2 = 1
    # j=1: rival={k=2}: (2-4)^2 = 4; same_excl_self={k=0}: (2-1)^2 = 1
    # j=2: rival={k=0,1}: 9+4=13; same_excl_self={}: 0
    # Market 1: x=[3,5,7], firms=[1,1,2].
    # j=3: rival={k=5}: (3-7)^2=16; same={k=4}: (3-5)^2=4
    # j=4: rival={k=5}: (5-7)^2=4; same={k=3}: 4
    # j=5: rival={k=3,4}: 16+4=20; same={}: 0
    # Market 2: x=[6,8,10], firms=[1,1,2].
    # j=6: rival={k=8}:16; same={k=7}:4
    # j=7: rival={k=8}:4;  same={k=6}:4
    # j=8: rival={k=6,7}:16+4=20; same={}:0
    exp_rival = np.array([9, 4, 13, 16, 4, 20, 16, 4, 20], dtype=float)
    exp_same = np.array([1, 1, 0, 4, 4, 0, 4, 4, 0], dtype=float)
    assert np.allclose(out['sum_squared_diff_rival'], exp_rival, atol=ATOL)
    assert np.allclose(out['sum_squared_diff_same_firm'], exp_same, atol=ATOL)


def test_differentiation_ivs_symmetry_and_shape():
    """Rival sums are symmetric between firms in the 2-firm 2-product case."""
    df = pd.DataFrame({
        'market_ids': [0, 0],
        'firm_ids': [1, 2],
        'x': [2.0, 5.0],
    })
    out = differentiation_ivs(df, 'x')
    # Both rows have exactly one rival with the same squared diff.
    expected_rival = np.array([(2 - 5) ** 2, (5 - 2) ** 2], dtype=float)
    assert np.allclose(out['sum_squared_diff_rival'], expected_rival, atol=ATOL)
    # Neither row has a same-firm partner.
    assert np.allclose(out['sum_squared_diff_same_firm'], np.zeros(2), atol=ATOL)


# ---------------------------------------------------------------------------
# blp_instruments
# ---------------------------------------------------------------------------

def test_blp_instruments_toy_panel(toy_panel):
    out = blp_instruments(toy_panel, ['x'])
    expected_keys = {'x_own_sum', 'x_rival_sum', 'own_count', 'rival_count'}
    assert set(out.keys()) == expected_keys

    # Market 0: firms=[1,1,2], x=[1,2,4].
    #   j=0: own={k=1}:2, rival={k=2}:4
    #   j=1: own={k=0}:1, rival={k=2}:4
    #   j=2: own={}:0,    rival={k=0,1}:3
    # Market 1: firms=[1,1,2], x=[3,5,7].
    #   j=3: own={k=4}:5, rival={k=5}:7
    #   j=4: own={k=3}:3, rival={k=5}:7
    #   j=5: own=0, rival=3+5=8
    # Market 2: firms=[1,1,2], x=[6,8,10].
    #   j=6: own=8, rival=10
    #   j=7: own=6, rival=10
    #   j=8: own=0, rival=14
    exp_own = np.array([2, 1, 0, 5, 3, 0, 8, 6, 0], dtype=float)
    exp_rival = np.array([4, 4, 3, 7, 7, 8, 10, 10, 14], dtype=float)
    # Own count: each firm-1 product has one same-firm partner; firm-2 products are alone.
    exp_own_count = np.array([1, 1, 0, 1, 1, 0, 1, 1, 0], dtype=int)
    exp_rival_count = np.array([1, 1, 2, 1, 1, 2, 1, 1, 2], dtype=int)
    assert np.allclose(out['x_own_sum'], exp_own, atol=ATOL)
    assert np.allclose(out['x_rival_sum'], exp_rival, atol=ATOL)
    assert np.array_equal(out['own_count'], exp_own_count)
    assert np.array_equal(out['rival_count'], exp_rival_count)


def test_blp_instruments_multiple_columns(toy_panel):
    out = blp_instruments(toy_panel, ['x', 'shares'])
    assert 'x_own_sum' in out
    assert 'x_rival_sum' in out
    assert 'shares_own_sum' in out
    assert 'shares_rival_sum' in out
    # Count keys are present exactly once (not duplicated per column).
    assert 'own_count' in out
    assert 'rival_count' in out
    # Rival_sum(shares) for j=0 in market 0: firm 2 share = 0.5
    # For j=2 (firm 2): rival = 0.2 + 0.3 = 0.5
    assert out['shares_rival_sum'][0] == pytest.approx(0.5, abs=ATOL)
    assert out['shares_rival_sum'][2] == pytest.approx(0.5, abs=ATOL)


def test_blp_instruments_rival_sum_matches_rival_sums(toy_panel):
    """blp_instruments('c')['c_rival_sum'] must equal rival_sums(..., 'c')."""
    out = blp_instruments(toy_panel, ['x'])
    direct = rival_sums(toy_panel, 'x')
    assert np.allclose(out['x_rival_sum'], direct, atol=ATOL)


# ---------------------------------------------------------------------------
# hausman
# ---------------------------------------------------------------------------

def test_hausman_leave_one_market_out(toy_panel):
    # Total sum of x = 1+2+4+3+5+7+6+8+10 = 46; N = 9.
    # Market 0 rows see mean of x in markets 1 and 2:
    #   sum = 3+5+7+6+8+10 = 39; count = 6; mean = 6.5
    # Market 1 rows see markets 0, 2:
    #   sum = 1+2+4+6+8+10 = 31; count = 6; mean = 31/6
    # Market 2 rows see markets 0, 1:
    #   sum = 1+2+4+3+5+7 = 22; count = 6; mean = 22/6
    got = hausman(toy_panel, 'x')
    expected = np.array([
        6.5, 6.5, 6.5,
        31 / 6, 31 / 6, 31 / 6,
        22 / 6, 22 / 6, 22 / 6,
    ])
    assert np.allclose(got, expected, atol=ATOL)


def test_hausman_period_restriction():
    df = pd.DataFrame({
        'market_ids': [0, 0, 1, 1, 0, 0, 1, 1],
        'period': [0, 0, 0, 0, 1, 1, 1, 1],
        'x': [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
    })
    # Period 0: rows 0,1 (market 0), rows 2,3 (market 1). Leave-one-out by
    # market within period: rows 0,1 get mean of [3,4] = 3.5; rows 2,3 get
    # mean of [1,2] = 1.5.
    # Period 1: similar, rows 4,5 get mean([30,40]) = 35; rows 6,7 get
    # mean([10,20]) = 15.
    got = hausman(df, 'x', period_column='period')
    expected = np.array([3.5, 3.5, 1.5, 1.5, 35.0, 35.0, 15.0, 15.0])
    assert np.allclose(got, expected, atol=ATOL)


def test_hausman_single_market_returns_nan():
    df = pd.DataFrame({'market_ids': [0, 0, 0], 'x': [1.0, 2.0, 3.0]})
    got = hausman(df, 'x')
    assert np.all(np.isnan(got))


# ---------------------------------------------------------------------------
# bartik
# ---------------------------------------------------------------------------

def test_bartik_formula():
    df = pd.DataFrame({
        'market_ids': [0, 0, 1, 1],
        'w': [1.0, 2.0, 3.0, 4.0],
        's': [10.0, 20.0, 30.0, 40.0],
    })
    # total_s = 100; N = 4.
    # loo_mean_j = (100 - s_j) / 3.
    # j=0: (100-10)/3 = 30 -> w*loo = 1*30 = 30
    # j=1: (100-20)/3 = 80/3 -> 2*80/3 = 160/3
    # j=2: (100-30)/3 = 70/3 -> 3*70/3 = 70
    # j=3: (100-40)/3 = 20   -> 4*20 = 80
    got = bartik(df, 'w', 's')
    expected = np.array([30.0, 160 / 3, 70.0, 80.0])
    assert np.allclose(got, expected, atol=ATOL)


def test_bartik_zero_weight_zeroes_output():
    df = pd.DataFrame({
        'market_ids': [0, 0, 1, 1],
        'w': [0.0, 0.0, 1.0, 0.0],
        's': [1.0, 2.0, 3.0, 4.0],
    })
    got = bartik(df, 'w', 's')
    # Only row 2 is non-zero. total_s = 10. loo_mean[2] = (10-3)/3 = 7/3.
    # got[2] = 1 * 7/3.
    expected = np.array([0.0, 0.0, 7 / 3, 0.0])
    assert np.allclose(got, expected, atol=ATOL)


# ---------------------------------------------------------------------------
# Integration: helpers work on structured numpy arrays, not just DataFrames.
# ---------------------------------------------------------------------------

def test_rival_sums_on_structured_array():
    dtype = [('market_ids', 'i4'), ('firm_ids', 'i4'), ('x', 'f8')]
    arr = np.array([
        (0, 1, 1.0), (0, 1, 2.0), (0, 2, 4.0),
    ], dtype=dtype)
    got = rival_sums(arr, 'x')
    assert np.allclose(got, np.array([4.0, 4.0, 3.0]), atol=ATOL)


def test_hausman_on_dict():
    data = {
        'market_ids': np.array([0, 0, 1, 1]),
        'x': np.array([1.0, 2.0, 3.0, 4.0]),
    }
    # Market 0: leave-out mean of markets != 0 is mean([3, 4]) = 3.5.
    # Market 1: leave-out mean of markets != 1 is mean([1, 2]) = 1.5.
    got = hausman(data, 'x')
    expected = np.array([3.5, 3.5, 1.5, 1.5])
    assert np.allclose(got, expected, atol=ATOL)
