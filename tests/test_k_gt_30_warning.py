"""v0.4.0rc1: K > 30 falls back to K=30 critical values with a UserWarning.

Background (Lorenzo 2026-04-18 review memo, P1 item 6 / audit B2): the
tabulated F-statistic critical values (size and power) are defined for
K=1..30. Previously, instrument counts above 30 silently fell back to
the K=30 row with no indication to the user. rc1 emits a ``UserWarning``
at the fallback site.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import pyRVtest


def _make_fixture_with_many_instruments(n_instruments: int, seed: int = 1) -> pd.DataFrame:
    """Tiny DGP with pre-computed markups and ``n_instruments`` columns ``z0..z{N-1}``.

    T is chosen large enough that the stacked [w, Z] matrix is full-rank for
    the collinearity check at ``Problem.__init__``.
    """
    rng = np.random.default_rng(seed=seed)
    T, J = 100, 2
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
    markups_m1 = np.zeros(N)
    for t in range(T):
        idx = np.where(market_ids == t)[0]
        s_t = shares[idx]
        fids = firm_ids[idx]
        O = (fids[:, None] == fids[None, :]).astype(float)
        D = alpha * (np.diag(s_t) - np.outer(s_t, s_t))
        markups_m1[idx] = -np.linalg.solve(O * D.T, s_t).flatten()
    data = {
        'market_ids': market_ids,
        'firm_ids': firm_ids,
        'prices': prices,
        'shares': shares,
        'markups_m1': markups_m1,
        'markups_m2': np.zeros(N),
        'cost_shifter': rng.uniform(0.5, 1.5, size=N),
    }
    for k in range(n_instruments):
        data[f'z{k}'] = rng.normal(size=N)
    return pd.DataFrame(data)


def _instrument_formula(n: int) -> str:
    return '0 + ' + ' + '.join(f'z{k}' for k in range(n))


class TestKGreaterThan30Warning:
    def test_k_31_emits_user_warning(self):
        """K_effective = 31 triggers the fallback warning."""
        df = _make_fixture_with_many_instruments(n_instruments=31)
        pyRVtest.options.verbose = False
        problem = pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
            instrument_formulation=pyRVtest.Formulation(_instrument_formula(31)),
            product_data=df,
            demand_results=None,
            models=[
                pyRVtest.UserSuppliedMarkups(markups='markups_m1', ownership='firm_ids'),
                pyRVtest.UserSuppliedMarkups(markups='markups_m2', ownership='firm_ids'),
            ],
        )
        with pytest.warns(UserWarning, match='exceeds the range'):
            problem.solve()

    def test_k_30_does_not_warn(self):
        """K=30 is still tabulated; no warning."""
        df = _make_fixture_with_many_instruments(n_instruments=30)
        pyRVtest.options.verbose = False
        problem = pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
            instrument_formulation=pyRVtest.Formulation(_instrument_formula(30)),
            product_data=df,
            demand_results=None,
            models=[
                pyRVtest.UserSuppliedMarkups(markups='markups_m1', ownership='firm_ids'),
                pyRVtest.UserSuppliedMarkups(markups='markups_m2', ownership='firm_ids'),
            ],
        )
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            problem.solve()
        k_warnings = [
            w for w in captured
            if issubclass(w.category, UserWarning) and 'exceeds the range' in str(w.message)
        ]
        assert not k_warnings, (
            f"Unexpected K>30 warning at K=30: {[str(w.message) for w in k_warnings]}"
        )

    def test_warning_names_instrument_set_and_K(self):
        """The warning message names both K_effective and the instrument set index."""
        df = _make_fixture_with_many_instruments(n_instruments=32)
        pyRVtest.options.verbose = False
        problem = pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
            instrument_formulation=pyRVtest.Formulation(_instrument_formula(32)),
            product_data=df,
            demand_results=None,
            models=[
                pyRVtest.UserSuppliedMarkups(markups='markups_m1', ownership='firm_ids'),
                pyRVtest.UserSuppliedMarkups(markups='markups_m2', ownership='firm_ids'),
            ],
        )
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            problem.solve()
        msgs = [str(w.message) for w in captured if 'exceeds the range' in str(w.message)]
        assert msgs, "Expected at least one K>30 warning."
        assert any('K_effective=32' in m for m in msgs), (
            f"Expected K_effective in message; got {msgs!r}"
        )
        assert any('instrument set 0' in m for m in msgs), (
            f"Expected instrument-set index in message; got {msgs!r}"
        )
