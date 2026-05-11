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


class TestKGreaterThan30Regression:
    """Value-level regression checks for the K>30 fallback path.

    The warning suite above proves the alert fires. These tests prove the
    fallback actually computes usable diagnostics: F-stats, CV columns,
    and the rendered summary survive end-to-end when ``K_effective`` is
    above the tabulated range.
    """

    @pytest.mark.parametrize('n_instruments', [31, 33, 40])
    def test_fallback_returns_finite_f_and_cvs(self, n_instruments):
        """K>30 path returns finite F-stats and populated CV columns
        for off-diagonal (i, m) pair entries.

        F is (n_models, n_models); only entries with ``i != m`` carry
        a populated F / CV cell (the pairwise comparison). Same-model
        entries are NaN by construction.
        """
        df = _make_fixture_with_many_instruments(n_instruments=n_instruments)
        pyRVtest.options.verbose = False
        problem = pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
            instrument_formulation=pyRVtest.Formulation(_instrument_formula(n_instruments)),
            product_data=df,
            demand_results=None,
            models=[
                pyRVtest.UserSuppliedMarkups(markups='markups_m1', ownership='firm_ids'),
                pyRVtest.UserSuppliedMarkups(markups='markups_m2', ownership='firm_ids'),
            ],
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            results = problem.solve()

        # Only one instrument-set was passed in, so F / CV lists each
        # have a single per-instrument-set entry.
        assert len(results.F) == 1
        f_mat = np.asarray(results.F[0])
        assert f_mat.shape == (2, 2), f"unexpected F shape {f_mat.shape}"
        # At least one off-diagonal pair cell must be a finite, non-negative F.
        off_diag = [f_mat[i, m] for i in range(2) for m in range(2) if i != m]
        finite_off_diag = [f for f in off_diag if np.isfinite(f)]
        assert finite_off_diag, (
            f"K={n_instruments}: no finite off-diagonal F entries: {f_mat!r}"
        )
        for f in finite_off_diag:
            assert f >= 0, (
                f"K={n_instruments}: negative F off the diagonal: {f}"
            )

        cv_size = results.F_cv_size_list[0]
        cv_power = results.F_cv_power_list[0]
        assert cv_size.shape == (2, 2)
        assert cv_power.shape == (2, 2)
        # For every finite F entry the corresponding CV cell must be a
        # populated length-3 row of finite, non-negative critical values.
        for i in range(2):
            for m in range(2):
                if i == m or not np.isfinite(f_mat[i, m]):
                    continue
                size_row = np.asarray(cv_size[i, m], dtype=float)
                power_row = np.asarray(cv_power[i, m], dtype=float)
                assert size_row.shape == (3,)
                assert power_row.shape == (3,)
                assert np.all(np.isfinite(size_row))
                assert np.all(np.isfinite(power_row))
                assert np.all(size_row >= 0)
                assert np.all(power_row >= 0)

    def test_fallback_uses_k30_row(self):
        """K_eff=33 falls back to the K=30 row of the CV table.

        Asserts the looked-up critical values match the K=30 entries of
        the published table at some tabulated rho. This is the load-
        bearing claim of the fallback: stars are computed against the
        K=30 row, not against some interpolation or against K=33.
        """
        from pyRVtest.data import read_critical_values_tables
        n_instruments = 33
        df = _make_fixture_with_many_instruments(n_instruments=n_instruments)
        pyRVtest.options.verbose = False
        problem = pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
            instrument_formulation=pyRVtest.Formulation(_instrument_formula(n_instruments)),
            product_data=df,
            demand_results=None,
            models=[
                pyRVtest.UserSuppliedMarkups(markups='markups_m1', ownership='firm_ids'),
                pyRVtest.UserSuppliedMarkups(markups='markups_m2', ownership='firm_ids'),
            ],
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            results = problem.solve()

        critical_values_power, critical_values_size = read_critical_values_tables()
        k30_size = critical_values_size[critical_values_size['K'] == 30]
        f_mat = np.asarray(results.F[0])
        cv_size = results.F_cv_size_list[0]

        # Build the (rho -> [r_125, r_10, r_075]) lookup from the K=30 row.
        k30_expected = {}
        for ind in range(len(k30_size)):
            rho = float(np.round(k30_size['rho'][ind], 2))
            k30_expected[rho] = np.array([
                float(k30_size['r_125'][ind]),
                float(k30_size['r_10'][ind]),
                float(k30_size['r_075'][ind]),
            ])

        checked = 0
        for i in range(2):
            for m in range(2):
                if i == m or not np.isfinite(f_mat[i, m]):
                    continue
                row = np.asarray(cv_size[i, m], dtype=float)
                matched = any(
                    np.allclose(row, expected, atol=1e-10)
                    for expected in k30_expected.values()
                )
                assert matched, (
                    f"CV row {row!r} at (i={i}, m={m}) does not match any "
                    f"rho in the K=30 critical-values table."
                )
                checked += 1
        assert checked > 0, "Did not check any CV row — fixture produced no usable F."

    def test_summary_df_renders_with_k_gt_30(self):
        """summary_df survives the K>30 fallback end-to-end."""
        df = _make_fixture_with_many_instruments(n_instruments=33)
        pyRVtest.options.verbose = False
        problem = pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
            instrument_formulation=pyRVtest.Formulation(_instrument_formula(33)),
            product_data=df,
            demand_results=None,
            models=[
                pyRVtest.UserSuppliedMarkups(markups='markups_m1', ownership='firm_ids'),
                pyRVtest.UserSuppliedMarkups(markups='markups_m2', ownership='firm_ids'),
            ],
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            results = problem.solve()
        df_out = results.summary_df()
        assert not df_out.empty
        assert df_out.shape[0] > 0
        assert df_out.shape[1] > 0
