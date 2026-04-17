"""v0.4 OQ 15: ``ProblemResults.passthrough_comparison`` diagnostic.

Tests the Dearing-style pass-through distance diagnostic added to
:class:`pyRVtest.ProblemResults`. The method surfaces
:func:`pyRVtest.build_passthrough` in a pairwise / pandas-friendly form
so users can flag candidate-model pairs with near-identical pass-through
matrices (weakly identified under pass-through-based instruments; see
Dearing et al. 2026 Remark 4).

Covers:

* DataFrame shape and column contract.
* Frobenius metric matches a hand-computed :math:`\\lVert P_i - P_j \\rVert_F`
  reduction.
* ``offdiag_frobenius`` is invariant to diagonal-only perturbations.
* ``max_abs`` returns the element-wise maximum absolute difference.
* ``market_id=<scalar>`` filters to that market (``n_pairs`` rows).
* ``attrs['metric']`` is recorded.
* Non-Vertical candidate model -> ``NotImplementedError`` with a clear
  v0.5 pointer.
* ``metric='invalid'`` -> ``ValidationError`` (``ValueError`` subclass).
* Invalid ``market_id`` -> ``ValidationError``.
* ``passthrough_matrix`` wrapper behaves like ``build_passthrough``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyRVtest
from pyRVtest import (
    Bertrand,
    Cournot,
    Monopoly,
    PerfectCompetition,
    Vertical,
)
from pyRVtest.exceptions import ValidationError


# ---------------------------------------------------------------------------
# Shared DGP
# ---------------------------------------------------------------------------
#
# Minimal plain-logit DGP lifted from ``tests/test_passthrough.py`` so the
# fixture cost and the model layout are familiar. We instantiate TWO
# Vertical models with different downstream / upstream conduct so the
# pass-through matrices genuinely differ — that way the distance metric
# tests are non-degenerate.

def _make_vertical_dgp(
    seed: int = 1234, T: int = 4, J: int = 4, alpha: float = -1.5,
):
    """Plain-logit DGP with upstream firm ids for Vertical composition."""
    rng = np.random.default_rng(seed=seed)
    N = T * J
    market_ids = np.repeat(np.arange(T), J)
    firm_ids = np.tile(np.arange(J), T)
    upstream_firm_ids = np.tile(['m1', 'm1', 'm2', 'm2'], T)
    vi_col = np.zeros(N, dtype=int)
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
    df = pd.DataFrame({
        'market_ids': market_ids,
        'firm_ids': firm_ids,
        'upstream_firm_ids': upstream_firm_ids,
        'vi_col': vi_col,
        'x1': x1,
        'prices': prices,
        'shares': shares,
        'rival_x1': rival_x1,
        'z1': z1,
        'intercept': np.ones(N),
    })
    return df, alpha


def _make_two_vertical_problem():
    """Problem with TWO Vertical candidate models (Bertrand-up-Monopoly
    and Cournot-up-Monopoly) so pass-through matrices differ."""
    df, alpha = _make_vertical_dgp()
    pyRVtest.options.verbose = False
    return pyRVtest.Problem(
        cost_formulation=pyRVtest.Formulation('1 + z1'),
        instrument_formulation=pyRVtest.Formulation('0 + rival_x1'),
        product_data=df,
        demand_params={
            'alpha': alpha, 'rho': [],
            'beta': np.array([0.0, 0.4]),
            'x_columns': ['intercept', 'x1'],
            'demand_instrument_columns': ['rival_x1', 'intercept', 'x1'],
        },
        models=[
            Vertical(
                downstream=Bertrand(ownership='firm_ids'),
                upstream=Monopoly(ownership='upstream_firm_ids'),
                vertical_integration='vi_col',
            ),
            Vertical(
                downstream=Cournot(ownership='firm_ids'),
                upstream=Monopoly(ownership='upstream_firm_ids'),
                vertical_integration='vi_col',
            ),
        ],
    )


def _make_mixed_vertical_and_nonvertical_problem():
    """Problem with ONE Vertical + ONE PerfectCompetition model.

    Used for the ``NotImplementedError`` path: the comparison method
    rejects any non-Vertical candidate.
    """
    df, alpha = _make_vertical_dgp()
    pyRVtest.options.verbose = False
    return pyRVtest.Problem(
        cost_formulation=pyRVtest.Formulation('1 + z1'),
        instrument_formulation=pyRVtest.Formulation('0 + rival_x1'),
        product_data=df,
        demand_params={
            'alpha': alpha, 'rho': [],
            'beta': np.array([0.0, 0.4]),
            'x_columns': ['intercept', 'x1'],
            'demand_instrument_columns': ['rival_x1', 'intercept', 'x1'],
        },
        models=[
            Vertical(
                downstream=Bertrand(ownership='firm_ids'),
                upstream=Monopoly(ownership='upstream_firm_ids'),
                vertical_integration='vi_col',
            ),
            PerfectCompetition(),
        ],
    )


@pytest.fixture(scope='module')
def two_vertical_results():
    problem = _make_two_vertical_problem()
    return problem.solve()


@pytest.fixture(scope='module')
def mixed_results():
    problem = _make_mixed_vertical_and_nonvertical_problem()
    return problem.solve()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPassthroughComparisonShape:
    """Column contract and filter behaviour."""

    def test_returns_dataframe_with_expected_columns(self, two_vertical_results):
        df = two_vertical_results.passthrough_comparison()
        assert isinstance(df, pd.DataFrame)
        expected_cols = [
            'market_id', 'model_i', 'model_j',
            'model_i_label', 'model_j_label',
            'distance', 'metric',
        ]
        assert list(df.columns) == expected_cols

    def test_one_row_per_market_per_pair(self, two_vertical_results):
        df = two_vertical_results.passthrough_comparison()
        n_markets = len(two_vertical_results.problem.unique_market_ids)
        # M=2 candidate models -> one unordered pair.
        n_pairs = 1
        assert len(df) == n_markets * n_pairs
        # Only the (0, 1) unordered pair should appear.
        assert (df['model_i'] < df['model_j']).all()
        assert df[['model_i', 'model_j']].drop_duplicates().shape[0] == 1

    def test_market_id_filter(self, two_vertical_results):
        unique_markets = two_vertical_results.problem.unique_market_ids
        target = unique_markets[0]
        df = two_vertical_results.passthrough_comparison(market_id=target)
        # One pair per market for M=2 candidate models.
        assert len(df) == 1
        assert df['market_id'].iloc[0] == target

    def test_attrs_metric_recorded(self, two_vertical_results):
        df = two_vertical_results.passthrough_comparison(metric='max_abs')
        assert df.attrs['metric'] == 'max_abs'

    def test_default_metric_is_frobenius(self, two_vertical_results):
        df = two_vertical_results.passthrough_comparison()
        assert df.attrs['metric'] == 'frobenius'
        assert (df['metric'] == 'frobenius').all()


class TestPassthroughComparisonMetrics:
    """Metric-specific numerical checks.

    Cross-checks each metric against a hand computation using the
    pass-through matrices returned by ``build_passthrough`` directly.
    This is the same math ``passthrough_comparison`` is supposed to do,
    but written out separately so a regression in the dispatch / frame
    construction would surface.
    """

    def test_frobenius_matches_hand_computation(self, two_vertical_results):
        problem = two_vertical_results.problem
        P0 = pyRVtest.build_passthrough(problem, 0)
        P1 = pyRVtest.build_passthrough(problem, 1)
        df = two_vertical_results.passthrough_comparison(metric='frobenius')
        for _, row in df.iterrows():
            t = row['market_id']
            expected = float(np.linalg.norm(P0[t] - P1[t], ord='fro'))
            np.testing.assert_allclose(
                row['distance'], expected, atol=1e-14,
                err_msg=(
                    f"Frobenius distance for market {t!r} disagrees "
                    f"with hand computation."
                ),
            )

    def test_offdiag_frobenius_ignores_diagonal_only_differences(
        self, two_vertical_results,
    ):
        """If two pass-through matrices differ ONLY on the diagonal, the
        off-diagonal Frobenius norm is zero. Simulated by subtracting
        the diagonal from the difference matrix and passing through the
        same metric helper used by the method.
        """
        problem = two_vertical_results.problem
        P0 = pyRVtest.build_passthrough(problem, 0)
        P1 = pyRVtest.build_passthrough(problem, 1)
        # Construct a modified "difference" where we zero out everything
        # except the diagonal: that is, a matrix that differs on the
        # diagonal only. Its off-diagonal Frobenius norm must be zero.
        for t in problem.unique_market_ids:
            diag_only = np.diag(np.diag(P0[t] - P1[t]))
            distance = two_vertical_results._passthrough_distance(
                diag_only, metric='offdiag_frobenius',
            )
            assert distance == pytest.approx(0.0, abs=1e-14)

    def test_offdiag_frobenius_matches_hand_computation(self, two_vertical_results):
        """The off-diagonal Frobenius norm equals ``sqrt(sum((P_i - P_j)
        _{kl}^2) for k != l)`` -- the explicit element sum is easy to
        hand-check.
        """
        problem = two_vertical_results.problem
        P0 = pyRVtest.build_passthrough(problem, 0)
        P1 = pyRVtest.build_passthrough(problem, 1)
        df = two_vertical_results.passthrough_comparison(
            metric='offdiag_frobenius',
        )
        for _, row in df.iterrows():
            t = row['market_id']
            diff = P0[t] - P1[t]
            off = diff - np.diag(np.diag(diff))
            expected = float(np.sqrt(np.sum(off ** 2)))
            np.testing.assert_allclose(
                row['distance'], expected, atol=1e-14,
                err_msg=(
                    f"Off-diagonal Frobenius distance for market {t!r} "
                    f"disagrees with hand computation."
                ),
            )

    def test_max_abs_matches_hand_computation(self, two_vertical_results):
        problem = two_vertical_results.problem
        P0 = pyRVtest.build_passthrough(problem, 0)
        P1 = pyRVtest.build_passthrough(problem, 1)
        df = two_vertical_results.passthrough_comparison(metric='max_abs')
        for _, row in df.iterrows():
            t = row['market_id']
            expected = float(np.max(np.abs(P0[t] - P1[t])))
            np.testing.assert_allclose(
                row['distance'], expected, atol=1e-14,
                err_msg=(
                    f"max_abs distance for market {t!r} disagrees with "
                    f"hand computation."
                ),
            )

    def test_frobenius_ge_offdiag(self, two_vertical_results):
        """Full Frobenius >= off-diagonal Frobenius (adding diagonal
        entries to the sum-of-squares can only grow the norm)."""
        df_full = two_vertical_results.passthrough_comparison(
            metric='frobenius',
        )
        df_off = two_vertical_results.passthrough_comparison(
            metric='offdiag_frobenius',
        )
        # Rows line up because they are constructed in the same order.
        for a, b in zip(df_full['distance'], df_off['distance']):
            assert a >= b - 1e-14


class TestPassthroughComparisonErrors:
    """Validation and NotImplementedError paths."""

    def test_invalid_metric_raises_validation_error(self, two_vertical_results):
        with pytest.raises(ValidationError, match="metric"):
            two_vertical_results.passthrough_comparison(metric='invalid')

    def test_invalid_metric_is_also_value_error(self, two_vertical_results):
        """``ValidationError`` subclasses ``ValueError``; legacy callers
        using ``except ValueError:`` continue to catch this case."""
        with pytest.raises(ValueError):
            two_vertical_results.passthrough_comparison(metric='L1')

    def test_invalid_market_id_raises_validation_error(self, two_vertical_results):
        with pytest.raises(ValidationError, match="market_id"):
            two_vertical_results.passthrough_comparison(market_id=9999)

    def test_non_vertical_model_raises_not_implemented(self, mixed_results):
        with pytest.raises(NotImplementedError, match="Vertical"):
            mixed_results.passthrough_comparison()

    def test_non_vertical_error_mentions_v0_5(self, mixed_results):
        """The error should point users at the v0.5 scope note."""
        with pytest.raises(NotImplementedError, match="v0.5"):
            mixed_results.passthrough_comparison()


class TestPassthroughMatrixWrapper:
    """Thin wrapper over ``build_passthrough``."""

    def test_matches_build_passthrough_single_market(self, two_vertical_results):
        problem = two_vertical_results.problem
        t = problem.unique_market_ids[0]
        wrapped = two_vertical_results.passthrough_matrix(
            model_index=0, market_id=t,
        )
        expected = pyRVtest.build_passthrough(problem, 0, market_id=t)
        np.testing.assert_allclose(wrapped, expected, atol=1e-14)

    def test_matches_build_passthrough_all_markets(self, two_vertical_results):
        problem = two_vertical_results.problem
        wrapped = two_vertical_results.passthrough_matrix(model_index=1)
        expected = pyRVtest.build_passthrough(problem, 1)
        assert isinstance(wrapped, dict)
        assert set(wrapped.keys()) == set(expected.keys())
        for t in wrapped:
            np.testing.assert_allclose(wrapped[t], expected[t], atol=1e-14)
