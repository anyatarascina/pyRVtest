"""Direct unit tests for :func:`pyRVtest.markups._compute_markups`.

Scaffolding for the rc10 batched-markups change. ``_compute_markups`` is
load-bearing for every ``Problem.solve`` call: it produces the per-model
markups consumed by the GMM / RV / F / MCS pipeline. Previously only
exercised through Problem.solve and via the build_markups public API;
this file pins its output directly so any internal change shows up
here first instead of cascading through ``Problem.solve`` and showing
up as a snapshot diff (which is harder to diagnose).

Coverage:

* Bertrand:      ``markup = -solve(O * D, s)`` per market.
* Cournot:       ``markup = -(O * inv(D)) @ s`` per market.
* Monopoly:      ``markup = -solve(D, s)`` per market.
* PerfectComp:   ``markup = 0`` per row.
* PartialColl:   bertrand math with kappa-modified ownership.
* Mixed candidate set (B + C + PC): per-model markup distinct.
* ConstantMarkup: ``markup[j] = zeta[j]`` per row.
* User-supplied: short-circuit returns the user column unchanged.
* Variable J_t:  markets with different product counts mixed in one
                 Problem.

All assertions are at ``atol=1e-12``. Anything looser silently lets
last-ULP changes through, which defeats the purpose of pinning the
function for refactor safety.

Pre-rc10 (scalar per-market loop) and rc10+ (batched LAPACK per
J-group for the safe conduct types) must agree to this tolerance.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest

import pyRVtest
from pyRVtest.backends.logit import LogitBackend
from pyRVtest.markups import _compute_markups


# ---------------------------------------------------------------------------
# Fixture: small Bertrand-style logit DGP shared across most tests.
# ---------------------------------------------------------------------------


def _build_logit_fixture(T: int = 5, J: int = 2, seed: int = 12321) -> Tuple[
    pd.DataFrame, float, np.ndarray
]:
    """Tiny logit DGP. Shares and prices drawn directly (don't need to be
    equilibrium for these tests — only need self-consistent xi and a
    well-conditioned demand Jacobian).
    """
    rng = np.random.default_rng(seed=seed)
    N = T * J
    alpha = -2.0
    market_ids = np.repeat(np.arange(T), J)
    firm_ids = np.tile(np.arange(J), T)
    intercept = np.ones(N)
    x1 = rng.uniform(0.5, 2.0, N)
    rival_x1 = np.zeros(N)
    for t in range(T):
        idx = np.where(market_ids == t)[0]
        for k_in, k_idx in enumerate(idx):
            rival_x1[k_idx] = x1[idx].sum() - x1[k_idx]
    z1 = rng.uniform(0.5, 2.0, N)
    rival_z1 = np.zeros(N)
    for t in range(T):
        idx = np.where(market_ids == t)[0]
        for k_in, k_idx in enumerate(idx):
            rival_z1[k_idx] = z1[idx].sum() - z1[k_idx]
    prices = rng.uniform(1.0, 3.0, N)
    # Build shares so they sum to <= 1 per market.
    shares = np.zeros(N)
    for t in range(T):
        idx = np.where(market_ids == t)[0]
        raw = rng.uniform(0.1, 0.4, J)
        if raw.sum() >= 0.95:
            raw = raw * 0.4 / raw.sum()
        shares[idx] = raw
    df = pd.DataFrame({
        'market_ids': market_ids,
        'firm_ids': firm_ids,
        'intercept': intercept,
        'x1': x1,
        'rival_x1': rival_x1,
        'z1': z1,
        'rival_z1': rival_z1,
        'prices': prices,
        'shares': shares,
    })
    return df, alpha, np.array([0.1, 1.3])  # beta = [intercept, x1]


def _backend_for(df: pd.DataFrame, alpha: float, beta: np.ndarray) -> LogitBackend:
    return LogitBackend(
        alpha=alpha, product_data=df, beta=beta,
        x_columns=['intercept', 'x1'],
        demand_instrument_columns=['rival_x1', 'intercept', 'x1'],
    )


def _build_ownership_per_firm(df: pd.DataFrame) -> np.ndarray:
    """(N, J_max) per-market ownership block, NaN-padded across markets.

    The contract expected by _compute_markups: ``ownership[index_t]`` is
    a (J_t, J_max) slice where columns past J_t are NaN, and the kept
    columns form the (J_t, J_t) ownership matrix for market t.
    """
    market_ids = df['market_ids'].to_numpy()
    firm_ids = df['firm_ids'].to_numpy()
    T_unique = sorted(np.unique(market_ids).tolist())
    J_max = max(int((market_ids == t).sum()) for t in T_unique)
    N = len(market_ids)
    own = np.full((N, J_max), np.nan)
    for t in T_unique:
        idx = np.where(market_ids == t)[0]
        J_t = len(idx)
        firm_t = firm_ids[idx]
        for r in range(J_t):
            for c in range(J_t):
                own[idx[r], c] = 1.0 if firm_t[r] == firm_t[c] else 0.0
    return own


# ---------------------------------------------------------------------------
# Hand-computed reference markups per conduct.
# ---------------------------------------------------------------------------


def _expected_bertrand_markups(df: pd.DataFrame, alpha: float) -> np.ndarray:
    """Bertrand: markup_t = -solve(O_t * D_t, s_t) per market."""
    market_ids = df['market_ids'].to_numpy()
    shares = df['shares'].to_numpy()
    firm_ids = df['firm_ids'].to_numpy()
    out = np.zeros(len(df))
    for t in np.unique(market_ids):
        idx = np.where(market_ids == t)[0]
        s_t = shares[idx]
        D_t = alpha * (np.diag(s_t) - np.outer(s_t, s_t))
        firm_t = firm_ids[idx]
        O_t = (firm_t[:, None] == firm_t[None, :]).astype(float)
        out[idx] = -np.linalg.solve(O_t * D_t, s_t.reshape(-1, 1)).flatten()
    return out


def _expected_cournot_markups(df: pd.DataFrame, alpha: float) -> np.ndarray:
    """Cournot: markup_t = -(O_t * inv(D_t)) @ s_t per market."""
    market_ids = df['market_ids'].to_numpy()
    shares = df['shares'].to_numpy()
    firm_ids = df['firm_ids'].to_numpy()
    out = np.zeros(len(df))
    for t in np.unique(market_ids):
        idx = np.where(market_ids == t)[0]
        s_t = shares[idx]
        D_t = alpha * (np.diag(s_t) - np.outer(s_t, s_t))
        firm_t = firm_ids[idx]
        O_t = (firm_t[:, None] == firm_t[None, :]).astype(float)
        out[idx] = (-(O_t * np.linalg.inv(D_t)) @ s_t.reshape(-1, 1)).flatten()
    return out


def _expected_monopoly_markups(df: pd.DataFrame, alpha: float) -> np.ndarray:
    """Monopoly: markup_t = -solve(D_t, s_t) per market."""
    market_ids = df['market_ids'].to_numpy()
    shares = df['shares'].to_numpy()
    out = np.zeros(len(df))
    for t in np.unique(market_ids):
        idx = np.where(market_ids == t)[0]
        s_t = shares[idx]
        D_t = alpha * (np.diag(s_t) - np.outer(s_t, s_t))
        out[idx] = -np.linalg.solve(D_t, s_t.reshape(-1, 1)).flatten()
    return out


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


class TestSingleConductBatchable:
    """Single-conduct tests for the four batchable types."""

    @pytest.fixture(scope='class')
    def fixture(self):
        return _build_logit_fixture()

    def test_bertrand_markups_match_hand_formula(self, fixture):
        df, alpha, beta = fixture
        backend = _backend_for(df, alpha, beta)
        own = _build_ownership_per_firm(df)
        markups, mdown, mup = _compute_markups(
            product_data=df.to_records(index=False),
            pyblp_results=None,
            model_downstream=['bertrand'],
            ownership_downstream=[own],
            demand_backend=backend,
        )
        expected = _expected_bertrand_markups(df, alpha)
        np.testing.assert_allclose(
            np.asarray(markups[0]).flatten(), expected, atol=1e-12,
            err_msg='bertrand markups deviated from hand formula',
        )

    def test_cournot_markups_match_hand_formula(self, fixture):
        df, alpha, beta = fixture
        backend = _backend_for(df, alpha, beta)
        own = _build_ownership_per_firm(df)
        markups, *_ = _compute_markups(
            product_data=df.to_records(index=False),
            pyblp_results=None,
            model_downstream=['cournot'],
            ownership_downstream=[own],
            demand_backend=backend,
        )
        expected = _expected_cournot_markups(df, alpha)
        np.testing.assert_allclose(
            np.asarray(markups[0]).flatten(), expected, atol=1e-12,
            err_msg='cournot markups deviated from hand formula',
        )

    def test_monopoly_markups_match_hand_formula(self, fixture):
        df, alpha, beta = fixture
        backend = _backend_for(df, alpha, beta)
        # Monopoly = single-firm ownership: every row claims every column.
        N = len(df)
        T = int(df['market_ids'].max() + 1)
        J_max = int((df['market_ids'] == 0).sum())
        own = np.full((N, J_max), np.nan)
        for t in range(T):
            idx = np.where(df['market_ids'].to_numpy() == t)[0]
            J_t = len(idx)
            for r in range(J_t):
                for c in range(J_t):
                    own[idx[r], c] = 1.0
        markups, *_ = _compute_markups(
            product_data=df.to_records(index=False),
            pyblp_results=None,
            model_downstream=['monopoly'],
            ownership_downstream=[own],
            demand_backend=backend,
        )
        expected = _expected_monopoly_markups(df, alpha)
        np.testing.assert_allclose(
            np.asarray(markups[0]).flatten(), expected, atol=1e-12,
            err_msg='monopoly markups deviated from hand formula',
        )

    def test_perfect_competition_markups_are_zero(self, fixture):
        df, alpha, beta = fixture
        backend = _backend_for(df, alpha, beta)
        own = _build_ownership_per_firm(df)  # ignored by perfect_competition
        markups, *_ = _compute_markups(
            product_data=df.to_records(index=False),
            pyblp_results=None,
            model_downstream=['perfect_competition'],
            ownership_downstream=[own],
            demand_backend=backend,
        )
        np.testing.assert_allclose(
            np.asarray(markups[0]).flatten(),
            np.zeros(len(df)),
            atol=1e-12,
            err_msg='perfect_competition markups should be zero',
        )


class TestMixedCandidateSet:
    """All four batchable conducts in one Problem — verify per-model output
    is the same as solving each in isolation."""

    @pytest.fixture(scope='class')
    def fixture(self):
        return _build_logit_fixture()

    def test_per_model_markups_match_isolated_runs(self, fixture):
        df, alpha, beta = fixture
        backend = _backend_for(df, alpha, beta)
        own_firm = _build_ownership_per_firm(df)
        N = len(df)
        J_max = own_firm.shape[1]
        # Monopoly ownership: all-claim block.
        own_mono = np.full((N, J_max), np.nan)
        T = int(df['market_ids'].max() + 1)
        for t in range(T):
            idx = np.where(df['market_ids'].to_numpy() == t)[0]
            J_t = len(idx)
            for r in range(J_t):
                for c in range(J_t):
                    own_mono[idx[r], c] = 1.0

        records = df.to_records(index=False)
        markups, *_ = _compute_markups(
            product_data=records,
            pyblp_results=None,
            model_downstream=['bertrand', 'cournot', 'monopoly', 'perfect_competition'],
            ownership_downstream=[own_firm, own_firm, own_mono, own_firm],
            demand_backend=backend,
        )
        expected = [
            _expected_bertrand_markups(df, alpha),
            _expected_cournot_markups(df, alpha),
            _expected_monopoly_markups(df, alpha),
            np.zeros(N),
        ]
        labels = ['bertrand', 'cournot', 'monopoly', 'perfect_competition']
        for k, (lab, exp) in enumerate(zip(labels, expected)):
            np.testing.assert_allclose(
                np.asarray(markups[k]).flatten(), exp, atol=1e-12,
                err_msg=f'{lab} markup deviated in mixed-set run',
            )


class TestUserSuppliedShortCircuit:
    """User-supplied markups are returned unchanged (per-market loop never
    entered for that model)."""

    @pytest.fixture(scope='class')
    def fixture(self):
        return _build_logit_fixture()

    def test_user_supplied_markups_pass_through(self, fixture):
        df, alpha, beta = fixture
        backend = _backend_for(df, alpha, beta)
        own = _build_ownership_per_firm(df)
        N = len(df)
        user_col = np.arange(N, dtype=float).reshape(-1, 1) * 0.01 + 1.0
        markups, mdown, _ = _compute_markups(
            product_data=df.to_records(index=False),
            pyblp_results=None,
            model_downstream=['bertrand'],  # ignored for user-supplied
            ownership_downstream=[own],
            demand_backend=backend,
            user_supplied_markups=[user_col],
        )
        np.testing.assert_allclose(
            np.asarray(markups[0]).flatten(), user_col.flatten(), atol=1e-12,
            err_msg='user-supplied markups should pass through unchanged',
        )
        np.testing.assert_allclose(
            np.asarray(mdown[0]).flatten(), user_col.flatten(), atol=1e-12,
            err_msg='downstream markups should equal user-supplied',
        )


class TestVariableJ:
    """Mix markets with different product counts in one Problem (J=2 and
    J=3). Pre-rc10 the scalar per-market loop handles this naturally;
    rc10 batched path must group by J_t."""

    @pytest.fixture(scope='class')
    def fixture(self):
        # Two markets of J=2, one market of J=3.
        rng = np.random.default_rng(seed=999)
        N = 7
        market_ids = np.array([0, 0, 1, 1, 2, 2, 2])
        firm_ids = np.array([0, 1, 0, 1, 0, 1, 2])
        shares = np.array([0.3, 0.2, 0.4, 0.3, 0.2, 0.25, 0.3])
        prices = rng.uniform(1.0, 2.0, N)
        intercept = np.ones(N)
        x1 = rng.uniform(0.5, 1.5, N)
        rival_x1 = np.array([x1[1], x1[0], x1[3], x1[2], x1[5] + x1[6], x1[4] + x1[6], x1[4] + x1[5]])
        df = pd.DataFrame({
            'market_ids': market_ids, 'firm_ids': firm_ids,
            'intercept': intercept, 'x1': x1, 'rival_x1': rival_x1,
            'shares': shares, 'prices': prices,
        })
        alpha = -2.0
        beta = np.array([0.1, 1.3])
        return df, alpha, beta

    def test_bertrand_mixed_J_match_hand_formula(self, fixture):
        df, alpha, beta = fixture
        backend = _backend_for(df, alpha, beta)
        own = _build_ownership_per_firm(df)
        markups, *_ = _compute_markups(
            product_data=df.to_records(index=False),
            pyblp_results=None,
            model_downstream=['bertrand'],
            ownership_downstream=[own],
            demand_backend=backend,
        )
        expected = _expected_bertrand_markups(df, alpha)
        np.testing.assert_allclose(
            np.asarray(markups[0]).flatten(), expected, atol=1e-12,
            err_msg='bertrand markups with variable J_t mismatch',
        )


# ---------------------------------------------------------------------------
# rc10 scaffolding (B): scalar-vs-batched parity over random fixtures.
# ---------------------------------------------------------------------------


class TestScalarBatchedParity:
    """Random-fixture parity between the rc10 batched path and the
    pre-rc10 per-market scalar loop.

    Calls ``_compute_markups`` (the new batched path) and compares the
    output to a reconstruction that goes through
    ``evaluate_first_order_conditions`` per market. They must agree to
    ``atol=1e-12``. If a future change tightens the batched path away
    from the scalar formula, this fires.
    """

    @pytest.mark.parametrize('model_type', ['bertrand', 'cournot', 'monopoly'])
    @pytest.mark.parametrize('seed', [101, 202, 303])
    def test_batched_path_matches_scalar_loop(self, model_type, seed):
        df, alpha, beta = _build_logit_fixture(T=8, J=2, seed=seed)
        backend = _backend_for(df, alpha, beta)
        own = (
            _build_ownership_per_firm(df) if model_type != 'monopoly'
            else _make_full_block_ownership(df)
        )

        # Batched path (current code).
        markups_batched, *_ = _compute_markups(
            product_data=df.to_records(index=False),
            pyblp_results=None,
            model_downstream=[model_type],
            ownership_downstream=[own],
            demand_backend=backend,
        )

        # Scalar reconstruction: do exactly what the per-market loop
        # would have done, bypassing the batched dispatch.
        from pyRVtest.markups import evaluate_first_order_conditions
        N = len(df)
        markups_scalar = np.zeros((N, 1))
        ds_dp = backend.compute_jacobian()
        market_ids = df['market_ids'].to_numpy()
        for t in np.unique(market_ids):
            idx_t = np.where(market_ids == t)[0]
            shares_t = df['shares'].to_numpy()[idx_t]
            resp_t = ds_dp[idx_t]
            resp_t = resp_t[:, ~np.isnan(resp_t).all(axis=0)]
            markups_scalar, _ = evaluate_first_order_conditions(
                idx_t, model_type, own, resp_t, shares_t,
                markups_scalar, None, markup_type='downstream',
            )
        np.testing.assert_allclose(
            np.asarray(markups_batched[0]).flatten(),
            markups_scalar.flatten(),
            atol=1e-12,
            err_msg=(
                f'batched and scalar markups disagree for '
                f'model_type={model_type!r}, seed={seed}'
            ),
        )


def _make_full_block_ownership(df: pd.DataFrame) -> np.ndarray:
    """Monopoly: every row claims every column in its market."""
    market_ids = df['market_ids'].to_numpy()
    N = len(market_ids)
    J_max = max(int((market_ids == t).sum()) for t in np.unique(market_ids))
    own = np.full((N, J_max), np.nan)
    for t in np.unique(market_ids):
        idx = np.where(market_ids == t)[0]
        J_t = len(idx)
        for r in range(J_t):
            for c in range(J_t):
                own[idx[r], c] = 1.0
    return own


# ---------------------------------------------------------------------------
# rc10 scaffolding (E): batchability decision matrix.
# ---------------------------------------------------------------------------


class TestBatchabilityMatrix:
    """Pin which (downstream, upstream, mix, custom) tuples route to the
    batched path. If a future change widens or narrows the batchable set
    without updating this test, the gate is wrong and CI catches it."""

    @pytest.mark.parametrize('model_type, expected', [
        ('bertrand', True),
        ('cournot', True),
        ('monopoly', True),
        ('perfect_competition', True),
        ('constant_markup', True),
        ('mix_cournot_bertrand', False),
        ('other', False),  # custom-model marker
        (None, False),
    ])
    def test_batchable_by_model_type(self, model_type, expected):
        from pyRVtest.markups import _is_batchable_downstream_model
        assert _is_batchable_downstream_model(
            model_downstream=model_type,
            model_upstream=None,
            mix_flag=None,
            custom_model_specification=None,
        ) is expected

    def test_upstream_disqualifies_batching(self):
        """Any non-None upstream forces the scalar Villas-Boas path."""
        from pyRVtest.markups import _is_batchable_downstream_model
        assert _is_batchable_downstream_model(
            model_downstream='bertrand',
            model_upstream='bertrand',
            mix_flag=None,
            custom_model_specification=None,
        ) is False

    def test_mix_flag_disqualifies_batching(self):
        from pyRVtest.markups import _is_batchable_downstream_model
        mix_flag = np.array([True, False, True, False])
        assert _is_batchable_downstream_model(
            model_downstream='bertrand',
            model_upstream=None,
            mix_flag=mix_flag,
            custom_model_specification=None,
        ) is False

    def test_custom_spec_disqualifies_batching(self):
        from pyRVtest.markups import _is_batchable_downstream_model
        spec = {'name': lambda O, D, s: -np.linalg.solve(O * D, s)}
        assert _is_batchable_downstream_model(
            model_downstream='bertrand',
            model_upstream=None,
            mix_flag=None,
            custom_model_specification=spec,
        ) is False
