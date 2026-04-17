"""v0.4 step 6a: verify Problem.Dict_K / Container.Dict_Z_formulation are
per-instance, not class-level.

Pre-v0.4 these were declared as ``class-level mutable dict = {}`` on the
``Container`` base (Dict_Z_formulation) and ``Problem`` subclass (Dict_K).
Two concurrent Problem instances would then share the dict and each would
accumulate the other's instrument counts / Z-formulation entries. Step 6a
makes them per-instance attributes assigned in ``__init__``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyRVtest


def _make_tiny_dgp(seed: int = 1, T: int = 6, J: int = 2):
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
    z2 = rng.normal(size=N)
    z3 = rng.normal(size=N) * 0.5
    # Pre-compute markups so we can use user_supplied_markups and avoid
    # a full pyblp solve — the test is specifically about Dict_K /
    # Dict_Z_formulation, not about solve behavior.
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
        'z1': z1, 'z2': z2, 'z3': z3,
        'markups_m1': markups_m1,
        'markups_m2': markups_m2,
        'cost_shifter': rng.uniform(0.5, 1.5, size=N),
    })


def _make_problem(df, *, instrument_formulation):
    pyRVtest.options.verbose = False
    return pyRVtest.Problem(
        cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
        instrument_formulation=instrument_formulation,
        product_data=df,
        demand_results=None,
        models=[
            pyRVtest.Bertrand(ownership='firm_ids', user_supplied_markups='markups_m1'),
            pyRVtest.PerfectCompetition(user_supplied_markups='markups_m2'),
        ],
    )


class TestConcurrentProblemsDoNotShareDictK:
    """Two Problem instances with different L must have independent Dict_K."""

    def test_different_L_gives_independent_dict_k(self):
        df = _make_tiny_dgp()
        # Problem A: one instrument set with 1 instrument.
        p_a = _make_problem(df, instrument_formulation=pyRVtest.Formulation('0 + z1'))
        assert p_a.L == 1
        assert p_a.Dict_K == {'K0': 1}

        # Problem B: two instrument sets with 2 and 3 instruments respectively.
        p_b = _make_problem(df, instrument_formulation=[
            pyRVtest.Formulation('0 + z1 + z2'),
            pyRVtest.Formulation('0 + z1 + z2 + z3'),
        ])
        assert p_b.L == 2
        assert p_b.Dict_K == {'K0': 2, 'K1': 3}, p_b.Dict_K

        # And Problem A's Dict_K must NOT have been mutated by Problem B's
        # construction (the pre-v0.4 bug was that Dict_K was class-level).
        assert p_a.Dict_K == {'K0': 1}, (
            f"Problem A's Dict_K was mutated by Problem B's construction. "
            f"Got {p_a.Dict_K}; expected only K0=1."
        )

    def test_dict_k_is_not_shared_across_instances(self):
        """Direct identity check: the Dict_K attributes are distinct dicts."""
        df = _make_tiny_dgp()
        p_a = _make_problem(df, instrument_formulation=pyRVtest.Formulation('0 + z1'))
        p_b = _make_problem(df, instrument_formulation=pyRVtest.Formulation('0 + z1 + z2'))
        assert p_a.Dict_K is not p_b.Dict_K


class TestConcurrentProblemsDoNotShareDictZFormulation:
    """Two Problem instances with different instrument columns must have
    independent Dict_Z_formulation (the per-column metadata).
    """

    def test_dict_z_formulation_is_not_shared_across_instances(self):
        df = _make_tiny_dgp()
        p_a = _make_problem(df, instrument_formulation=pyRVtest.Formulation('0 + z1'))
        p_b = _make_problem(df, instrument_formulation=pyRVtest.Formulation('0 + z1 + z2'))
        assert p_a.Dict_Z_formulation is not p_b.Dict_Z_formulation
        assert '_Z0_formulation' in p_a.Dict_Z_formulation
        assert '_Z0_formulation' in p_b.Dict_Z_formulation
        # Pre-v0.4 bug: p_a's dict would accumulate p_b's Z formulation on
        # top of its own, creating confused state.
