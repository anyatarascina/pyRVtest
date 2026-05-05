"""Synthetic product-level data generators for regression tests.

The generators produce deterministic (seed-fixed) toy datasets small enough
to run a full `Problem.solve()` in well under a second, yet rich enough to
exercise every code path in `problem.py` and `markups.py`.
"""

import numpy as np
import pandas as pd


def make_tiny_data(T: int = 20, J: int = 6, seed: int = 42) -> pd.DataFrame:
    """Generate a minimal product-level dataset.

    Parameters
    ----------
    T : int
        Number of markets.
    J : int
        Number of products per market.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: market_ids, firm_ids, shares, prices, x1, x2, z1, z2, z3,
        clustering_ids, log_shares. Shares sum to < 1 within each market.
    """
    rng = np.random.default_rng(seed)
    N = T * J

    market_ids = np.repeat(np.arange(T), J)
    firm_ids = np.tile(np.arange(J) % 3, T)  # 3 firms per market

    # share construction: draw, then rescale so within-market sum is 0.8
    raw_shares = rng.uniform(0.05, 0.2, N)
    shares = np.empty(N)
    for t in range(T):
        mask = market_ids == t
        shares[mask] = raw_shares[mask] / raw_shares[mask].sum() * 0.8

    prices = rng.uniform(1.0, 5.0, N)
    x1 = rng.normal(0.0, 1.0, N)
    x2 = rng.normal(0.0, 1.0, N)
    z1 = rng.normal(0.0, 1.0, N)
    z2 = rng.normal(0.0, 1.0, N)
    z3 = rng.normal(0.0, 1.0, N)
    clustering_ids = rng.integers(0, T // 2, N)  # 10 clusters

    return pd.DataFrame({
        'market_ids': market_ids,
        'firm_ids': firm_ids,
        'shares': shares,
        'prices': prices,
        'x1': x1,
        'x2': x2,
        'z1': z1,
        'z2': z2,
        'z3': z3,
        'clustering_ids': clustering_ids,
        'log_shares': np.log(shares),
    })


def attach_user_supplied_markups(
        data: pd.DataFrame, column_name: str, multiplier: float) -> pd.DataFrame:
    """Attach a user-supplied markup column to `data`, derived from prices.

    Used to build two-model tests without requiring a pyblp demand solve.
    """
    out = data.copy()
    out[column_name] = -multiplier * out['prices'].to_numpy()
    return out
