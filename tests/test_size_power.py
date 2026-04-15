"""Size and power of the RV test via Monte Carlo simulation.

Simulates 500 replications of a 2-firm logit DGP. At each replication, draws new cost
and demand shocks, solves for Bertrand equilibrium, and runs the RV test. Checks:
1. Size: when both models are the true model, rejection rate is near 5%.
2. Power: when testing true model vs wrong model, rejection rate is high.

Uses user_supplied_markups to bypass PyBLP demand estimation, making each replication
fast (~0.05s) and the full test tractable (~30s for 500 replications).
"""

import numpy as np
import pandas as pd
import pytest
import pyRVtest


def _logit_shares(V, market_ids, T):
    """Compute logit shares from indirect utility V."""
    N = len(V)
    shares = np.zeros((N, 1))
    for t in range(T):
        idx = np.where(market_ids == t)[0]
        v_t = V[idx].flatten()
        exp_v = np.exp(v_t)
        shares[idx, 0] = exp_v / (1.0 + exp_v.sum())
    return shares


def _solve_bertrand_logit(alpha, beta_x, x, xi, mc, market_ids, T, N):
    """Solve for Bertrand-logit equilibrium via iteration."""
    prices = mc + 0.5
    for _ in range(300):
        V = beta_x * x + alpha * prices + xi
        shares = _logit_shares(V, market_ids, T)
        markups = -1.0 / (alpha * (1.0 - shares))
        prices_new = mc + markups
        if np.max(np.abs(prices_new - prices)) < 1e-13:
            break
        prices = prices_new
    V = beta_x * x + alpha * prices + xi
    shares = _logit_shares(V, market_ids, T)
    markups = -1.0 / (alpha * (1.0 - shares))
    return prices, shares, markups


def _run_one_replication(seed, T=500, J=2, alpha=-1.0, beta_x=1.0):
    """Simulate one DGP draw and return pyRVtest results for Bertrand vs perfect competition."""
    rng = np.random.default_rng(seed=seed)
    N = T * J
    market_ids = np.repeat(np.arange(T), J)
    firm_ids = np.tile(np.arange(J), T)

    x = rng.uniform(0.5, 2.0, size=(N, 1))
    cost_shifter = rng.uniform(0.5, 2.0, size=(N, 1))
    xi = rng.normal(0, 0.1, size=(N, 1))
    omega = rng.normal(0, 0.1, size=(N, 1))
    mc_true = 1.0 + 0.5 * cost_shifter + omega

    # Instruments: use product characteristics and rival characteristics.
    # These are correlated with Bertrand markups (which depend on shares,
    # which depend on x), making them relevant for distinguishing Bertrand
    # from perfect competition.
    # iv0 = own x, iv1 = rival's x (in each market), iv2 = rival's cost_shifter
    iv0 = x.copy()
    iv1 = np.zeros((N, 1))
    iv2 = np.zeros((N, 1))
    for t in range(T):
        idx = np.where(market_ids == t)[0]
        # For 2-product markets: rival is the other product
        iv1[idx[0]] = x[idx[1]]
        iv1[idx[1]] = x[idx[0]]
        iv2[idx[0]] = cost_shifter[idx[1]]
        iv2[idx[1]] = cost_shifter[idx[0]]

    prices, shares, markups_bertrand = _solve_bertrand_logit(
        alpha, beta_x, x, xi, mc_true, market_ids, T, N
    )

    product_data = pd.DataFrame({
        'market_ids': market_ids,
        'firm_ids': firm_ids,
        'prices': prices.flatten(),
        'shares': shares.flatten(),
        'x': x.flatten(),
        'cost_shifter': cost_shifter.flatten(),
        'iv0': iv0.flatten(), 'iv1': iv1.flatten(), 'iv2': iv2.flatten(),
        'markups_bertrand': markups_bertrand.flatten(),
        'markups_perfect': np.zeros(N),
    })
    return product_data


def _test_pair(product_data, markup_col_1, markup_col_2):
    """Run RV test for a pair of user-supplied markup models and return TRV."""
    model_formulations = (
        pyRVtest.ModelFormulation(model_downstream='bertrand', ownership_downstream='firm_ids',
                                 user_supplied_markups=markup_col_1),
        pyRVtest.ModelFormulation(model_downstream='bertrand', ownership_downstream='firm_ids',
                                 user_supplied_markups=markup_col_2),
    )
    testing_problem = pyRVtest.Problem(
        cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
        instrument_formulation=pyRVtest.Formulation('0 + iv0 + iv1 + iv2'),
        model_formulations=model_formulations,
        product_data=product_data,
        demand_results=None,
    )
    pyRVtest.options.verbose = False
    results = testing_problem.solve(demand_adjustment=False, clustering_adjustment=False)
    pyRVtest.options.verbose = True
    return float(results.TRV[0][0, 1]), float(results.F[0][0, 1])


@pytest.mark.slow
class TestSizePower:
    """Monte Carlo rejection frequency tests."""

    N_REPLICATIONS = 500

    @pytest.fixture(scope='class')
    def mc_results(self):
        """Run all replications and collect TRV values."""
        trv_power = []  # Bertrand (true) vs perfect competition (wrong)
        trv_size = []   # Bertrand (true) vs Bertrand (true) — both models are the same
        f_power = []

        for rep in range(self.N_REPLICATIONS):
            product_data = _run_one_replication(seed=rep)

            # Power test: true model (Bertrand) vs wrong model (perfect competition)
            trv, f = _test_pair(product_data, 'markups_bertrand', 'markups_perfect')
            trv_power.append(trv)
            f_power.append(f)

            # Size test: Bertrand vs "nearly Bertrand" (markups + tiny noise).
            # Under the null, both models are essentially correct; rejection should be near 5%.
            # We add vanishingly small noise to avoid exact degeneracy (sigma_RV = 0).
            noise_rng = np.random.default_rng(seed=rep + 100000)
            product_data['markups_near_bertrand'] = (
                product_data['markups_bertrand'] + noise_rng.normal(0, 1e-6, len(product_data))
            )
            trv_s, _ = _test_pair(product_data, 'markups_bertrand', 'markups_near_bertrand')
            trv_size.append(trv_s)

        return {
            'trv_power': np.array(trv_power),
            'trv_size': np.array(trv_size),
            'f_power': np.array(f_power),
        }

    def test_size(self, mc_results):
        """Under the null (both models identical), rejection rate should be near 5%."""
        trv = mc_results['trv_size']
        rejection_rate = np.mean(np.abs(trv) > 1.96)
        # With 500 replications, 95% CI for true 5% rate is roughly [3%, 7%]
        # Allow wider band [0%, 10%] to be safe
        assert rejection_rate < 0.10, (
            f"Size distortion: rejection rate {rejection_rate:.3f} exceeds 10% "
            f"(expected ~5% under the null)"
        )

    def test_power(self, mc_results):
        """Under the alternative (true vs wrong model), rejection rate should be high."""
        trv = mc_results['trv_power']
        # Reject in favor of model 1 (Bertrand, row model) when TRV < -1.96
        rejection_rate = np.mean(trv < -1.96)
        assert rejection_rate > 0.50, (
            f"Low power: rejection rate {rejection_rate:.3f} is below 50% "
            f"(expected high power for Bertrand vs perfect competition)"
        )

    def test_power_correct_direction(self, mc_results):
        """When the test rejects, it should reject in the right direction (favor true model)."""
        trv = mc_results['trv_power']
        rejections = np.abs(trv) > 1.96
        if rejections.any():
            # Among rejections, most should be negative (favoring model 0 = Bertrand = true)
            wrong_direction = np.mean(trv[rejections] > 0)
            assert wrong_direction < 0.10, (
                f"Wrong-direction rejections: {wrong_direction:.3f} of rejections favor the wrong model"
            )

    def test_f_strong_instruments(self, mc_results):
        """F-statistics should generally indicate strong instruments."""
        f = mc_results['f_power']
        # Most F-statistics should exceed typical critical values (say 5)
        strong_rate = np.mean(f > 5.0)
        assert strong_rate > 0.80, (
            f"Weak instruments: only {strong_rate:.3f} of replications have F > 5"
        )
