"""Integration tests for demand_params across all model types and configurations.

Constructs a nested logit DGP and runs solve() through every supported model,
instrument configuration, and demand adjustment setting. Tests that:
1. Every model type produces markups and TRV without crashing
2. Nested logit (sigma > 0) works end-to-end
3. Vertical models work with the analytical Hessian
4. Multiple instrument sets work
5. Demand adjustment produces different (wider) standard errors
6. Markups have correct economic properties per model type
"""

import numpy as np
import pandas as pd
import pytest
import pyRVtest
from pyRVtest.demand_jacobian import _logit_jacobian, _nested_logit_jacobian


# ---------------------------------------------------------------------------
# Shared DGP: nested logit with 2 nests, firm structure for vertical/mixed
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def nested_logit_data():
    """Build a nested logit DGP with sigma=0.4, suitable for all model types."""
    rng = np.random.default_rng(seed=777)
    T = 40
    J = 4  # 4 products per market (2 firms × 2 products each)
    N = T * J
    alpha = -1.5
    sigma = 0.4

    market_ids = np.repeat(np.arange(T), J)
    # Firm 0 owns products 0,1; Firm 1 owns products 2,3
    firm_ids = np.tile([0, 0, 1, 1], T)
    # Nests: products 0,2 in nest A; products 1,3 in nest B
    nesting_ids = np.tile(['A', 'B', 'A', 'B'], T)

    x = rng.uniform(0.5, 2.0, size=N)
    cost_shifter = rng.uniform(0.5, 2.0, size=N)
    xi = rng.normal(0, 0.1, size=N)
    omega = rng.normal(0, 0.05, size=N)
    mc = 1.0 + 0.3 * cost_shifter + omega

    # Solve nested logit Bertrand equilibrium
    prices = mc + 0.5
    for _ in range(500):
        V = x + alpha * prices + xi
        shares = np.zeros(N)
        for t in range(T):
            idx = market_ids == t
            v = V[idx]
            exp_v_sig = np.exp(v / (1 - sigma))
            # Within-nest denominators
            nest_ids_t = nesting_ids[idx]
            denom_nests = {}
            for g in np.unique(nest_ids_t):
                mask = nest_ids_t == g
                denom_nests[g] = exp_v_sig[mask].sum()
            # Shares
            s_t = np.zeros(J)
            outside_sum = 1.0
            for g in np.unique(nest_ids_t):
                mask = nest_ids_t == g
                D_g = denom_nests[g]
                outside_sum += D_g ** (1 - sigma)
            for j_pos in range(J):
                g = nest_ids_t[j_pos]
                D_g = denom_nests[g]
                s_t[j_pos] = (exp_v_sig[j_pos] / D_g) * (D_g ** (1 - sigma)) / outside_sum
            shares[idx] = s_t

        # Nested logit Bertrand markups
        D_all = np.zeros((N, J))  # will be sliced per market
        markups_b = np.zeros(N)
        for t in range(T):
            idx_arr = np.where(market_ids == t)[0]
            s_t = shares[idx_arr]
            nest_t = [nesting_ids[idx_arr]]
            D_t = _nested_logit_jacobian(alpha, [sigma], s_t, nest_t)
            O_t = np.zeros((J, J))
            fids_t = firm_ids[idx_arr]
            for j in range(J):
                for k in range(J):
                    O_t[j, k] = 1.0 if fids_t[j] == fids_t[k] else 0.0
            markups_b[idx_arr] = np.linalg.solve(O_t * D_t.T, -s_t)

        prices_new = mc + markups_b
        if np.max(np.abs(prices_new - prices)) < 1e-12:
            break
        prices = prices_new

    # Build instruments: rival x, rival cost_shifter, own x
    iv1 = np.zeros(N)
    iv2 = np.zeros(N)
    iv3 = np.zeros(N)
    iv4 = x.copy()
    for t in range(T):
        idx = np.where(market_ids == t)[0]
        for j_idx in idx:
            rival = [i for i in idx if firm_ids[i] != firm_ids[j_idx]]
            own_other = [i for i in idx if firm_ids[i] == firm_ids[j_idx] and i != j_idx]
            iv1[j_idx] = x[rival].mean() if rival else 0
            iv2[j_idx] = cost_shifter[rival].mean() if rival else 0
            iv3[j_idx] = x[own_other].mean() if own_other else 0

    # Mix flag: firm 0 products are Bertrand, firm 1 products are Cournot
    mix_flag = (firm_ids == 0)

    data = pd.DataFrame({
        'market_ids': market_ids,
        'firm_ids': firm_ids,
        'firm_ids_up': firm_ids,  # same ownership for upstream
        'nesting_ids': nesting_ids,
        'prices': prices,
        'shares': shares,
        'x': x,
        'cost_shifter': cost_shifter,
        'iv1': iv1, 'iv2': iv2, 'iv3': iv3, 'iv4': iv4,
        'clustering_ids': market_ids,
        'mix_flag': mix_flag,
    })

    return data, alpha, sigma, xi


# ---------------------------------------------------------------------------
# Test: every model type runs without crashing
# ---------------------------------------------------------------------------

class TestAllModelTypes:
    """Verify every supported model produces results with demand_params."""

    @pytest.fixture(scope='class')
    def base_args(self, nested_logit_data):
        data, alpha, sigma, _ = nested_logit_data
        pyRVtest.options.verbose = False
        return {
            'cost_formulation': pyRVtest.Formulation('1 + cost_shifter'),
            'instrument_formulation': pyRVtest.Formulation('0 + iv1 + iv2 + iv3'),
            'product_data': data,
            'demand_params': {'alpha': alpha, 'sigma': [sigma]},
        }

    def _solve_with_model(self, base_args, model_formulations):
        p = pyRVtest.Problem(**base_args, model_formulations=model_formulations)
        return p.solve(demand_adjustment=False)

    def test_bertrand(self, base_args):
        r = self._solve_with_model(base_args, (
            pyRVtest.ModelFormulation(model_downstream='bertrand', ownership_downstream='firm_ids'),
            pyRVtest.ModelFormulation(model_downstream='perfect_competition'),
        ))
        assert not np.isnan(r.TRV[0][0, 1])
        # Bertrand markups should be positive
        assert np.all(r.markups[0] > 0)

    def test_cournot(self, base_args):
        r = self._solve_with_model(base_args, (
            pyRVtest.ModelFormulation(model_downstream='cournot', ownership_downstream='firm_ids'),
            pyRVtest.ModelFormulation(model_downstream='perfect_competition'),
        ))
        assert not np.isnan(r.TRV[0][0, 1])
        assert np.all(r.markups[0] > 0)

    def test_monopoly(self, base_args):
        r = self._solve_with_model(base_args, (
            pyRVtest.ModelFormulation(model_downstream='monopoly'),
            pyRVtest.ModelFormulation(model_downstream='bertrand', ownership_downstream='firm_ids'),
        ))
        assert not np.isnan(r.TRV[0][0, 1])
        # Monopoly markups should be larger than Bertrand
        assert r.markups[0].mean() > r.markups[1].mean()

    def test_perfect_competition(self, base_args):
        r = self._solve_with_model(base_args, (
            pyRVtest.ModelFormulation(model_downstream='perfect_competition'),
            pyRVtest.ModelFormulation(model_downstream='bertrand', ownership_downstream='firm_ids'),
        ))
        assert not np.isnan(r.TRV[0][0, 1])
        assert np.allclose(r.markups[0], 0)

    def test_mixed_cournot_bertrand(self, base_args):
        r = self._solve_with_model(base_args, (
            pyRVtest.ModelFormulation(model_downstream='mix_cournot_bertrand',
                                     ownership_downstream='firm_ids', mix_flag='mix_flag'),
            pyRVtest.ModelFormulation(model_downstream='bertrand', ownership_downstream='firm_ids'),
        ))
        assert not np.isnan(r.TRV[0][0, 1])

    def test_vertical_bertrand_bertrand(self, base_args):
        r = self._solve_with_model(base_args, (
            pyRVtest.ModelFormulation(model_downstream='bertrand', ownership_downstream='firm_ids',
                                     model_upstream='bertrand', ownership_upstream='firm_ids_up'),
            pyRVtest.ModelFormulation(model_downstream='bertrand', ownership_downstream='firm_ids'),
        ))
        assert not np.isnan(r.TRV[0][0, 1])
        # Vertical markups should be larger than horizontal (double marginalization)
        assert r.markups[0].mean() > r.markups[1].mean()

    def test_markup_ordering(self, base_args):
        """Monopoly > Cournot > Bertrand > Perfect competition."""
        r = self._solve_with_model(base_args, (
            pyRVtest.ModelFormulation(model_downstream='monopoly'),
            pyRVtest.ModelFormulation(model_downstream='cournot', ownership_downstream='firm_ids'),
            pyRVtest.ModelFormulation(model_downstream='bertrand', ownership_downstream='firm_ids'),
            pyRVtest.ModelFormulation(model_downstream='perfect_competition'),
        ))
        means = [r.markups[m].mean() for m in range(4)]
        assert means[0] > means[1] > means[2] > means[3]


# ---------------------------------------------------------------------------
# Test: nested logit end-to-end
# ---------------------------------------------------------------------------

class TestNestedLogitEndToEnd:
    """Verify nested logit (sigma > 0) produces correct TRV through the full pipeline."""

    def test_nested_logit_trv_finite(self, nested_logit_data):
        data, alpha, sigma, _ = nested_logit_data
        pyRVtest.options.verbose = False
        p = pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
            instrument_formulation=pyRVtest.Formulation('0 + iv1 + iv2 + iv3'),
            model_formulations=(
                pyRVtest.ModelFormulation(model_downstream='bertrand', ownership_downstream='firm_ids'),
                pyRVtest.ModelFormulation(model_downstream='cournot', ownership_downstream='firm_ids'),
            ),
            product_data=data,
            demand_params={'alpha': alpha, 'sigma': [sigma]},
        )
        r = p.solve(demand_adjustment=False)
        assert not np.isnan(r.TRV[0][0, 1])
        assert not np.isnan(r.F[0][0, 1])

    def test_nested_vs_logit_markups_differ(self, nested_logit_data):
        """Nested logit markups should differ from plain logit markups."""
        data, alpha, sigma, _ = nested_logit_data
        pyRVtest.options.verbose = False
        models = (
            pyRVtest.ModelFormulation(model_downstream='bertrand', ownership_downstream='firm_ids'),
            pyRVtest.ModelFormulation(model_downstream='perfect_competition'),
        )
        r_nested = pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
            instrument_formulation=pyRVtest.Formulation('0 + iv1 + iv2'),
            model_formulations=models, product_data=data,
            demand_params={'alpha': alpha, 'sigma': [sigma]},
        ).solve(demand_adjustment=False)

        r_logit = pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
            instrument_formulation=pyRVtest.Formulation('0 + iv1 + iv2'),
            model_formulations=models, product_data=data,
            demand_params={'alpha': alpha, 'sigma': []},
        ).solve(demand_adjustment=False)

        # Markups should differ because the Jacobian differs
        assert not np.allclose(r_nested.markups[0], r_logit.markups[0])


# ---------------------------------------------------------------------------
# Test: multiple instrument sets
# ---------------------------------------------------------------------------

class TestMultipleInstrumentSets:
    """Verify multiple instrument sets produce independent results."""

    def test_two_instrument_sets(self, nested_logit_data):
        data, alpha, sigma, _ = nested_logit_data
        pyRVtest.options.verbose = False
        p = pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
            instrument_formulation=[
                pyRVtest.Formulation('0 + iv1 + iv2'),
                pyRVtest.Formulation('0 + iv3 + iv4'),
            ],
            model_formulations=(
                pyRVtest.ModelFormulation(model_downstream='bertrand', ownership_downstream='firm_ids'),
                pyRVtest.ModelFormulation(model_downstream='perfect_competition'),
            ),
            product_data=data,
            demand_params={'alpha': alpha, 'sigma': [sigma]},
        )
        r = p.solve(demand_adjustment=False)
        # Should have 2 instrument sets
        assert len(r.TRV) == 2
        assert not np.isnan(r.TRV[0][0, 1])
        assert not np.isnan(r.TRV[1][0, 1])
        # TRV should generally differ across instrument sets
        assert r.TRV[0][0, 1] != r.TRV[1][0, 1]


# ---------------------------------------------------------------------------
# Test: demand adjustment with nonzero xi
# ---------------------------------------------------------------------------

class TestDemandAdjustmentNonzeroXi:
    """Verify demand adjustment changes TRV when xi is nonzero."""

    def test_demand_adj_changes_trv(self, nested_logit_data):
        data, alpha, sigma, xi = nested_logit_data
        pyRVtest.options.verbose = False

        # The DGP has nonzero xi built in
        assert np.std(xi) > 0.01, "xi should be nonzero for this test"

        models = (
            pyRVtest.ModelFormulation(model_downstream='bertrand', ownership_downstream='firm_ids'),
            pyRVtest.ModelFormulation(model_downstream='perfect_competition'),
        )
        common = dict(
            cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
            instrument_formulation=pyRVtest.Formulation('0 + iv1 + iv2 + iv3'),
            model_formulations=models,
            product_data=data,
        )

        # Without demand adjustment
        r_no = pyRVtest.Problem(
            **common,
            demand_params={'alpha': alpha, 'sigma': [sigma]},
        ).solve(demand_adjustment=False)

        # With demand adjustment
        # Berry inversion: log(s/s0) = x*beta + alpha*p + sigma*log(s_{j|g}) + xi
        # beta = 1.0 (coefficient on x in the DGP utility V = x + alpha*p + xi)
        r_adj = pyRVtest.Problem(
            **common,
            demand_params={
                'alpha': alpha, 'sigma': [sigma],
                'beta': np.array([1.0]),
                'x_columns': ['x'],
                'demand_instrument_columns': ['iv1', 'iv2', 'iv3'],
            },
        ).solve(demand_adjustment=True)

        # TRV should differ (demand adjustment changes the variance)
        assert not np.isclose(r_no.TRV[0][0, 1], r_adj.TRV[0][0, 1]), (
            f"TRV unchanged by demand adjustment: {r_no.TRV[0][0, 1]} vs {r_adj.TRV[0][0, 1]}"
        )

    def test_demand_adj_vertical_model(self, nested_logit_data):
        """Demand adjustment should work with vertical models (sigma derivative via finite diff)."""
        data, alpha, sigma, _ = nested_logit_data
        pyRVtest.options.verbose = False
        models = (
            pyRVtest.ModelFormulation(model_downstream='bertrand', ownership_downstream='firm_ids',
                                     model_upstream='bertrand', ownership_upstream='firm_ids_up'),
            pyRVtest.ModelFormulation(model_downstream='bertrand', ownership_downstream='firm_ids'),
        )
        r = pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
            instrument_formulation=pyRVtest.Formulation('0 + iv1 + iv2 + iv3'),
            model_formulations=models, product_data=data,
            demand_params={
                'alpha': alpha, 'sigma': [sigma],
                'beta': np.array([1.0]),
                'x_columns': ['x'],
                'demand_instrument_columns': ['iv1', 'iv2', 'iv3'],
            },
        ).solve(demand_adjustment=True)
        assert not np.isnan(r.TRV[0][0, 1])

    def test_demand_adj_custom_model(self, nested_logit_data):
        """Demand adjustment with custom model uses finite differences for all derivatives."""
        data, alpha, sigma, _ = nested_logit_data
        pyRVtest.options.verbose = False
        models = (
            pyRVtest.ModelFormulation(
                model_downstream='other', ownership_downstream='firm_ids',
                custom_model_specification={
                    'rule_of_thumb': lambda O, D, s: 0.5 * s / np.diag(D).reshape(-1, 1)
                }
            ),
            pyRVtest.ModelFormulation(model_downstream='bertrand', ownership_downstream='firm_ids'),
        )
        r = pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
            instrument_formulation=pyRVtest.Formulation('0 + iv1 + iv2 + iv3'),
            model_formulations=models, product_data=data,
            demand_params={
                'alpha': alpha, 'sigma': [sigma],
                'beta': np.array([1.0]),
                'x_columns': ['x'],
                'demand_instrument_columns': ['iv1', 'iv2', 'iv3'],
            },
        ).solve(demand_adjustment=True)
        assert not np.isnan(r.TRV[0][0, 1])

    def test_demand_adj_with_clustering(self, nested_logit_data):
        """Demand adjustment + clustering should both work together."""
        data, alpha, sigma, _ = nested_logit_data
        pyRVtest.options.verbose = False
        p = pyRVtest.Problem(
            cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
            instrument_formulation=pyRVtest.Formulation('0 + iv1 + iv2 + iv3'),
            model_formulations=(
                pyRVtest.ModelFormulation(model_downstream='bertrand', ownership_downstream='firm_ids'),
                pyRVtest.ModelFormulation(model_downstream='perfect_competition'),
            ),
            product_data=data,
            demand_params={
                'alpha': alpha, 'sigma': [sigma],
                'beta': np.array([1.0]),
                'x_columns': ['x'],
                'demand_instrument_columns': ['iv1', 'iv2', 'iv3'],
            },
        )
        r = p.solve(demand_adjustment=True, clustering_adjustment=True)
        assert not np.isnan(r.TRV[0][0, 1])
