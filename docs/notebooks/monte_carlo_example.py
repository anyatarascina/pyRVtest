# Monte Carlo Example
# Demonstrates the v0.4 class-based ConductModel API together with
# new features: Dearing simple-markup models, UserSuppliedMarkups,
# pyRVtest.instruments helpers, passthrough diagnostics, and the
# analytical (demand_params) code path.

import pyblp
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import pyRVtest
from pyRVtest.instruments import blp_instruments, differentiation_ivs
import pandas as pd

pyblp.options.digits = 2
pyblp.options.verbose = False

# %% DGP

rng = np.random.default_rng(seed=0)

# Market structure
F = 5           # number of firms (constant across markets)
J_min = 20      # minimum products per market
J_max = 40      # maximum products per market
target_obs = 10000
T = target_obs // ((J_min + J_max) // 2)
J_per_market = rng.integers(J_min, J_max + 1, size=T)
market_ids = np.repeat(np.arange(T), J_per_market)
firm_ids = np.concatenate([np.arange(J_t) % F for J_t in J_per_market])
id_data = pd.DataFrame({'market_ids': market_ids, 'firm_ids': firm_ids})
integration = pyblp.Integration('product', 9)
X1 = pyblp.Formulation('1 + prices + x1 + x2')
X2 = pyblp.Formulation('0 + x1')
X3 = pyblp.Formulation('1  + z + shares')

# True DGP: Bertrand competition with cost formulation 1 + z + shares (non-constant marginal cost).
# gamma = [1, 4, 0.5]: intercept, z coefficient, shares coefficient.
simulation = pyblp.Simulation(
    product_formulations=(X1, X2, X3),
    beta=[0, -2, 1, 1],
    sigma=1,
    xi_variance=0.2,
    omega_variance=0.2,
    correlation=0,
    gamma=[1, 4, 0.5],
    product_data=id_data,
    integration=integration,
    seed=0
)
simulation_results = simulation.replace_endogenous(constant_costs=False)
data = pd.DataFrame(pyblp.data_to_dict(simulation_results.product_data))

# %% Instruments (using pyRVtest.instruments helpers instead of raw pyblp)

# Demand instruments: local differentiation IVs on x1, x2, z
data['demand_instruments0'] = data['z']
count=0
for char in ['x1', 'x2', 'z']:
    divs = differentiation_ivs(data, char)
    count=count+1
    data[f'demand_instruments{count}'] = divs['sum_squared_diff_rival']
    count=count+1
    data[f'demand_instruments{count}'] = divs['sum_squared_diff_same_firm']


# Test instruments (set 0): BLP instruments on z, x1, x2
blp_ivs = blp_instruments(data, columns=['z', 'x1', 'x2'])
for key, arr in blp_ivs.items():
    data[f'test_iv0_{key}'] = arr

# Test instruments (set 1): local differentiation instruments on z, x1, x2
for char in ['z', 'x1', 'x2']:
    divs = differentiation_ivs(data, char)
    data[f'test_iv1_{char}_rival'] = divs['sum_squared_diff_rival']
    data[f'test_iv1_{char}_same'] = divs['sum_squared_diff_same_firm']

instr_form0 = '+'.join(c for c in data.columns if c.startswith('test_iv0_'))
instr_form1 = '+'.join(c for c in data.columns if c.startswith('test_iv1_'))

data['clustering_ids'] = data['market_ids']

# Mix flag: firms 0, 1, 2 compete in prices (Bertrand); firms 3, 4 compete in quantities (Cournot)
data['mix_flag'] = (data['firm_ids'] < 3)

# Cost scaling (lambda=0.5) for rule-of-thumb and scaled-cost models
data['cost_scaling_col'] = 0.5

# Vertical integration indicator: products of firm 0 are vertically integrated
data['vi_id'] = (data['firm_ids'] == 0).astype(int)

# Partial collusion: firms put weight 0.5 on own-portfolio profits
def kappa_partial_collusion(f, g):
    return 1.0 if f == g else 0.5

# Non-profit conduct: welfare weight lambda=0.7 implies kappa = 1/lambda for own-firm product pairs
def kappa_nonprofit(f, g):
    return 1 / 0.7 if f == g else 0.0

# %% Demand estimation
problem = pyblp.Problem(
    (X1, X2),
    product_data=data, integration=integration
)
pyblp_results = problem.solve(sigma=0.5, method='1s')
print(pyblp_results)

# %% Model library — v0.4 class-based API

# Core 13-model set (analogous to the original v0.3 library): used for the
# main RV conduct test where numerical stability of the MCS is well-established.
model_core = [
    # --- Standard models ---
    pyRVtest.Bertrand(ownership='firm_ids'),
    pyRVtest.Cournot(ownership='firm_ids'),
    pyRVtest.Monopoly(),
    pyRVtest.PerfectCompetition(),

    # --- Profit-weight models (partial collusion, kappa=0.5) ---
    pyRVtest.PartialCollusion(ownership='firm_ids',
                              kappa_specification=kappa_partial_collusion),
    pyRVtest.Cournot(ownership='firm_ids',
                     kappa_specification=kappa_partial_collusion),

    # --- Non-profit conduct (Bertrand, welfare weight lambda=0.7) ---
    pyRVtest.Bertrand(ownership='firm_ids',
                      kappa_specification=kappa_nonprofit),

    # --- Rule-of-thumb (lambda=0.5) and Bertrand with scaled costs ---
    pyRVtest.Bertrand(ownership='firm_ids', cost_scaling='cost_scaling_col'),

    # --- Vertical: linear pricing (Bertrand downstream + Bertrand upstream) ---
    pyRVtest.Vertical(
        downstream=pyRVtest.Bertrand(ownership='firm_ids'),
        upstream=pyRVtest.Bertrand(ownership='firm_ids'),
    ),

    # --- Vertical with partial VI (firm 0 products are vertically integrated) ---
    pyRVtest.Vertical(
        downstream=pyRVtest.Bertrand(ownership='firm_ids'),
        upstream=pyRVtest.Bertrand(ownership='firm_ids'),
        vertical_integration='vi_id',
    ),

    # --- Mixed Cournot-Bertrand ---
    pyRVtest.MixCournotBertrand(mix_flag='mix_flag', ownership='firm_ids'),

    # --- Custom model: single-product Bertrand (each product ignores cross-portfolio effects) ---
    # markup_j = -s_j / (ds_j/dp_j), using only own-price derivatives
    pyRVtest.CustomConductModel(
        markup_fn=lambda O, D, s: -s / np.diag(D).reshape(-1, 1),
        ownership='firm_ids',
        name='single_product_bertrand',
    ),
]

# Dearing et al. (2026) simple-markup models added in v0.4 step 12.
# Shown separately in the markup table and in their own testing problem.
dearing_models = [
    pyRVtest.RuleOfThumb(phi=2.0),        # phi=2: price = 2 * mc (50%-of-price markup)
    pyRVtest.RuleOfThumb(phi=3.0),        # 200%-over-cost markup
    pyRVtest.ConstantMarkup(markup=0.3),  # fixed dollar markup (Example 7)
]

# Full combined library — used for the markup comparison table only.
model_library = model_core + dearing_models

core_names = [
    'Bertrand',
    'Cournot',
    'Monopoly',
    'Perfect competition',
    'Bertrand partial collusion (kappa=0.5)',
    'Cournot partial collusion (kappa=0.5)',
    'Non-profit Bertrand (lambda=0.7)',
    'Bertrand scaled costs (lambda=0.5)',
    'Vertical Bertrand+Bertrand',
    'Vertical Bertrand+Bertrand with VI',
    'Mixed Cournot-Bertrand',
    'Single-product Bertrand (custom)',
]
dearing_names = [
    'Rule-of-thumb (phi=2.0)',
    'Rule-of-thumb (phi=3.0)',
    'Constant markup (zeta=0.3)',
]
model_names = core_names + dearing_names

# %% Standalone markup construction
markups, markups_down, markups_up = pyRVtest.build_markups(model_library, data, pyblp_results)

rows = []
for name, mu in zip(model_names, markups):
    diff = mu.flatten()
    rows.append({
        'Model': name,
        'Mean': diff.mean(),
        'Median': np.median(diff),
        'Std': diff.std(),
        'Min': diff.min(),
        'Max': diff.max(),
    })

markup_table = pd.DataFrame(rows).set_index('Model')
print("\nMarkup distribution:")
print(markup_table.to_string(float_format='{:.4f}'.format))

# %% Conduct test (core 13-model library, two instrument sets, endogenous cost component)

testing_problem = pyRVtest.Problem(
    cost_formulation=pyRVtest.Formulation('1+z+shares'),
    instrument_formulation=[
        pyRVtest.Formulation('0+x1+x2+' + instr_form0),
        pyRVtest.Formulation('0+x1+x2+' + instr_form1),
    ],
    models=model_core,
    product_data=data,
    demand_results=pyblp_results,
    endogenous_cost_component='shares'
)

testing_results = testing_problem.solve(
    demand_adjustment=True,
    clustering_adjustment=True,
)
print(testing_results)

# %% UserSuppliedMarkups sanity check
# Pre-computed Bertrand markups fed back in as a column; demand_adjustment must be False
# because there is no demand system to perturb. TRV vs. Bertrand should be ≈ 0.
data['mkup_bertrand'] = markups[0].flatten()
usm_problem = pyRVtest.Problem(
    cost_formulation=pyRVtest.Formulation('1+z+shares'),
    instrument_formulation=pyRVtest.Formulation('0+x1+x2+' + instr_form0),
    models=[
        pyRVtest.Bertrand(ownership='firm_ids'),
        pyRVtest.UserSuppliedMarkups(markups='mkup_bertrand', ownership='firm_ids'),
    ],
    product_data=data,
    demand_results=None,
    endogenous_cost_component='shares'
)
usm_results = usm_problem.solve(demand_adjustment=False, clustering_adjustment=True)
print("\nUserSuppliedMarkups sanity check (TRV vs. Bertrand should be ≈ 0):")
print(usm_results)

# %% Dearing et al. (2026) simple-markup models — focused test against Bertrand
# Tests RuleOfThumb and ConstantMarkup against multi-product Bertrand;
# under the true DGP (Bertrand), all three should be rejected.
dearing_problem = pyRVtest.Problem(
    cost_formulation=pyRVtest.Formulation('1+z+shares'),
    instrument_formulation=pyRVtest.Formulation('0+x1+x2+' + instr_form0),
    models=[pyRVtest.Bertrand(ownership='firm_ids')] + dearing_models,
    product_data=data,
    demand_results=pyblp_results,
    endogenous_cost_component='shares'
)
dearing_results = dearing_problem.solve(demand_adjustment=True, clustering_adjustment=True)
print("\nDearing simple-markup models vs. Bertrand:")
print(dearing_results)

# %% Analytical (demand_params) code path
# demand_params supports plain logit and nested logit only (not BLP random coefficients).
# We estimate a simple logit (no X2 random coefficient) on the same data so that the
# analytical Jacobian path is valid.  The result should still clearly reject
# PerfectCompetition in favour of Bertrand under the true Bertrand DGP.

logit_problem = pyblp.Problem(
    pyblp.Formulation('1 + prices + x1 + x2'),
    product_data=data,
)
logit_results = logit_problem.solve(method='1s')
logit_alpha = float(logit_results.beta.flatten()[logit_results.beta_labels.index('prices')])
logit_beta  = logit_results.beta.flatten().tolist()
demand_iv_cols = [c for c in data.columns if c.startswith('demand_instruments')]
data['const']=1
analytical_problem = pyRVtest.Problem(
    cost_formulation=pyRVtest.Formulation('1+z+shares'),
    instrument_formulation=[
        pyRVtest.Formulation('0+x1+x2+' + instr_form0),
        pyRVtest.Formulation('0+x1+x2+' + instr_form1),
    ],
    models=[pyRVtest.Bertrand(ownership='firm_ids'), pyRVtest.PerfectCompetition()],
    product_data=data,
    demand_params={
        'x_columns':['const','x1','x2'],
        'alpha': logit_alpha,
        'beta': [logit_beta[i] for i in [0,2,3]],
        'rho': [],                           # plain logit (no nesting)
        'demand_instrument_columns': demand_iv_cols,
    },
    endogenous_cost_component='shares'
)
analytical_results = analytical_problem.solve(
    demand_adjustment=True,
    clustering_adjustment=True,
)
print("\nAnalytical demand_params path (Bertrand vs. PerfectCompetition):")
print(analytical_results)

# %% Passthrough diagnostics — Dearing et al. (2026) Remark 4
# Only Vertical models support passthrough_matrix / passthrough_comparison in v0.4.
vertical_indices = [
    i for i, m in enumerate(model_library)
    if isinstance(m, pyRVtest.Vertical)
]

vertical_problem = pyRVtest.Problem(
    cost_formulation=pyRVtest.Formulation('1+z+shares'),
    instrument_formulation=pyRVtest.Formulation('0+x1+x2+' + instr_form0),
    models=[model_library[i] for i in vertical_indices],
    product_data=data,
    demand_results=pyblp_results,
    endogenous_cost_component='shares'
)
vertical_results = vertical_problem.solve(
    demand_adjustment=True,
    clustering_adjustment=True,
)

# Pairwise off-diagonal Frobenius distance between pass-through matrices.
# Near-zero ⟹ hard to separate the pair with pass-through instruments.
pt_comparison = vertical_results.passthrough_comparison(metric='offdiag_frobenius')
print("\nPassthrough comparison (Dearing Remark 4, off-diagonal Frobenius):")
print(pt_comparison.groupby(['model_i_label', 'model_j_label'])['distance'].mean().to_string())

# %% Labor-side example (experimental, v0.4 step 14)
# Synthetic wage panel: 2 firms, 50 markets.
# Tests Monopsony vs. PerfectCompetition on the supply side.
rng_labor = np.random.default_rng(seed=42)
T_labor = 50
labor_data = pd.DataFrame({
    'market_ids':       np.repeat(np.arange(T_labor), 2),
    'firm_ids':         np.tile([0, 1], T_labor),
    'wages':            np.clip(rng_labor.normal(12.0, 1.5, T_labor * 2), 5.0, None),
    'employment_share': np.clip(rng_labor.uniform(0.10, 0.45, T_labor * 2), 0.01, 0.49),
    'cost_shifter':     rng_labor.uniform(0.5, 1.5, T_labor * 2),
    'iv0':              rng_labor.uniform(0.8, 1.2, T_labor * 2),
    # Pre-supplied markdowns: Monopsony uses column 'md_monopsony', PC uses zero
    'md_monopsony':     rng_labor.uniform(0.05, 0.30, T_labor * 2),
    'md_pc':            np.zeros(T_labor * 2),
})
labor_data['clustering_ids'] = labor_data['market_ids']

labor_problem = pyRVtest.Problem(
    cost_formulation=pyRVtest.Formulation('1 + cost_shifter'),
    instrument_formulation=pyRVtest.Formulation('0 + iv0'),
    product_data=labor_data,
    models=[
        pyRVtest.UserSuppliedMarkups(markups='md_monopsony', ownership='firm_ids'),
        pyRVtest.UserSuppliedMarkups(markups='md_pc',        ownership='firm_ids'),
    ],
    market_side='labor',
)
labor_results = labor_problem.solve(clustering_adjustment=True)
print("\nLabor-side conduct test (Monopsony vs. PerfectCompetition):")
print(labor_results)

# %% Save results
results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'monte_carlo_results.txt')
with open(results_path, 'w') as f:
    f.write(str(testing_results))
print(f"\nResults written to {results_path}")
