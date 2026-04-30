#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 13:02:32 2026

@author: md
"""
import os
import pyblp
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import pyRVtest
import pandas as pd


# %% CarRV
product_data=pd.read_csv('/home/md/Dropbox/Projects/CarRV/Data/product_dataS.csv',
                         low_memory=False)
demand=pyblp.read_pickle('/home/md/Dropbox/Projects/CarRV/Data/demand.p')

# Markups
product_data['asian_B']=(product_data.regionUS==0) & (product_data.regionEU==0)
product_data['us_B']=(product_data.regionUS==1)
product_data['eu_C']=product_data.regionEU==0
models = [
    pyRVtest.Bertrand(ownership='firm_ids'),
    pyRVtest.Cournot(ownership='firm_ids'),
    pyRVtest.MixCournotBertrand(mix_flag='asian_B', ownership='firm_ids'),
    pyRVtest.MixCournotBertrand(mix_flag='us_B', ownership='firm_ids'),
    pyRVtest.MixCournotBertrand(mix_flag='eu_C', ownership='firm_ids')
]
markups, markups_down, markups_up = pyRVtest.build_markups(models, product_data, demand)
product_data['mkup_bertrand']=markups[0]
product_data['mkup_cournot']=markups[1]
product_data['mkup_BC_asian']=markups[2]
product_data['mkup_BC_us']=markups[3]
product_data['mkup_BC_euC']=markups[4]

# Test
df=product_data.loc[product_data['test_sample'],:].copy().reset_index()
#Take logs
df['log_total_production2'] =np.log(df['total.production2'])
#Standardize
std_vars=['n_oppo','zage','zhhincome','rxr_oppo']
for var in std_vars:
    df[f'{var}_z'] = (df[f'{var}'] - df[f'{var}'].mean()) / df[f'{var}'].std()
  

W=['1','log_total_production2','rxr','t','t2','log_height','log_footprint',
           'log_hp','log_mpg','log_curbweight','log_number_trims',
           'releaseYear','suv', 'truck','van','PHEV_EV','sport','yearsSinceDesign','make2']
Z=['0', 'n_oppo_z','zage_z','zhhincome_z', 't_suv','rxr_oppo_z']

def build_problem():
    return pyRVtest.Problem(
        cost_formulation=pyRVtest.Formulation('+'.join(W)),
        instrument_formulation=pyRVtest.Formulation('+'.join(Z)),
        models=[
            pyRVtest.UserSuppliedMarkups(markups='mkup_bertrand', ownership='firm_ids'),
            pyRVtest.UserSuppliedMarkups(markups='mkup_cournot', ownership='firm_ids'),
            pyRVtest.UserSuppliedMarkups(markups='mkup_BC_asian', ownership='firm_ids'),
            pyRVtest.UserSuppliedMarkups(markups='mkup_BC_us', ownership='firm_ids'),
            pyRVtest.UserSuppliedMarkups(markups='mkup_BC_euC', ownership='firm_ids'),
            ],
        product_data=df,
        endogenous_cost_component='log_total_production2',
        demand_results=None,
    )

# (1) With Appendix B influence-function correction (current methodology)
testing_problem = build_problem()
testing_results = testing_problem.solve(
    costs_type='log',
    demand_adjustment=False,
    clustering_adjustment=False,
)
print('=== With Appendix B correction (default) ===')
print(testing_results)

# (2) Without Appendix B correction (replicates conduct_test.py legacy SEs)
testing_problem_legacy = build_problem()
testing_problem_legacy._skip_appendix_b = True
testing_results_legacy = testing_problem_legacy.solve(
    costs_type='log',
    demand_adjustment=False,
    clustering_adjustment=False,
)
print('=== Without Appendix B correction (_skip_appendix_b=True) ===')
print(testing_results_legacy)