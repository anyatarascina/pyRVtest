#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 15:53:15 2026

@author: md
"""
import os
import pyblp
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import pyRVtest as pyRV
import pickle
from sklearn.decomposition import PCA

dpath=lambda x: os.path.join('/home/md/Dropbox/Projects/RVpaper/RVpaper_complete/Data/',x)

# %% Functions
def Instr(product_data,instr):
    # Remove old instruments
    col=product_data.columns
    old_instr=[i for i,item in enumerate(col) if item.startswith("demand_instruments")]
    n=len(old_instr)
    for ii in range(n):
        del product_data[f'demand_instruments{ii}']
    
    #Set new instruments
    col=product_data.columns
    if sum(col.isin(instr))==len(instr):
        Z=product_data.loc[:,col.isin(instr)]
        Z.columns=[f'demand_instruments{ii}' for ii in range(Z.shape[1])]
        product_data=pd.concat([product_data,Z],axis=1)
    else:
        raise Exception("Can't find all instruments in data frame")
    return(product_data)

def Instruments(product_data,spec):
    diff_instr = pyblp.build_differentiation_instruments(
        pyblp.Formulation(f'0+{spec}'),product_data,version='local')
    for i, column in enumerate(diff_instr.T):
        product_data[f'diff_instr{i}'] = column
        
    blp_instr = pyblp.build_blp_instruments(
        pyblp.Formulation(f'1+{spec}'),product_data)
    for i, column in enumerate(blp_instr.T):
        product_data[f'blp_instr{i}'] = column
    
    return(product_data)

def PCAInstr(Z_names,n_comp,prefix,product_data):
    N = len(product_data)
    Z=product_data[Z_names]
    pca = PCA(n_components=n_comp,random_state=1234)
    Z_pca=pca.fit_transform(Z)
    #print(pca.explained_variance_ratio_)
    for ii in range(n_comp):
        tmp=Z_pca[:,ii]
        tmp=tmp.reshape(N,1)
        product_data[f"{prefix}{ii}_pca"]=tmp
    return(product_data)
def CostInstr(product_data):
    N = np.size(product_data.prices)
    mkts = np.unique(product_data.market_ids)
    Z_cost = np.zeros((N,1))
    ownership = pyblp.build_ownership(product_data)
    for mm in mkts:
        ind_mm = np.where(product_data.market_ids == mm)[0]
        tmp = product_data.freight_cost[ind_mm]
        J = np.shape(tmp)[0]
        tmp = tmp.values.reshape(J,1)
        O_mm = ownership[ind_mm]
        O_mm = O_mm[:, ~np.isnan(O_mm).all(axis=0)]
        brandsum = O_mm@tmp
        count = O_mm@np.ones((J,1))
        Z_cost[ind_mm] = (sum(tmp)-brandsum)/(J-count)
     
    tmp = Z_cost[:,0]
    tmp = tmp.reshape(N,1)
    product_data["cost_instr0"] = tmp
    
    return(product_data)


# %% Demand 
product_data = pd.read_csv(dpath('product_data.csv'),na_values='NA',low_memory=False)
agent_data = pd.read_csv(dpath('agent_data.csv'),na_values='NA',low_memory=False)

monopoly=pyblp.build_ownership(product_data, 'monopoly')
logit = pyblp.Formulation('prices+size+light+flavor_Plain+log_nflavors',
                              absorb='C(brand)+C(quarter)+C(store)')
Z=['freight_cost','freightxinc','freight2xinc','incxlight',
    'agexlight','agexsize']
product_data=Instr(product_data,Z)
rc=pyblp.Formulation('0+prices+light+size')
demo=pyblp.Formulation('0 + log_income + age')
initial_pi=np.array([[4.3327,  0],
                    [0.215, -.562],
                    [0, -.067]])
initial_sigma=np.zeros((3,3))
optim=pyblp.Optimization('l-bfgs-b', {'gtol':1e-5,'ftol':1e-5})
iteration = pyblp.Iteration('squarem', {'max_evaluations':200},universal_display=False)
problem = pyblp.Problem((logit,rc), product_data, demo, agent_data)
rc_result = problem.solve(sigma=initial_sigma,pi=initial_pi,iteration=iteration,
                            optimization=optim, method='1s')

# %% Data
#rc_result.to_pickle("/home/md/Desktop/rc_result_main.p")
#rc_result = pickle.load(open(dpath("SupplyData/rc_result_main.p"),"rb"))
#product_data = pd.read_csv(dpath('product_data.csv'),low_memory=False)

# Instrument construction
product_data['log_nflavors']=np.log(product_data['nflavors'])
product_data=Instruments(product_data,'size+light+flavor_Plain+log_nflavors')
product_data=PCAInstr([x for x in product_data.columns if x.startswith('diff_instr')],
                      5,'diff_instr',product_data)
product_data=CostInstr(product_data)
product_data["vertical_ids"] = product_data.firm_ids.astype(str).str.contains("PRIVATE LABEL").replace({True: 1, False: 0})


# %% Formulation
Cost= (
    pyRV.Formulation('1 + freight_cost', absorb = 'C(brand)+C(quarter)+C(city)' )
    )

Instr=(
    pyRV.Formulation('0 + blp_instr0 + blp_instr1'),
    pyRV.Formulation('0 +incxlight + agexlight + agexsize'),
    pyRV.Formulation('0 + cost_instr0'),
    pyRV.Formulation('0 + diff_instr0_pca + diff_instr1_pca + diff_instr2_pca + diff_instr3_pca + diff_instr4_pca')
    )
Models=[
    pyRV.Bertrand(ownership='firm_ids'),
    pyRV.Cournot(ownership='firm_ids'),
    pyRV.Monopoly(),
    pyRV.Vertical(
         downstream=pyRV.Monopoly(ownership='firm_ids'),
         upstream=pyRV.Bertrand(ownership='firm_ids'),
     ),
    pyRV.Vertical(
         downstream=pyRV.Monopoly(ownership='firm_ids'),
         upstream=pyRV.Bertrand(ownership='firm_ids'),
         vertical_integration='vertical_ids',
     )
]
product_data["clustering_ids"] = product_data.market_ids 

# %% Run test without first stage adjustment
testing_problem = pyRV.Problem(
    cost_formulation = Cost,
    instrument_formulation = Instr ,
    model_formulations = Models,      
    product_data = product_data,
    demand_results = rc_result
    )

testing_results = testing_problem.solve(
    demand_adjustment = False,
    clustering_adjustment=True
    )

# %% Run test with first stage adjustment
testing_problem = pyRV.Problem(
    cost_formulation = Cost,
    instrument_formulation = Instr ,
    model_formulations = Models,      
    product_data = product_data,
    demand_results = rc_result
    )

testing_results = testing_problem.solve(
    demand_adjustment = True,
    clustering_adjustment=True
    )