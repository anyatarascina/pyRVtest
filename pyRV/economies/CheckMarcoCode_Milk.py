import pickle

data_file = "/Volumes/RiversVuong_Endogeneity/RVpaper/CodeCheck/Data/DemandData/rc2_result.p"
cost_file = "/Volumes/RiversVuong_Endogeneity/RVpaper/CodeCheck/Data/SupplyData3/costshifters.csv"


pyRV_path = '/Users/cjs/Dropbox/Economics/Research/RiversVuong_Endogeneity/pyRV_folder'

import sys
sys.path.append(pyRV_path)

import numpy as np
import pandas as pd
import pyblp
import pyRV_multipleinstruments as pyRV

# RV TESTING
pickle_off = open(data_file,"rb")
emp = pickle.load(pickle_off)
product_data = emp['product_data']
pyblp_results = emp['results']


#pip3 install scikit-learn
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X  = product_data[['blp_instr0', 'blp_instr1', 'blp_instr2','blp_instr3', 'blp_instr4', 'blp_instr5', 'blp_instr6', 'blp_instr7','blp_instr8', 'blp_instr9']]
Z_blp_PCA = pca.fit_transform(X)
print(pca.explained_variance_ratio_)

pca = PCA(n_components=5)
X  = product_data[['diff_instr0','diff_instr1', 'diff_instr2', 'diff_instr3', 'diff_instr4','diff_instr5', 'diff_instr6', 'diff_instr7']]
Z_gh_PCA = pca.fit_transform(X)
print(pca.explained_variance_ratio_)

N = np.size(product_data.prices)
mkts = np.unique(product_data.market_ids)


ownership = pyRV.build_ownership_testing(product_data,"firm_ids")


temp_data = pd.read_csv('/Volumes/sullivan/collusion_markups/Sarah/yogurt/Data/DemandData/product_data_quarterstoreall_milk.csv')
product_data['milk_price'] = temp_data['milk_price']
product_data['milk_price'] = product_data['milk_price'].fillna(0)


cost = np.array(product_data[['freight_cost', 'milk_price']]) 
Z_cost = np.zeros((N,np.shape(cost)[1]))
for mm in mkts:
    ind_mm = np.where(product_data.market_ids == mm)[0]
    tmp = cost[ind_mm,:]
    J = np.shape(tmp)[0]
    tmp = tmp.reshape(J,np.shape(cost)[1])
    O_mm = ownership[ind_mm]
    O_mm = O_mm[:, ~np.isnan(O_mm).all(axis=0)]
    brandsum = O_mm@tmp
    count = O_mm@np.ones((J,1))
    Z_cost[ind_mm,:] = (np.sum(tmp, axis = 0)-brandsum)/(J-count)




tmp = Z_blp_PCA[:,0]
tmp = tmp.reshape(N,1)
product_data["blppca_instr0"] = tmp
tmp = Z_blp_PCA[:,1]
tmp = tmp.reshape(N,1)
product_data["blppca_instr1"] = tmp

tmp = Z_gh_PCA[:,0]
tmp = tmp.reshape(N,1)
product_data["ghpca_instr0"] = tmp
tmp = Z_gh_PCA[:,1]
tmp = tmp.reshape(N,1)
product_data["ghpca_instr1"] = tmp
tmp = Z_gh_PCA[:,2]
tmp = tmp.reshape(N,1)
product_data["ghpca_instr2"] = tmp
tmp = Z_gh_PCA[:,3]
tmp = tmp.reshape(N,1)
product_data["ghpca_instr3"] = tmp
tmp = Z_gh_PCA[:,4]
tmp = tmp.reshape(N,1)
product_data["ghpca_instr4"] = tmp

tmp = Z_cost[:,0]
tmp = tmp.reshape(N,1)
product_data["cost_instr0"] = tmp

product_data["clustering_ids"] = product_data.market_ids

vi_var = np.zeros((N,1))

for ii in range(N):
    if "PRIVATE LABEL" in  product_data.firm_ids[ii]:
        vi_var[ii] = 1

product_data["vertical_ids"] = vi_var


