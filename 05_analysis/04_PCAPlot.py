#!/usr/bin/env python3

import sys as SYS
import pandas as PD
import numpy as NP
import matplotlib as MP
import matplotlib.pyplot as PLT

# load self-made toolboxes
SYS.path.append('../toolboxes') # makes the folder where the toolbox files are located accessible to python
import EigenToolbox as ET    # PCA


master_data = PD.read_csv(f'../data/master_data.csv', sep = ';')
master_data['cycle_idx'] = [f'c{idx:02.0f}' for idx in master_data['cycle_nr'].values]
master_data.set_index('cycle_idx', inplace = True, drop = True)

posture = PD.read_csv(f'../data/fcas_posture.csv', sep = ';')
posture.set_index('cycle_idx', inplace = True, drop = True)
coordination_raw = PD.read_csv(f'../data/fcas_coordination_raw.csv', sep = ';')
coordination_raw.set_index('cycle_idx', inplace = True, drop = True)
coordination_trafo = PD.read_csv(f'../data/fcas_coordination_trafo.csv', sep = ';')
coordination_trafo.set_index('cycle_idx', inplace = True, drop = True)

# print (master_data.sample(3).T)
# print (posture.sample(3).T)
# print (coordination_trafo.sample(3).T.iloc[:5, :])

pca = ET.PrincipalComponentAnalysis.Load('../data/fcas_coordination_pca.pca')
# print (pca.weights)
n_components = NP.argmax(NP.cumsum(pca.weights) >= 0.8)
coordination_trafo = coordination_trafo.loc[:, ['PC%i'%(i+1) for i in range(n_components)]]

data = master_data.join(posture).join(coordination_trafo)
print (data.sample(5).T)


data = data.loc[data['dutyfactor'].values > 0.5, :]


params = [ \
      "speed" \
    # , "clearance" \
    , "dutyfactor" \
    , "μ_wrist" \
    , "α_wrist" \
    , "PC1" \
    , "PC2" \
    # , "PC3" \
    ]

import seaborn as SB
SB.set_theme(style="ticks")
df = data.loc[:, params]
dpi = 300
PLT.rcParams['figure.figsize']=(1920/dpi, 1080/dpi)
PLT.rcParams['figure.dpi']=dpi
SB.pairplot(df)
PLT.gcf().savefig('../figures/zebra_data_overview.png', transparent = False, dpi = dpi)
PLT.show()
