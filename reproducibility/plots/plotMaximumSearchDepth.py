# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 16:44:30 2020

@author: Hugo Manuel Proenca
"""
from reproducibility.makegraphs import tableau20,make_graph
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import matplotlib.markers as marker
import matplotlib.axes as axes

from matplotlib.ticker import FormatStrFormatter
from RSD.util.results2folder import makefolder_name
###############################################################################
# maximum depth of search categorical
###############################################################################
folder_load = os.path.join("results","hyperparameter testing","categorical_max_depth_results","summary.csv")
folder_save = "categorical_hyperparameters_plots"
folder_path = makefolder_name(folder_save)
df = pd.read_csv(folder_load,index_col=False)

datasetsnames= np.unique(df.datasetname)
results2plot = dict()
for datname in datasetsnames:
    results2plot[datname] = dict()
    results2plot[datname]["maxdepth"] = df[df.datasetname == datname].max_depth.to_numpy()
    results2plot[datname]["compression"] = df[df.datasetname == datname].length_ratio.to_numpy()
    #results2plot[datname]["time"] = df[df.datasetname == datname].runtime.to_numpy()
    results2plot[datname]["conditions"] = df[df.datasetname == datname].avg_items.to_numpy()

fig,lgd = make_graph(results2plot,"maxdepth","compression",size_marker = 6,color = tableau20,typeofplot ="plot",separate_colour =1)
plt.axvline(x=5,linestyle= '--', linewidth=0.6, color='k')
plt.xlabel("maximum depth")
plt.ylabel("relative compression")
save_path = os.path.join(folder_path,"maxdepth_vs_compression.pdf")
fig.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight')



fig,lgd = make_graph(results2plot,"maxdepth","conditions",size_marker = 6,color = tableau20,typeofplot ="plot",separate_colour =1)
plt.axvline(x=5,linestyle= '--', linewidth=0.6, color='k')
plt.xlabel("maximum depth")
plt.ylabel("average conditions per subgroup")
save_path = os.path.join(folder_path,"maxdepth_vs_conditions.pdf")
fig.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight')

###############################################################################
# maximum depth of search numeric
###############################################################################
folder_load = os.path.join("results","hyperparameter testing","gaussian_max_depth_results","summary.csv")
folder_save = "gaussian_hyperparameters_plots"
folder_path = makefolder_name(folder_save)
df = pd.read_csv(folder_load,index_col=False)

datasetsnames= np.unique(df.datasetname)
results2plot = dict()
for datname in datasetsnames:
    results2plot[datname] = dict()
    results2plot[datname]["maxdepth"] = df[df.datasetname == datname].max_depth.to_numpy()
    results2plot[datname]["compression"] = df[df.datasetname == datname].length_ratio.to_numpy()
    #results2plot[datname]["time"] = df[df.datasetname == datname].runtime.to_numpy()
    results2plot[datname]["conditions"] = df[df.datasetname == datname].avg_items.to_numpy()

fig,lgd = make_graph(results2plot,"maxdepth","compression",size_marker = 6,color = tableau20,typeofplot ="plot",separate_colour =1)
plt.axvline(x=5,linestyle= '--', linewidth=0.6, color='k')
plt.xlabel("maximum depth")
plt.ylabel("relative compression")
save_path = os.path.join(folder_path,"maxdepth_vs_compression.pdf")
fig.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight')


fig,lgd = make_graph(results2plot,"maxdepth","conditions",size_marker = 6,color = tableau20,typeofplot ="plot",separate_colour =1)
plt.axvline(x=5,linestyle= '--', linewidth=0.6, color='k')
plt.xlabel("maximum depth")
plt.ylabel("average conditions per subgroup")
save_path = os.path.join(folder_path,"maxdepth_vs_conditions.pdf")
fig.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight')