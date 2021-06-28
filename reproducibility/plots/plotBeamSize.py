# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 16:41:42 2020

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
# beamsize  categorical
###############################################################################

folder_load = os.path.join("results","hyperparameter testing","categorical_beam_width_results","summary.csv")
folder_save = "categorical_hyperparameters_plots"
folder_path = makefolder_name(folder_save)
df_beam = pd.read_csv(folder_load,index_col=False)

datasetsnames= np.unique(df_beam.datasetname)
results2plot = dict()
for datname in datasetsnames:
    results2plot[datname] = dict()
    results2plot[datname]["beamsize"] = df_beam[df_beam.datasetname == datname].beam_width.to_numpy()
    results2plot[datname]["compression"] = df_beam[df_beam.datasetname == datname].length_ratio.to_numpy()
    results2plot[datname]["wkl_sum"] = df_beam[df_beam.datasetname == datname].wkl_sum.to_numpy()

fig,lgd = make_graph(results2plot,"beamsize","compression",size_marker = 6,color = tableau20,typeofplot ="semilogx",separate_colour =1)
plt.axvline(x=100,linestyle= '--', linewidth=0.6, color='k')
plt.xlabel("beam width")
plt.ylabel("relative compression")
save_path = os.path.join(folder_path,"beam_vs_compression.pdf")
fig.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight')


###############################################################################
# beamsize  numeric targets
###############################################################################

folder_load = os.path.join("results","hyperparameter testing","gaussian_beam_width_results","summary.csv")
folder_save = "gaussian_hyperparameters_plots"
folder_path = makefolder_name(folder_save)
df_beam = pd.read_csv(folder_load,index_col=False)

datasetsnames= np.unique(df_beam.datasetname)
results2plot = dict()
for datname in datasetsnames:
    results2plot[datname] = dict()
    results2plot[datname]["beamsize"] = df_beam[df_beam.datasetname == datname].beam_width.to_numpy()
    results2plot[datname]["compression"] = df_beam[df_beam.datasetname == datname].length_ratio.to_numpy()
    results2plot[datname]["wkl_sum"] = df_beam[df_beam.datasetname == datname].wkl_sum.to_numpy()

    #results2plot[datname]["time"] = df_beam[df_beam.datasetname == datname].runtime.to_numpy()

fig,lgd = make_graph(results2plot,"beamsize","compression",size_marker = 6,color = tableau20,typeofplot ="semilogx",separate_colour =1)
plt.axvline(x=100,linestyle= '--', linewidth=0.6, color='k')
plt.xlabel("beam width")
plt.ylabel("relative compression")
save_path = os.path.join(folder_path,"beam_vs_compression.pdf")
fig.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight')