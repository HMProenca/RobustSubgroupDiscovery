# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 17:49:57 2020

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
# runtime plot
###############################################################################

name_save = "plot_runtime"
datatype= '-nominal'
algorithms = ["RSD","top-k","seq-cover","CN2-SD"]
folder_path = makefolder_name(name_save)
variable = "runtime"
s=50
alp = 0.7
fig = plt.figure()
ax = plt.gca()
list_markers=['s','D','v','^','<',"o",'>']
# load data
results = dict()
for ialg,alg in enumerate(algorithms):
    folder_load = os.path.join("results",alg+datatype,"summary.csv")
    results[alg]= pd.read_csv(folder_load,index_col=False)

labelstotal = results["top-k"].datasetname.to_numpy()
#ax.axvline(10.5,linewidth =1,linestyle="-.", color =(0,0,0))
for ialg,alg in enumerate(algorithms):
    print(alg)
    labels = results[alg].datasetname.to_numpy()
    labels = [lb.replace(" ", "") for lb in labels]
    x = [np.where(labelstotal == lb) for i,lb in enumerate(labels)]
    #x = [i for i,lb in enumerate(labelstotal) if lb in labels]
    resultsalg = results[alg].loc[:,variable].to_numpy()
    ax.scatter(x, resultsalg,s,alpha =alp,
               c=np.array(tableau20[2*ialg])/255,edgecolor = (0,0,0),
               marker=list_markers[ialg],label=alg)

plt.axvline(x=9.5, linestyle='--', linewidth=0.6, color='k')
plt.axvline(x=19.5, linestyle='--', linewidth=0.6, color='k')

ax.set_yscale('log')
x = np.arange(len(labelstotal))  # the label locations
ax.set_xticks(x)
ax.set_xticklabels(labelstotal,fontdict={'fontsize':10,\
                                       'rotation':'45',\
                                       "horizontalalignment":"right"})  
ax.grid(b=True, which='major', axis='y', linestyle= '--', linewidth=0.6)
#ax.legend()
ax.legend(loc='best')
plt.ylabel("runtime (seconds)", fontsize= 12)
save_path = os.path.join(folder_path,"plot_runtime"+datatype+ ".pdf")
plt.show()
fig.savefig(save_path, bbox_inches='tight')

###############################################################################
# runtime plot numeric
###############################################################################

name_save = "plot_runtime"
datatype= '-numeric'
algorithms = ["RSD","top-k","seq-cover"]
folder_path = makefolder_name(name_save)
variable = "runtime"
s=50
alp = 0.7
fig = plt.figure()
ax = plt.gca()
list_markers=['s','D','v','^','<',"o",'>']
# load data
results = dict()
for ialg,alg in enumerate(algorithms):
    folder_load = os.path.join("results",alg+datatype,"summary.csv")
    results[alg]= pd.read_csv(folder_load,index_col=False)

labelstotal = results["RSD"].datasetname.to_numpy()
#ax.axvline(10.5,linewidth =1,linestyle="-.", color =(0,0,0))
for ialg,alg in enumerate(algorithms):
    print(alg)
    labels = results[alg].datasetname.to_numpy()
    labels = [lb.replace(" ", "") for lb in labels]
    x = [np.where(labelstotal == lb)[0] for i,lb in enumerate(labels)]
    #x = [i for i,lb in enumerate(labelstotal) if lb in labels]
    resultsalg = results[alg].loc[:,variable].to_numpy()
    ax.scatter(x, resultsalg,s,alpha =alp,
               c=np.array(tableau20[2*ialg])/255,edgecolor = (0,0,0),
               marker=list_markers[ialg],label=alg)

plt.axvline(x=14.5, linestyle='--', linewidth=0.6, color='k')

ax.set_yscale('log')
x = np.arange(len(labelstotal))  # the label locations
ax.set_xticks(x)
ax.set_xticklabels(labelstotal,fontdict={'fontsize':10,\
                                       'rotation':'45',\
                                       "horizontalalignment":"right"})
ax.grid(b=True, which='major', axis='y', linestyle= '--', linewidth=0.6)
#ax.legend()
ax.legend(loc='best')
plt.ylabel("runtime (seconds)", fontsize= 12)
save_path = os.path.join(folder_path,"plot_runtime"+datatype+ ".pdf")
plt.show()
fig.savefig(save_path, bbox_inches='tight')

###############################################################################
# jaccard plot
###############################################################################

name_save = "plot_jaccard"
algorithms = ["RSD","seqcover","topk"]
folder_path = makefolder_name(name_save)
variable = "jacc_avg"
s=50
alp = 0.7
fig = plt.figure()
ax = plt.gca()
list_markers=['s','D','v','^','<',"o",'>']
# load data
results = dict()
for ialg,alg in enumerate(algorithms):
    folder_load = os.path.join("results",alg,"summary.csv")
    results[alg]= pd.read_csv(folder_load,index_col=False)

labelstotal = results["RSD"].datasetname.to_numpy()
#ax.axvline(10.5,linewidth =1,linestyle="-.", color =(0,0,0))
for ialg,alg in enumerate(algorithms):
    labels = results[alg].datasetname.to_numpy()
    labels = [lb.replace(" ", "") for lb in labels]
    x = [i for i,lb in enumerate(labelstotal) if lb in labels]
    resultsalg = results[alg][variable]
    ax.scatter(x, resultsalg,s,alpha =alp,
               c=np.array(tableau20[2*ialg])/255,edgecolor = (0,0,0),
               marker=list_markers[ialg],label=alg)
x = np.arange(len(labelstotal))  # the label locations
ax.set_xticks(x)
ax.set_xticklabels(labelstotal,fontdict={'fontsize':11,\
                                       'rotation':'45',\
                                       "horizontalalignment":"right"})  
#ax.legend()
ax.grid(b=True, which='major', axis='y', linestyle= '--', linewidth=0.6)

ax.legend(loc='best')
plt.ylabel("jaccard average")
save_path = os.path.join(folder_path,name_save+ ".pdf")
fig.savefig(save_path, bbox_inches='tight')



