# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:42:04 2020

@author: Hugo Manuel Proenca
"""
from reproducibility.makegraphs import tableau20,make_graph
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import matplotlib.markers as marker
import matplotlib.axes as axes

from matplotlib.ticker import FormatStrFormatter
from RSD.util.results2folder import makefolder_name
###############################################################################
# Absolute vs Normalized plots! compression ratio
###############################################################################
folder_load = os.path.join("results","hyperparameter testing","categorical_alpha_gain_results","summary.csv")
folder_save = "categorical_hyperparameters_plots"
name_save = "beta_compression"
#save_path = os.path.join(folder_path,name_save+".pdf")
save_path = os.path.join("results",folder_save,name_save+".pdf")

df_all = pd.read_csv(folder_load,index_col=False)

df_norm = df_all.loc[df_all.alpha_gain == 1,:]
df_05 = df_all.loc[df_all.alpha_gain == 0.5,:]
df_abs = df_all.loc[df_all.alpha_gain == 0,:]

labels = df_norm.datasetname.to_numpy()
compression_norm = df_norm.length_ratio.to_numpy()
compression_05 = df_05.length_ratio.to_numpy()
compression_abs = df_abs.length_ratio.to_numpy()

x = np.arange(len(labels))  # the label locations
width_size = 0.2  # the width of the bars
color1=np.array(tableau20[0])/255
color2=np.array(tableau20[2])/255
color3=np.array(tableau20[4])/255
label1= r'$\beta = 1$(normalized)'
label2= r'$\beta = 0.5$'
label3= r'$\beta = 0$ (absolute)'
fig, ax = plt.subplots()
rects1 = ax.bar(x - width_size, compression_norm,
                width= width_size,color = color1, label=label1)
rects2 = ax.bar(x, compression_05,
                width= width_size,color = color3, label=label2)
rects3 = ax.bar(x + width_size, compression_abs,width= width_size,
                color = color2, label=label3)


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('compression ratio')
ax.set_xticks(x)
ax.set_xticklabels(labels,fontdict={'fontsize':11,\
                                       'rotation':'45',\
                                       "horizontalalignment":"right"})
#ax.legend(loc='best', bbox_to_anchor=(0.0, 0., 0.5, 0.5))
ax.legend(loc='best')

fig.tight_layout()
plt.show()
fig.savefig(save_path, bbox_inches='tight')

###############################################################################
# Absolute vs Normalized plots! runtime
###############################################################################
folder_save = "categorical_hyperparameters_plots"
name_save = "beta_runtime"
#save_path = os.path.join(folder_path,name_save+".pdf")
save_path = os.path.join("results",folder_save,name_save+".pdf")

labels = df_norm.datasetname.to_numpy()
compression_norm = df_norm.runtime.to_numpy()/60
compression_05 = df_05.runtime.to_numpy()/60
compression_abs = df_abs.runtime.to_numpy()/60

fig, ax = plt.subplots()
rects1 = ax.bar(x - width_size, compression_norm,
                width= width_size,color = color1, label=label1)
rects2 = ax.bar(x, compression_05,
                width= width_size,color = color3, label=label2)
rects3 = ax.bar(x + width_size, compression_abs,width= width_size,
                color = color2, label=label3)
ax.set_yscale('log')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('runtime (minutes)')
#ax.set_title('dataset')
ax.set_xticks(x)
ax.set_xticklabels(labels,fontdict={'fontsize':11,\
                                       'rotation':'45',\
                                       "horizontalalignment":"right"})  
ax.legend(loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))
ax.legend(loc='best')

fig.tight_layout()
plt.show()
fig.savefig(save_path, bbox_inches='tight')


###############################################################################
# Absolute vs Normalized plots!
###############################################################################
folder_save = "categorical_hyperparameters_plots"
name_save = "beta_SWKL"
#save_path = os.path.join(folder_path,name_save+".pdf")
save_path = os.path.join("results",folder_save,name_save+".pdf")
datainstances = df_abs.wkl_sum/df_abs.wkl_sum_norm
datainstances =datainstances.to_numpy()

compression_norm = df_norm.wkl_sum_norm.to_numpy()
compression_05 = df_05.wkl_sum_norm.to_numpy()
compression_abs = df_abs.wkl_sum_norm.to_numpy()

fig, ax = plt.subplots()
rects1 = ax.bar(x - width_size, compression_norm,
                width= width_size,color = color1, label=label1)
rects2 = ax.bar(x, compression_05,
                width= width_size,color = color3, label=label2)
rects3 = ax.bar(x + width_size, compression_abs,width= width_size,
                color = color2, label=label3)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('SWKL (normalized per $|D|$')
#ax.set_title('dataset')
ax.set_xticks(x)
ax.set_xticklabels(labels,fontdict={'fontsize':11,\
                                       'rotation':'45',\
                                       "horizontalalignment":"right"})  
ax.legend(loc='best')

fig.tight_layout()
plt.show()
fig.savefig(save_path, bbox_inches='tight')
###############################################################################
# absolute vs normalize number of rules 
###############################################################################
folder_save = "categorical_hyperparameters_plots"
name_save = "beta_rules"
#save_path = os.path.join(folder_path,name_save+".pdf")
save_path = os.path.join("results",folder_save,name_save+".pdf")

compression_norm = df_norm.n_rules.to_numpy()
compression_05 = df_05.n_rules.to_numpy()
compression_abs = df_abs.n_rules.to_numpy()

fig, (ax, ax2) = plt.subplots(2, 1, sharex=True)

rects1_top = ax.bar(x - width_size, compression_norm,
                width= width_size,color = color1, label=label1)
rects2_top = ax.bar(x, compression_05,
                width= width_size,color = color3, label=label2)
rects3_top = ax.bar(x + width_size, compression_abs,width= width_size,
                color = color2, label=label3)

rects1_bottom = ax2.bar(x - width_size, compression_norm,
                width= width_size,color = color1, label=label1)
rects2_bottom = ax2.bar(x, compression_05,
                width= width_size,color = color3, label=label2)
rects3_bottom = ax2.bar(x + width_size, compression_abs,width= width_size,
                color = color2, label=label3)


ax.set_ylim(50, 400)  # outliers only
ax2.set_ylim(0, 38)  # most of the data

ax.spines['bottom'].set_visible(False)

ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()
# Add some text for labels, title and custom x-axis tick labels, etc.
plt.grid(False)
fig.text(0.00, 0.6, 'number of rules', va='center', rotation='vertical')

ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
ax2.set_xticks(x)
ax2.set_xticklabels(labels,fontdict={'fontsize':11,\
                                       'rotation':'45',\
                                       "horizontalalignment":"right"})  
ax.legend(loc='best')

#ax.legend()
d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

fig.tight_layout()
plt.show()
fig.savefig(save_path, bbox_inches='tight')

###########################################################
# for numeric targets

###############################################################################
# Absolute vs Normalized plots! compression ratio
###############################################################################
folder_load = os.path.join("results","hyperparameter testing","gaussian_alpha_gain_results","summary.csv")
folder_save = "gaussian_hyperparameters_plots"
name_save = "beta_compression"
#save_path = os.path.join(folder_path,name_save+".pdf")
save_path = os.path.join("results",folder_save,name_save+".pdf")

df_all = pd.read_csv(folder_load,index_col=False)

df_norm = df_all.loc[df_all.alpha_gain == 1,:]
df_05 = df_all.loc[df_all.alpha_gain == 0.5,:]
df_abs = df_all.loc[df_all.alpha_gain == 0,:]

labels = df_norm.datasetname.to_numpy()
compression_norm = df_norm.length_ratio.to_numpy()
compression_05 = df_05.length_ratio.to_numpy()
compression_abs = df_abs.length_ratio.to_numpy()

x = np.arange(len(labels))  # the label locations
width_size = 0.2  # the width of the bars
color1=np.array(tableau20[0])/255
color2=np.array(tableau20[2])/255
color3=np.array(tableau20[4])/255
label1= r'$\beta = 1$(normalized)'
label2= r'$\beta = 0.5$'
label3= r'$\beta = 0$ (absolute)'
fig, ax = plt.subplots()
rects1 = ax.bar(x - width_size, compression_norm,
                width= width_size,color = color1, label=label1)
rects2 = ax.bar(x, compression_05,
                width= width_size,color = color3, label=label2)
rects3 = ax.bar(x + width_size, compression_abs,width= width_size,
                color = color2, label=label3)


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('compression ratio')
ax.set_xticks(x)
ax.set_xticklabels(labels,fontdict={'fontsize':11,\
                                       'rotation':'45',\
                                       "horizontalalignment":"right"})
#ax.legend(loc='best', bbox_to_anchor=(0.0, 0., 0.5, 0.5))
ax.legend(loc='best')

fig.tight_layout()
plt.show()
fig.savefig(save_path, bbox_inches='tight')

###############################################################################
# Absolute vs Normalized plots! runtime
###############################################################################
name_save = "beta_runtime"
#save_path = os.path.join(folder_path,name_save+".pdf")
save_path = os.path.join("results",folder_save,name_save+".pdf")

labels = df_norm.datasetname.to_numpy()
compression_norm = df_norm.runtime.to_numpy()/60
compression_05 = df_05.runtime.to_numpy()/60
compression_abs = df_abs.runtime.to_numpy()/60

fig, ax = plt.subplots()
rects1 = ax.bar(x - width_size, compression_norm,
                width= width_size,color = color1, label=label1)
rects2 = ax.bar(x, compression_05,
                width= width_size,color = color3, label=label2)
rects3 = ax.bar(x + width_size, compression_abs,width= width_size,
                color = color2, label=label3)
ax.set_yscale('log')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('runtime (minutes)')
#ax.set_title('dataset')
ax.set_xticks(x)
ax.set_xticklabels(labels,fontdict={'fontsize':11,\
                                       'rotation':'45',\
                                       "horizontalalignment":"right"})
ax.legend(loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))
ax.legend(loc='best')

fig.tight_layout()
plt.show()
fig.savefig(save_path, bbox_inches='tight')


###############################################################################
# Absolute vs Normalized plots!
###############################################################################
name_save = "beta_SWKL"
#save_path = os.path.join(folder_path,name_save+".pdf")
save_path = os.path.join("results",folder_save,name_save+".pdf")
datainstances = df_abs.wkl_sum/df_abs.wkl_sum_norm
datainstances =datainstances.to_numpy()

compression_norm = df_norm.wkl_sum_norm.to_numpy()
compression_05 = df_05.wkl_sum_norm.to_numpy()
compression_abs = df_abs.wkl_sum_norm.to_numpy()

fig, ax = plt.subplots()
rects1 = ax.bar(x - width_size, compression_norm,
                width= width_size,color = color1, label=label1)
rects2 = ax.bar(x, compression_05,
                width= width_size,color = color3, label=label2)
rects3 = ax.bar(x + width_size, compression_abs,width= width_size,
                color = color2, label=label3)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('SWKL (normalized per $|D|$')
#ax.set_title('dataset')
ax.set_xticks(x)
ax.set_xticklabels(labels,fontdict={'fontsize':11,\
                                       'rotation':'45',\
                                       "horizontalalignment":"right"})
ax.legend(loc='best')

fig.tight_layout()
plt.show()
fig.savefig(save_path, bbox_inches='tight')
###############################################################################
# absolute vs normalize number of rules
###############################################################################
folder_save = "categorical_hyperparameters_plots"
name_save = "beta_rules"
#save_path = os.path.join(folder_path,name_save+".pdf")
save_path = os.path.join("results",folder_save,name_save+".pdf")

compression_norm = df_norm.n_rules.to_numpy()
compression_05 = df_05.n_rules.to_numpy()
compression_abs = df_abs.n_rules.to_numpy()

fig, (ax, ax2) = plt.subplots(2, 1, sharex=True)

rects1_top = ax.bar(x - width_size, compression_norm,
                width= width_size,color = color1, label=label1)
rects2_top = ax.bar(x, compression_05,
                width= width_size,color = color3, label=label2)
rects3_top = ax.bar(x + width_size, compression_abs,width= width_size,
                color = color2, label=label3)

rects1_bottom = ax2.bar(x - width_size, compression_norm,
                width= width_size,color = color1, label=label1)
rects2_bottom = ax2.bar(x, compression_05,
                width= width_size,color = color3, label=label2)
rects3_bottom = ax2.bar(x + width_size, compression_abs,width= width_size,
                color = color2, label=label3)


ax.set_ylim(50, 400)  # outliers only
ax2.set_ylim(0, 38)  # most of the data

ax.spines['bottom'].set_visible(False)

ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()
# Add some text for labels, title and custom x-axis tick labels, etc.
plt.grid(False)
fig.text(0.00, 0.6, 'number of rules', va='center', rotation='vertical')

ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
ax2.set_xticks(x)
ax2.set_xticklabels(labels,fontdict={'fontsize':11,\
                                       'rotation':'45',\
                                       "horizontalalignment":"right"})
ax.legend(loc='best')

#ax.legend()
d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

fig.tight_layout()
plt.show()
fig.savefig(save_path, bbox_inches='tight')
