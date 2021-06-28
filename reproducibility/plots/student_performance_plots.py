import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import seaborn as sns
from matplotlib.patches import Ellipse
folder_save = "exams_grade_plot"
name_save = "first_subroup_absolute"
save_path = os.path.join("results",folder_save,name_save+".pdf")

data = pd.read_csv("../../data/application/student_performance_colombia.csv")
fig, ax = plt.subplots(1, 1, figsize = (6, 7), dpi=150)
mean_data =  [data.loc[:,'QR_PRO'].mean(), data.loc[:,'ENG_PRO'].mean()]
std_data= [data.loc[:,'QR_PRO'].std(), data.loc[:,'ENG_PRO'].std()]


df = data.iloc[:,-2:]
df['subgroup'] = 'default'
with open('../../results/colombia_students/performance_abs.pickle', 'rb') as handle:
    model_abs = pickle.load(handle)
condition1 = pd.Series([True if i in model_abs.rule_sets[0] else False
         for i in range(data.shape[0]) ])
mean_s1= [data.loc[condition1,'QR_PRO'].mean(), data.loc[condition1,'ENG_PRO'].mean()]
std_s1= [data.loc[condition1,'QR_PRO'].std(), data.loc[condition1,'ENG_PRO'].std()]

condition2 = pd.Series([True if i in model_abs.rule_sets[1]
                                and condition1[i] == False
                        else False
         for i in range(data.shape[0]) ])
mean_s2= [data.loc[condition2,'QR_PRO'].mean(), data.loc[condition2,'ENG_PRO'].mean()]
std_s2= [data.loc[condition2,'QR_PRO'].std(), data.loc[condition2,'ENG_PRO'].std()]

# You can use sns.scatterplot + hue parameter
#condition1 = (data.PUBLIC_SCHOOL == 'no') & (data.HOUSEHOLD_INCOME >= 5.0) &\
#            (data.SCHOOL_TYPE == 'ACADEMIC') & (data.SOCIAL_SUPPORT == 'None')


ax.scatter(x='QR_PRO', y='ENG_PRO',data=data[~condition1], color='skyblue', alpha=0.5, label='dataset', s=30)
ax.plot(mean_data[0],mean_data[1],'bx',markeredgewidth=2, markersize=8, label='$\mu$ dataset')

ellipse = Ellipse(xy=(mean_data[0], mean_data[1]),
                  width=std_data[0] * 2,
                  height=std_data[1] * 2,
                  alpha=0.5,
                  edgecolor='b', fc='None', lw=2,label ='$\sigma$ dataset')
ax.add_patch(ellipse)


ax.scatter(x='QR_PRO', y='ENG_PRO',data=data[condition1], color='salmon', alpha=0.5, label='subgroup 1', s=30)
ax.plot(mean_s1[0],mean_s1[1],'rx',markeredgewidth=2, markersize=8, label='$\mu$ subgroup 1')

ellipse = Ellipse(xy=(mean_s1[0], mean_s1[1]),
                  width=std_s1[0] * 2,
                  height=std_s1[1] * 2,
                  alpha=0.5,
                  edgecolor='r', fc='None', lw=2, label ='$\sigma$ subgroup 1')
ax.add_patch(ellipse)

#ax.scatter(x='QR_PRO', y='ENG_PRO',data=data[condition2], color='yellow', alpha=0.5, label='subgroup 2', s=30)

#point = pd.DataFrame({'x': [0.90], 'y': [0.30]})
#ax = ax.plot(x='x', y='y', ax=ax, style='r-', label='point')
#ax.set_title('After', fontsize=15, fontweight='bold')
handles, labels = ax.get_legend_handles_labels()
# sort both labels and handles by labels
labels = [labels[4],labels[0],labels[2],labels[-1],labels[1],labels[3]]
handles = [handles[4],handles[0],handles[2],handles[-1],handles[1],handles[3]]
ax.legend(handles, labels)

# upper & right border remove
ax.set_ylim(0, 101)  # outliers only
ax.set_xlim(0, 101)  # outliers only

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xlabel("Quantitative reasoning grade (%)", fontsize= 12)
plt.ylabel("English grade (%)", fontsize= 12)
fig.tight_layout()
plt.show()
fig.savefig(save_path, bbox_inches='tight')
###############################
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from matplotlib.patches import Ellipse
name_save = "second_subroup_absolute"
#save_path = os.path.join(folder_path,name_save+".pdf")
save_path = os.path.join("results",folder_save,name_save+".pdf")

data = pd.read_csv("../../data/application/student_performance_colombia.csv")
fig, ax = plt.subplots(1, 1, figsize = (6, 7), dpi=150)
mean_data =  [data.loc[:,'QR_PRO'].mean(), data.loc[:,'ENG_PRO'].mean()]
std_data= [data.loc[:,'QR_PRO'].std(), data.loc[:,'ENG_PRO'].std()]


df = data.iloc[:,-2:]
df['subgroup'] = 'default'
with open('../../results/colombia_students/performance_abs.pickle', 'rb') as handle:
    model_abs = pickle.load(handle)
condition1 = pd.Series([True if i in model_abs.rule_sets[0] else False
         for i in range(data.shape[0]) ])
mean_s1= [data.loc[condition1,'QR_PRO'].mean(), data.loc[condition1,'ENG_PRO'].mean()]
std_s1= [data.loc[condition1,'QR_PRO'].std(), data.loc[condition1,'ENG_PRO'].std()]

condition2 = pd.Series([True if i in model_abs.rule_sets[1]
                                and condition1[i] == False
                        else False
         for i in range(data.shape[0]) ])
mean_s2= [data.loc[condition2,'QR_PRO'].mean(), data.loc[condition2,'ENG_PRO'].mean()]
std_s2= [data.loc[condition2,'QR_PRO'].std(), data.loc[condition2,'ENG_PRO'].std()]

# You can use sns.scatterplot + hue parameter
#condition1 = (data.PUBLIC_SCHOOL == 'no') & (data.HOUSEHOLD_INCOME >= 5.0) &\
#            (data.SCHOOL_TYPE == 'ACADEMIC') & (data.SOCIAL_SUPPORT == 'None')


ax.scatter(x='QR_PRO', y='ENG_PRO',data=data[~condition1], color='skyblue', alpha=0.5, label='dataset', s=30)
ax.plot(mean_data[0],mean_data[1],'bx',markeredgewidth=2, markersize=8, label='$\mu$ dataset')

ellipse = Ellipse(xy=(mean_data[0], mean_data[1]),
                  width=std_data[0] * 2,
                  height=std_data[1] * 2,
                  alpha=0.5,
                  edgecolor='b', fc='None', lw=2,label ='$\sigma$ dataset')
ax.add_patch(ellipse)


ax.scatter(x='QR_PRO', y='ENG_PRO',data=data[condition2], color='chocolate', alpha=0.5, label='subgroup 2', s=30)
ax.plot(mean_s2[0],mean_s2[1],'x',color="brown",markeredgewidth=2, markersize=8, label='$\mu$ subgroup 2')

ellipse = Ellipse(xy=(mean_s2[0], mean_s2[1]),
                  width=std_s2[0] * 2,
                  height=std_s2[1] * 2,
                  alpha=0.8,
                  edgecolor='brown', fc='None', lw=2, label ='$\sigma$ subgroup 2')
ax.add_patch(ellipse)

#ax.scatter(x='QR_PRO', y='ENG_PRO',data=data[condition2], color='yellow', alpha=0.5, label='subgroup 2', s=30)

#point = pd.DataFrame({'x': [0.90], 'y': [0.30]})
#ax = ax.plot(x='x', y='y', ax=ax, style='r-', label='point')
#ax.set_title('After', fontsize=15, fontweight='bold')
handles, labels = ax.get_legend_handles_labels()
# sort both labels and handles by labels
labels = [labels[4],labels[0],labels[2],labels[-1],labels[1],labels[3]]
handles = [handles[4],handles[0],handles[2],handles[-1],handles[1],handles[3]]
ax.legend(handles, labels)

# upper & right border remove
ax.set_ylim(0, 101)  # outliers only
ax.set_xlim(0, 101)  # outliers only

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xlabel("Quantitative reasoning grade ()", fontsize= 12)
plt.ylabel("English grade (%)", fontsize= 12)
fig.tight_layout()
plt.show()
fig.savefig(save_path, bbox_inches='tight')

#############################################################################

#############################################################################
#############################################################################
#############################################################################
name_save = "first_subroup_normalized"
#save_path = os.path.join(folder_path,name_save+".pdf")
save_path = os.path.join("results",folder_save,name_save+".pdf")
data = pd.read_csv("../../data/application/student_performance_colombia.csv")
fig, ax = plt.subplots(1, 1, figsize = (6, 7), dpi=150)

df = data.iloc[:,-2:]
df['subgroup'] = 'default'
with open('../../results/colombia_students/performance_norm.pickle', 'rb') as handle:
    model_abs = pickle.load(handle)
condition1 = pd.Series([True if i in model_abs.rule_sets[0] else False
         for i in range(data.shape[0]) ])
mean_s1= [data.loc[condition1,'QR_PRO'].mean(), data.loc[condition1,'ENG_PRO'].mean()]
std_s1= [data.loc[condition1,'QR_PRO'].std(), data.loc[condition1,'ENG_PRO'].std()]

condition2 = pd.Series([True if i in model_abs.rule_sets[1]
                                and condition1[i] == False
                        else False
         for i in range(data.shape[0]) ])
mean_s2= [data.loc[condition2,'QR_PRO'].mean(), data.loc[condition2,'ENG_PRO'].mean()]
std_s2= [data.loc[condition2,'QR_PRO'].std(), data.loc[condition2,'ENG_PRO'].std()]

# You can use sns.scatterplot + hue parameter
#condition1 = (data.PUBLIC_SCHOOL == 'no') & (data.HOUSEHOLD_INCOME >= 5.0) &\
#            (data.SCHOOL_TYPE == 'ACADEMIC') & (data.SOCIAL_SUPPORT == 'None')


ax.scatter(x='QR_PRO', y='ENG_PRO',data=data[~condition1], color='skyblue', alpha=0.5, label='dataset', s=30)
ax.plot(mean_data[0],mean_data[1],'bx',markeredgewidth=2, markersize=8, label='$\mu$ dataset')

ellipse = Ellipse(xy=(mean_data[0], mean_data[1]),
                  width=std_data[0] * 2,
                  height=std_data[1] * 2,
                  alpha=0.5,
                  edgecolor='b', fc='None', lw=2,label ='$\sigma$ dataset')
ax.add_patch(ellipse)


ax.scatter(x='QR_PRO', y='ENG_PRO',data=data[condition1], color='salmon', alpha=0.5, label='subgroup 1', s=30)
ax.plot(mean_s1[0],mean_s1[1],'rx',markeredgewidth=2, markersize=8, label='$\mu$ subgroup 1')

ellipse = Ellipse(xy=(mean_s1[0], mean_s1[1]),
                  width=std_s1[0] * 2,
                  height=std_s1[1] * 2,
                  alpha=0.5,
                  edgecolor='r', fc='None', lw=2, label ='$\sigma$ subgroup 1')
ax.add_patch(ellipse)

#ax.scatter(x='QR_PRO', y='ENG_PRO',data=data[condition2], color='yellow', alpha=0.5, label='subgroup 2', s=30)

#point = pd.DataFrame({'x': [0.90], 'y': [0.30]})
#ax = ax.plot(x='x', y='y', ax=ax, style='r-', label='point')
#ax.set_title('After', fontsize=15, fontweight='bold')
handles, labels = ax.get_legend_handles_labels()
# sort both labels and handles by labels
labels = [labels[4],labels[0],labels[2],labels[-1],labels[1],labels[3]]
handles = [handles[4],handles[0],handles[2],handles[-1],handles[1],handles[3]]
ax.legend(handles, labels)

# upper & right border remove
ax.set_ylim(0, 101)  # outliers only
ax.set_xlim(0, 101)  # outliers only

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xlabel("Quantitative reasoning grade (%)", fontsize= 12)
plt.ylabel("English grade (%)", fontsize= 12)
fig.tight_layout()
plt.show()
fig.savefig(save_path, bbox_inches='tight')
###############################
name_save = "second_subroup_normalized"
#save_path = os.path.join(folder_path,name_save+".pdf")
save_path = os.path.join("results",folder_save,name_save+".pdf")
fig, ax = plt.subplots(1, 1, figsize = (6, 7), dpi=150)

ax.scatter(x='QR_PRO', y='ENG_PRO',data=data[~condition1], color='skyblue', alpha=0.5, label='dataset', s=30)
ax.plot(mean_data[0],mean_data[1],'bx',markeredgewidth=2, markersize=8, label='$\mu$ dataset')

ellipse = Ellipse(xy=(mean_data[0], mean_data[1]),
                  width=std_data[0] * 2,
                  height=std_data[1] * 2,
                  alpha=0.5,
                  edgecolor='b', fc='None', lw=2,label ='$\sigma$ dataset')
ax.add_patch(ellipse)


ax.scatter(x='QR_PRO', y='ENG_PRO',data=data[condition2], color='chocolate', alpha=0.5, label='subgroup 2', s=30)
ax.plot(mean_s2[0],mean_s2[1],'x',color="brown",markeredgewidth=2, markersize=8, label='$\mu$ subgroup 2')

ellipse = Ellipse(xy=(mean_s2[0], mean_s2[1]),
                  width=std_s2[0] * 2,
                  height=std_s2[1] * 2,
                  alpha=0.8,
                  edgecolor='brown', fc='None', lw=2, label ='$\sigma$ subgroup 2')
ax.add_patch(ellipse)

#ax.scatter(x='QR_PRO', y='ENG_PRO',data=data[condition2], color='yellow', alpha=0.5, label='subgroup 2', s=30)

#point = pd.DataFrame({'x': [0.90], 'y': [0.30]})
#ax = ax.plot(x='x', y='y', ax=ax, style='r-', label='point')
#ax.set_title('After', fontsize=15, fontweight='bold')
handles, labels = ax.get_legend_handles_labels()
# sort both labels and handles by labels
labels = [labels[4],labels[0],labels[2],labels[-1],labels[1],labels[3]]
handles = [handles[4],handles[0],handles[2],handles[-1],handles[1],handles[3]]
ax.legend(handles, labels)

# upper & right border remove
ax.set_ylim(0, 101)  # outliers only
ax.set_xlim(0, 101)  # outliers only

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xlabel("Quantitative reasoning grade (%)", fontsize= 12)
plt.ylabel("English grade (%)", fontsize= 12)
fig.tight_layout()
plt.show()
fig.savefig(save_path, bbox_inches='tight')