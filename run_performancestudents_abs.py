import pandas as pd
import numpy as np
from RSD.rulelist_class import MDLRuleList
import pickle
from RSD.measures.subgroup_measures import numeric_discovery_measures, nominal_discovery_measures

# absolute gain experiment
dirctory = 'data/application/student_performance_colombia.csv'
data = pd.read_csv(dirctory)
task = 'discovery'
target_model = 'gaussian'
Y = data.iloc[:,-2:]
X = data.iloc[:,:-2]
model_abs = MDLRuleList(task = task, target_model = target_model,max_rules=4, alpha_gain = 1.0)
model_abs.fit(X, Y)
with open('results/performance_abs.pickle', 'wb') as handle:
    pickle.dump(model_abs, handle)
print(model_abs)

# normalized gain experiment
dirctory = 'data/application/student_performance_colombia.csv'
data = pd.read_csv(dirctory)
task = 'discovery'
target_model = 'gaussian'
Y = data.iloc[:,-2:]
X = data.iloc[:,:-2]
model_norm = MDLRuleList(task = task, target_model = target_model,max_rules=4, alpha_gain = 0.0)
model_norm.fit(X, Y)
with open('results/performance_abs.pickle', 'wb') as handle:
    pickle.dump(model_norm, handle)
print(model_norm)

