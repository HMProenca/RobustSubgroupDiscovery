"""
For the most recent version of the Subgroup Lists and Rule Lists code please refer to
https://github.com/HMProenca/RuleList

"""

import pandas as pd
#from rulelist.rulelist import RuleList
from RSD.rulelist_class import MDLRuleList

target_model = 'categorical'
task = "discovery"

# user configuration
datasetname= "breastCancer"
delim = ','
disc_type = "static"
max_len = 5
beamsize = 100
ncutpoints = 5
# load data
filename =  "./data/single-nominal/"+datasetname+".csv"
df = pd.read_csv(filename,delimiter=delim, na_values='?')
Y = df.iloc[:,-1:]
X = df.iloc[:,:-1]
model = MDLRuleList(task = task, target_model = target_model,max_rules=3, n_cutpoints=ncutpoints)
model.fit(X, Y)

print(model)
