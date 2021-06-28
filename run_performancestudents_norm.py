import pandas as pd
import numpy as np
from rulelist.rulelist import RuleList
import pickle

dirctory = 'data/application/student-combined-processed.csv'
data = pd.read_csv(dirctory)
task = 'discovery'
target_model = 'gaussian'

Y = data.iloc[:,-2:]
X = data.iloc[:,:-2]

model_norm = RuleList(task = task, target_model = target_model,alpha_gain=1.0)

model_norm.fit(X, Y)
print(model_norm)
with open('results/performance_norm.pickle', 'wb') as handle:
    pickle.dump(model_norm, handle)

model_abs = RuleList(task = task, target_model = target_model,alpha_gain = 0.0)
model_abs.fit(X, Y)

with open("results/colombia_students/performance_abs_QR&ENG_PRO.pickle", "rb") as input_file:
    model_abs = pickle.load(input_file)

model_abs.__dict__.keys()