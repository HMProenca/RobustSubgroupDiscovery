import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

from RSD.rulelist_class import MDLRuleList
from RSD.measures.subgroup_measures import nominal_discovery_measures
from RSD.util.results2folder import attach_results, print2folder, makefolder_name

directory = "data/single-nominal/"
folder2save_name = "RSD-nominal/RSD-single-nominal-generalisation"
list_datasets = [file.replace(".csv", "") for file in os.listdir(directory)]

task_name = "discovery"
target_type = "categorical"
delim = ","
# user configuration
results = ""
for datasetname in list_datasets:
    # load data
    datasetname = "iris"
    filename =  "./data/single-nominal/"+datasetname+".csv"
    df = pd.read_csv(filename,delimiter=delim)
    X = df.iloc[:,:-1]
    Y= df.iloc[:,-1]
    Y = pd.DataFrame(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.50, random_state = 42)
    model = MDLRuleList(target_type,task = task_name,beam_width = 100)
    model.fit(X_train,Y_train)
    folder_path = makefolder_name(folder2save_name)
    save_rulelist_path = os.path.join(folder_path,datasetname+"_rulelist.pickle")
    with open(save_rulelist_path, 'wb') as f:
        pickle.dump(model, f)
    measures = nominal_discovery_measures(model._rulelist,X_train,Y_train)
    measures["runtime"] = model.runtime
    measures["nsamples_train"] = X.shape[0]

    #add more measures on generalisation
    swkl_test = model.swkl_generalise(X_test, Y_test)

    results = attach_results(measures, results, datasetname)
print2folder(measures, results, folder2save_name)