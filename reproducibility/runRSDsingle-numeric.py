import pandas as pd
from RSD.rulelist_class import MDLRuleList
from RSD.measures.subgroup_measures import numeric_discovery_measures
import os
import pickle

from RSD.util.results2folder import attach_results, print2folder, makefolder_name

directory = "data/single-numeric/"
folder2save_name = "RSD-numeric/RSD-single-numeric"
list_datasets = [file.replace(".csv", "") for file in os.listdir(directory)]
list_datasets = ["baseball","autoMPG8","dee","ele-1","forestFires","concrete",\
                "treasury","wizmir","abalone","puma32h","ailerons","elevators",\
                "bikesharing","california","house"]
task_name = "discovery"
target_type = "gaussian"
delim = ","
# user configuration
results = ""
for datasetname in list_datasets:
    # load data
    filename =  "./data/single-numeric/"+datasetname+".csv"
    df = pd.read_csv(filename,delimiter=delim)
    X = df.iloc[:,:-1]
    Y= df.iloc[:,-1]
    Y = pd.DataFrame(Y)
    model = MDLRuleList(target_type,task = task_name,beam_width = 100)
    model.fit(X,Y)
    folder_path = makefolder_name(folder2save_name)
    save_rulelist_path = os.path.join(folder_path,datasetname+"_rulelist.pickle")
    with open(save_rulelist_path, 'wb') as f:
        pickle.dump(model, f)
    measures = numeric_discovery_measures(model._rulelist,X,Y)
    measures["runtime"] = model.runtime
    measures["nsamples_train"] = X.shape[0]
    results = attach_results(measures, results, datasetname)
print2folder(measures, results, folder2save_name)