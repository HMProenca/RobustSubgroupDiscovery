import pandas as pd
from RSD.rulelist_class import MDLRuleList
from RSD.measures.subgroup_measures import nominal_discovery_measures
import os
import pickle

from RSD.util.results2folder import attach_results, print2folder, makefolder_name

directory = "data/multi-nominal/"
folder2save_name = "RSD-nominal/RSD-multi-nominal"
list_datasets = [file.replace(".csv", "") for file in os.listdir(directory)]
list_datasets = ['birds','CAL500','Corel5k','emotions','flags','genbase','mediamill','scene','yeast']

dataset_number_targets = pd.read_csv(directory+"number_targets.csv",index_col=0)
task_name = "discovery"
target_type = "categorical"
delim = ","
# user configuration
results = ""
for datasetname in list_datasets:
    # load data
    filename =  directory+datasetname+".csv"
    number_targets = dataset_number_targets.loc[datasetname,"number_targets"]

    df = pd.read_csv(filename,delimiter=delim)
    X = df.iloc[:,:-number_targets]
    Y= df.iloc[:,-number_targets:]
    Y = pd.DataFrame(Y)
    model = MDLRuleList(target_type,task = task_name,beam_width = 100)
    model.fit(X,Y)
    folder_path = makefolder_name(folder2save_name)
    save_rulelist_path = os.path.join(folder_path,datasetname+"_rulelist.pickle")
    with open(save_rulelist_path, 'wb') as f:
        pickle.dump(model, f)
    measures = nominal_discovery_measures(model._rulelist,X,Y)
    measures["runtime"] = model.runtime
    measures["nsamples_train"] = X.shape[0]
    results = attach_results(measures, results, datasetname)
print2folder(measures, results, folder2save_name)