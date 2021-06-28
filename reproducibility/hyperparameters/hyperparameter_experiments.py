import pandas as pd

from RSD.util.results2folder import attach_results,print2folder
from RSD.rulelist_class import MDLRuleList
from RSD.measures.subgroup_measures import numeric_discovery_measures, nominal_discovery_measures

def run_hyper_exps(hyperparameter, list_values_hyper, list_datasets, task_name, target_type):
    results = ""
    for datasetname in list_datasets:
        print("Dataset name: " + datasetname)
        for hyper_value in list_values_hyper:
            print("   "+ hyperparameter + " : " + str(hyper_value))
            if target_type == "gaussian":
                filename =  "./data/single-numeric/"+datasetname+".csv"
            elif target_type == "categorical":
                filename =  "./data/single-nominal/"+datasetname+".csv"
            df = pd.read_csv(filename,delimiter=",")
            X = df.iloc[:, :-1]
            Y = df.iloc[:, -1]
            Y = pd.DataFrame(Y)
            if hyperparameter == "beam_width":
                model = MDLRuleList(target_type,beam_width = hyper_value,task = task_name)
            elif hyperparameter == "max_depth":
                model = MDLRuleList(target_type,max_depth = hyper_value,task = task_name)
            elif hyperparameter == "n_cutpoints":
                model = MDLRuleList(target_type,n_cutpoints = hyper_value,task = task_name)
            elif hyperparameter == "min_support":
                model = MDLRuleList(target_type,min_support = hyper_value,task = task_name)
            elif hyperparameter == "alpha_gain":
                model = MDLRuleList(target_type,alpha_gain = hyper_value,task = task_name)
            model.fit(X,Y)
            if target_type == "gaussian":
                measures = numeric_discovery_measures(model._rulelist, X, Y)
            elif target_type == "categorical":
                measures = nominal_discovery_measures(model._rulelist, X, Y)
            measures[hyperparameter]=hyper_value
            results = attach_results(measures,results,datasetname)
    results = results.rstrip(", \n")
    name_folder = target_type+"_"+hyperparameter+"_results"
    print2folder(measures,results,name_folder)

run_hyper_exps