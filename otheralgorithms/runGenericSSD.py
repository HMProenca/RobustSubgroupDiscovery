# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 17:01:01 2019

@author: gathu
"""
import numpy as np

from otheralgorithms.CN2SD_related_funcs import run_CN2SD_wrapper
from otheralgorithms.DSSD_related_functions import run_DSSD_wrapper

from math import log

from otheralgorithms.FSSD.FSSDCODE import transform_dataset_to_attributes
from otheralgorithms.FSSD.util.csvProcessing import readCSVwithHeader
from otheralgorithms.FSSD_related_functions import run_FSSD_wrapper
from otheralgorithms.all_ssd_related_funcs import (nominal_discovery_measures,
                                                   append_result2file, \
                                                   numeric_discovery_measures, writedf2arff)
log2 = lambda n: log(n or 1, 2)

from RSD.util.results2folder import makefolder_name
from  scipy.io.arff import loadarff
import pandas as pd

def runOtherSSDalgorithms(task, algorithmname, list_datasets,arff_algorithm):
    directory_arff = "data/" + task + "-arff/"
    directory_csv = "data/" + task + "/"

    depthmax = 5.0
    beam_width = 100

    savefile = makefolder_name(algorithmname + task)
    savefile = savefile + "/summary.csv"

    if task == "multi-numeric" or task == "multi-nominal":
        dataset_number_targets = pd.read_csv(directory_arff + "number_targets.csv", index_col=0)
    elif task == "single-numeric" or task == "single-nominal":
        dataset_number_targets = pd.DataFrame(index=[datasetname for datasetname in list_datasets], columns=["number_targets"],
                                                data=[1 for datasetname in list_datasets])

    number_rules_SSD = pd.read_csv(directory_arff + "number_rules_SSD.csv", index_col=0)
    for dataset_number, datasetname in enumerate(list_datasets):
        print("Dataset name: " + str(datasetname))
        number_targets = dataset_number_targets.loc[datasetname, "number_targets"]
        if arff_algorithm:
            file_arff_data = directory_arff + datasetname + ".arff"
            data_arff_scipy = loadarff(file_arff_data)
            attribute_names = [att for att in data_arff_scipy[1]._attributes]
            df = pd.DataFrame(data_arff_scipy[0])

        else:
            file_csv_data =  directory_csv + datasetname + ".csv"
            df = pd.read_csv(file_csv_data)
            attribute_names = df.columns[0:-number_targets]
        # train test split
        indeces_train, indeces_test = np.arange(df.shape[0]), np.arange(0)
        # indeces_train, indeces_test = train_test_split(indexes_alldataset, test_size = 0.33, random_state = 42)
        X_train, Y_train = df.iloc[indeces_train, :-number_targets], df.iloc[indeces_train, -number_targets:]
        X_test, Y_test = df.iloc[indeces_test, :-number_targets], df.iloc[indeces_test, -number_targets:]
        Y_train = pd.DataFrame(Y_train)
        Y_test = pd.DataFrame(Y_test)

        # change configuration file of DSSD
        if algorithmname in ['top-k','seq-cover', 'DSSD']:
            save_file_tmp_arff = 'otheralgorithms/DSSD/data/datasets/tmp/tmp.arff'
            writedf2arff(df, indeces_train, file_arff_data, save_file_tmp_arff)
            nitems, subgroup_sets_support_bitset, timespent =run_DSSD_wrapper(algorithmname, beam_width,
                        number_rules_SSD, datasetname, df, task, depthmax, attribute_names,number_targets)
        elif algorithmname == 'MCTS4DM':
            save_file_tmp_arff = 'otheralgorithms/MCTS4DM/data/datasets/tmp/tmp.arff'
            writedf2arff(df, indeces_train, file_arff_data, save_file_tmp_arff)

        elif algorithmname == 'FSSD':
            class_attribute = 'class'
            attributes, types = transform_dataset_to_attributes(file_csv_data, class_attribute, delimiter=',')
            dataset, header = readCSVwithHeader(file_csv_data,numberHeader=[a for a, t in zip(attributes, types) if t == 'numeric'],
                                                delimiter=',')
            #dataset_train = [dataset[irow] for irow in indeces_train] # for nursery something weird happens here...
            dataset_train = dataset
            nitems, subgroup_sets_support_bitset, timespent = run_FSSD_wrapper(dataset_train, attributes,
                                                                               class_attribute, types, depthmax)
        elif algorithmname in ['CN2SD-entro','CN2SD-wracc']:
            class_attribute = 'class'
            attributes, types = transform_dataset_to_attributes(file_csv_data, class_attribute, delimiter=',')
            dataset, header = readCSVwithHeader(file_csv_data,numberHeader=[a for a, t in zip(attributes, types) if t == 'numeric'],
                                                delimiter=',')
            #dataset_train = [dataset[irow] for irow in indeces_train]
            dataset_train = dataset
            if algorithmname == 'CN2SD-entro':
                quality = 'entropy'
            elif algorithmname == 'CN2SD-wracc':
                quality = 'wracc'
            nitems, subgroup_sets_support_bitset, timespent = run_CN2SD_wrapper(dataset_train, attributes,types,
                                                                               class_attribute, beam_width, depthmax,quality)
        else:
            raise ValueError("Wrong algorithmname selected. please try one from this list ['top-k','seq-cover','DSSD',"
                             "'CN2SD','MCTS4DM','FSSD']")
        # Train dataset
        nrows_train = Y_train.shape[0]
        if task == "single-nominal" or task == "multi-nominal":
            default_prob_per_class_train = {name: {category: sum(columnvals == category) / nrows_train for category in columnvals.unique()}
                                  for name, columnvals in Y_train.items()}
            measures_train = nominal_discovery_measures(default_prob_per_class_train, subgroup_sets_support_bitset, X_train, Y_train)

        elif task == "single-numeric" or task == "multi-numeric":
            # other measures
            measures_train = numeric_discovery_measures(subgroup_sets_support_bitset, X_train, Y_train)
        else:
            raise Exception("Wrong task name")
        measures_train["avg_items"] = sum(nitems) / len(nitems)
        measures_train["runtime"] = timespent

        if dataset_number == 0:
            string = "datasetname," + ",".join([meas for meas in measures_train]) + " \n"
            with open(savefile, 'w') as file:
                file.write("%s" % string)
        append_result2file(measures_train, datasetname, savefile)
