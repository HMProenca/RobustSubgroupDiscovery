import csv
import numpy as np
import pandas as pd

from gmpy2 import xmpz, mpz
from pandas.core.dtypes.common import is_numeric_dtype

from RSD.measures.subgroup_measures import jaccard_index_model, numeric_single2multitargets_function, \
    kullbackleibler_gaussian_paramters, wkl_wracc, wracc_numeric
from RSD.util.bitset_operations import indexes2bitset, bitset2indexes
from RSD.util.build.extra_maths import log2_0


def append_result2file(measures,datasetname,resultsfile):
    string = datasetname + ","
    for meas in measures:
        string +=  str(round(measures[meas],4)) + ","
    string += " \n"
    with open(resultsfile, 'a') as file:
        file.write("%s" % string)
    return string

def discoverymetrics_numeric(targetvalues, nrules, rules_supp, rules_usg, subgroup_sets_support, subgroup_sets_usage):
    mean_dataset = np.mean(targetvalues)
    variance_dataset = np.var(targetvalues)
    kl_supp, kl_usg, wkl_supp, wkl_usg, wkl_sum = np.zeros(nrules), np.zeros(nrules), np.zeros(nrules), np.zeros(
        nrules), np.zeros(nrules)
    wacc_supp, wacc_usg = np.zeros(nrules), np.zeros(nrules)
    support, usage = np.zeros(nrules), np.zeros(nrules)
    stdrules = []
    top1_std = 0
    for r in range(nrules):
        idx_support = list(subgroup_sets_support[r])
        values_support = targetvalues[idx_support]
        support[r] = len(values_support)
        kl_supp[r], wkl_supp[r] = kullbackleibler_gaussian(mean_dataset, variance_dataset, values_support)
        wacc_supp[r] = wracc_numeric(mean_dataset, values_support)
    for r in range(len(subgroup_sets_usage)):
        idx_usage = list(subgroup_sets_usage[r])
        values_usage = targetvalues[idx_usage]
        usage[r] = len(values_usage)
        kl_usg[r], wkl_usg[r] = kullbackleibler_gaussian(mean_dataset, variance_dataset, values_usage)
        wacc_usg[r] = wracc_numeric(mean_dataset, values_usage)
        if usage[r]:
            stdrules.append(np.std(values_usage))
        if r == 0:
            top1_std = np.std(values_usage)

    wkl_sum = sum(wkl_usg)
    #  Average them all!!!!
    measures = dict()
    measures["avg_supp"] = np.mean(support)
    measures["kl_supp"] = np.mean(kl_supp)
    measures["wkl_supp"] = np.mean(wkl_supp)

    measures["avg_usg"] = np.mean(usage)
    measures["kl_usg"] = np.mean(kl_usg)
    measures["wkl_usg"] = np.mean(wkl_usg)

    measures["wacc_supp"] = np.mean(wacc_supp)
    measures["wacc_usg"] = np.mean(wacc_usg)
    measures["wkl_sum"] = wkl_sum

    measures["std_rules"] = np.mean(stdrules)
    measures["top1_std"] = top1_std
    return measures

def writedf2arff(df, indeces_train,load_file,save_file):
    ninstances = df.shape[0]

    # make train dataset in arff
    with open(load_file, 'r') as file:
        dataset = file.readlines()
        metadata_rows = len(dataset) - ninstances
        dataset_train = dataset[:metadata_rows]
        onlydata = dataset[metadata_rows:]
        dataset_train += [onlydata[irow] for irow in indeces_train]
    with open(save_file, 'w') as file:
        file.writelines(dataset_train)

def read_csvfile(source):
    with open(source, 'r') as csvfile:
        readfile = csv.reader(csvfile, delimiter='\t')
        results = [row for row in readfile if len(row) > 0]
    return results


def write_file_dssd(listofthings, file2write):
    with open(file2write, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(listofthings)


def findnumber(item):
    for i in item.split():
        try:
            # trying to convert i to float
            number = float(i)
            # break the loop if i is the first string that's successfully converted
            break
        except:
            continue
    return number


# estimate in unseed data functions
def decision_pattern(pattern, x):
    decision = True
    for nit in range(pattern["nitems"]):
        type = pattern["type"][nit]
        column = pattern["column"][nit]
        subset = pattern["subset"][nit]
        decision &= belongingtest[type](column, subset, x)
    return decision


def belongingtest_numeric(column, subset, x):
    if subset[0] == np.NINF:
        partial_decision = (x[column] > subset[0]) & (x[column] < subset[1])
    elif subset[1] == np.inf:
        partial_decision = (x[column] > subset[0]) & (x[column] < subset[1])
    return partial_decision


def belongingtest_binary(column, subset, x):
    partial_decision = x[column] == subset
    return partial_decision


belongingtest = {
    "numeric": belongingtest_numeric,
    "binary": belongingtest_binary,
    "nominal": belongingtest_binary
}


def kullback1(value, meanval, var):
    k = 0.5 * log2_0(var) + 0.5 * (value - meanval) ** 2 / var * log2_0(exp(1))
    return k


def kullbackleibler(value, mean1, var1, mean2, var2):
    k1 = kullback1(value, mean1, var1)
    k2 = kullback1(value, mean2, var2)
    kl = k2 - k1
    return kl


def numeric_discovery_measures(subgroup_bitarray, X,Y):
    nrules = len(subgroup_bitarray)
    nrows= X.shape[0]
    wkl_supp, wkl_usg, wkl_sum = np.zeros(nrules), np.zeros(nrules), np.zeros(nrules)
    wacc_supp, wacc_usg = np.zeros(nrules), np.zeros(nrules)
    support, usage = np.zeros(nrules), np.zeros(nrules)
    std_rules = [np.std(colvalues[bitset2indexes(subgroup_bitarray[0])]) for colname, colvalues in  Y.iteritems()]
    std_rulesalternative = []
    data_mean = [np.mean(colvalues) for name, colvalues in Y.items()]
    data_var = [np.var(colvalues) for name, colvalues in Y.items()]
    tid_covered = mpz()
    list_bitsets = []
    number_targets = Y.shape[1]
    for r in range(nrules):
        tid_support = subgroup_bitarray[r]
        list_bitsets.append(tid_support)
        tid_usage = tid_support & ~ tid_covered
        tid_covered = tid_covered | tid_support
        aux_bitset = xmpz(tid_support)
        idx_bits = list(aux_bitset.iter_set())
        values_support = Y.iloc[idx_bits, :].values
        aux_bitset = xmpz(tid_usage)
        idx_bits = list(aux_bitset.iter_set())
        values_usage = Y.iloc[idx_bits, :].values
        support[r] = values_support.shape[0]
        usage[r] = values_usage.shape[0]
        wkl_supp[r] = numeric_single2multitargets_function(kullbackleibler_gaussian_paramters, data_mean, data_var,
                                                           values_support, number_targets)
        wkl_usg[r] = numeric_single2multitargets_function(kullbackleibler_gaussian_paramters, data_mean, data_var,
                                                          values_usage, number_targets)
        #std_rulesalternative.append(np.std(values_usage) if values_usage.shape[0] > 1)
        wacc_supp[r] = numeric_single2multitargets_function(wracc_numeric, data_mean, data_var, values_support,
                                                            number_targets)
        wacc_usg[r] = numeric_single2multitargets_function(wracc_numeric, data_mean, data_var, values_usage,
                                                           number_targets)

    wkl_sum = sum(wkl_usg)
    #  Average them all!!!!
    measures = dict()
    measures["avg_supp"] = np.mean(support)
    measures["wkl_supp"] = np.mean(wkl_supp)

    measures["avg_usg"] = np.mean(usage)
    measures["wkl_usg"] = np.mean(wkl_usg)

    measures["wacc_supp"] = np.mean(wacc_supp)
    measures["wacc_usg"] = np.mean(wacc_usg)

    measures["jacc_avg"], jaccard_matrix = jaccard_index_model(list_bitsets)
    measures["n_rules"] = nrules
    measures["wkl_sum"] = wkl_sum
    measures["wkl_sum_norm"] = wkl_sum/X.shape[0]

    measures["wacc_supp_sum"] = np.sum(wacc_supp)
    measures["wacc_usg_sum"] = np.sum(wacc_usg)

    measures["std_rules"] = np.mean(std_rules)
    measures["top1_std"] = std_rules[0]

    return measures

def nominal_discovery_measures(default_prob_per_class,subgroup_bitarray, X,Y):
    nrules = len(subgroup_bitarray)
    nusage_fail = 0 # number of rules that the usage fails
    nrows= X.shape[0]
    data_prob_class = default_prob_per_class
    wkl_supp,wkl_usg,wkl_sum = np.zeros(nrules), np.zeros(nrules), np.zeros(nrules)
    wacc_supp, wacc_usg = np.zeros(nrules),np.zeros(nrules)
    support, usage = np.zeros(nrules),np.zeros(nrules)
    tid_covered =  mpz()
    list_bitsets = []
    number_targets = len(data_prob_class)
    for r, bitarray in enumerate(subgroup_bitarray):
        tid_support = bitarray
        list_bitsets.append(tid_support)
        tid_usage = tid_support & ~ tid_covered
        tid_covered = tid_covered | tid_support
        aux_bitset = xmpz(tid_support)
        idx_bits = list(aux_bitset.iter_set())
        values_support  = Y.iloc[idx_bits, :].values
        aux_bitset = xmpz(tid_usage)
        idx_bits = list(aux_bitset.iter_set())
        values_usage = Y.iloc[idx_bits, :].values
        support[r] = values_support.shape[0]
        usage[r] = values_usage.shape[0]
        wkl_supp[r], wacc_supp[r] = wkl_wracc(data_prob_class,values_support,nrows, number_targets)
        if usage[r] != 0:
            wkl_usg[r], wacc_usg[r] = wkl_wracc(data_prob_class,values_usage,nrows, number_targets)
        else:
            nusage_fail +=1

    wkl_sum = sum(wkl_usg)
    #  Average them all!!!!
    measures = dict()
    measures["avg_supp"] = np.mean(support)
    measures["wkl_supp"] = np.mean(wkl_supp)

    measures["avg_usg"] = np.sum(usage)/(nrules-nusage_fail)
    measures["wkl_usg"] = np.sum(wkl_usg)/(nrules-nusage_fail)

    measures["wacc_supp"] = np.mean(wacc_supp)
    measures["wacc_usg"] = np.sum(wacc_usg)/(nrules-nusage_fail)


    measures["jacc_avg"], jaccard_matrix = jaccard_index_model(list_bitsets)
    measures["n_rules"] = nrules-nusage_fail
    #measures["avg_items"] = sum([len(sg.pattern) for sg in rulelist.subgroups]) / rulelist.number_rules
    measures["wkl_sum"] = wkl_sum
    measures["wkl_sum_norm"] = wkl_sum/X.shape[0]

    measures["wacc_supp_sum"] = np.sum(wacc_supp)
    measures["wacc_usg_sum"] = np.sum(wacc_usg)
    return measures

def info4prediction(df, number_targets):
    #preprocessing dataset info
    columnames = df.columns[:-number_targets]
    typevar = ["numeric" if is_numeric_dtype(df[col]) else "nominal" for col in columnames]
    limits = []
    for icol,col in enumerate(columnames):
        if typevar[icol] == "numeric":
            minval = min(df[col])
            maxval = max(df[col])
            limits.append([minval,maxval])
        elif typevar[icol] == "nominal":
            categories = np.unique(df[col])
            limits.append(categories)
        else:
            print("went wrong")
    return columnames, typevar, limits


def make_patterns4prediction(descriptions,columnames,typevar,limits):
    nitems = []
    pattern4prediction = []
    for row in descriptions[1:]:
        #count items
        nitems.append(1+row[0].count("&&"))
        # find pattern descritpion
        subsetdefinition = {"type":[],"var_name": [],
                            "subset":[],
                            "column":  [],"nitems" : 0}
        rowsplit = row[0].split(";")
        pattern = rowsplit[-1].split("&&")
        #print(description)
        for item in pattern:
            for icol,col in enumerate(columnames):
                if col in item:
                    if typevar[icol] == "numeric":
                        number = findnumber(item)
                        if ">" in item:
                            subset = [number,np.inf]
                        elif "<" in item:
                            subset = [np.NINF,number]
                        else:
                            print("went wrong")
                    elif typevar[icol] == "nominal":
                        for cat in limits[icol]:
                            if cat in item:
                                subset = cat
                    else:
                        print("went wrong")
                    subsetdefinition["type"].append(typevar[icol])
                    subsetdefinition["var_name"].append(col)
                    subsetdefinition["subset"].append(subset)
                    subsetdefinition["column"].append(icol)
                    subsetdefinition["nitems"] += 1
        pattern4prediction.append(subsetdefinition)
    return pattern4prediction

def findbitsets(patterns4prediction,X,Y):
    indeces_subgroups = [[] for pattern in patterns4prediction]
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(Y, pd.DataFrame):
        Y = Y.values

    # find indices
    for ix,x in enumerate(X):
        for nr in range(len(patterns4prediction)):
            decision = decision_pattern(patterns4prediction[nr],x)
            if decision:
                indeces_subgroups[nr].append(ix)
    # clean the empty ones
    indeces_subgroups = [indices for indices in indeces_subgroups if indices]

    # pass to bitsets
    bitsets_subgroups = [indexes2bitset(indices) for indices in indeces_subgroups]
    return bitsets_subgroups