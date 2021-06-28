import os
import shutil
import pandas as pd
from time import time
from subprocess import call,check_output

from otheralgorithms.all_ssd_related_funcs import read_csvfile, write_file_dssd
from RSD.util.bitset_operations import indexes2bitset


def run_DSSD_wrapper(algorithmname, beam_width, number_rules_SSD, datasetname, df, task, depthmax,attribute_names,number_targets):
    if algorithmname == "seq-cover":
        conf_file = read_csvfile('./otheralgorithms/DSSD/bin/tmp_sequential.conf')
    elif algorithmname == "DSSD":
        conf_file = read_csvfile('./otheralgorithms/DSSD/bin/tmp_dssd_diverse.conf')
        conf_file[12] = ['postSelect = ' + str(int(number_rules_SSD.loc[datasetname, "number_rules"]))]
    elif algorithmname == "top-k":
        conf_file = read_csvfile('./otheralgorithms/DSSD/bin/tmp_topk.conf')
        conf_file[12] = ['postSelect = ' + str(int(number_rules_SSD.loc[datasetname, "number_rules"]))]
        nrows = df.shape[0]
        if nrows < 2000 and task == "single-nominal":
            conf_file[14] = ['searchType = ' + "dfs"]
        else:
            conf_file[14] = ['searchType = ' + "beam"]
    else:
        raise Exception("Wrong aglorithm name")

    conf_file[19] = ['beamWidth = ' + str(int(beam_width))]
    conf_file[15] = ['maxDepth = ' + str(min(int(depthmax), 10))]

    if task == "multi-nominal" or task == "single-nominal":
        conf_file[23] = ['measure = WKL']
        # conf_file[24] = ['WRAccMode = 1vsAll']
    elif task == "multi-numeric" or task == "single-numeric":
        conf_file[23] = ['measure = meantest']
        conf_file[24] = ['WRAccMode = 1vsAll']
    else:
        raise Exception("Wrong task name")

    write_file_dssd(conf_file, './otheralgorithms/DSSD/bin/tmp.conf')

    # check if path exists
    if not os.path.exists('.//otheralgorithms//DSSD//xps//dssd'):
        os.makedirs('.//otheralgorithms//DSSD//xps//dssd')
    else:
        shutil.rmtree('.//otheralgorithms//DSSD//xps//dssd')
        os.makedirs('.//otheralgorithms//DSSD//xps//dssd')

    # change target variable file - target variables are at the end!
    name_targets = attribute_names[-number_targets:]
    targets_file = pd.read_csv('./otheralgorithms/DSSD/data/datasets/tmp/emmModel.emm', delimiter="=", header=None)
    targets_file.iloc[1, 1] = ' ' + ','.join([tg_name for tg_name in name_targets])
    targets_file.to_csv('./otheralgorithms/DSSD/data/datasets/tmp/tmp.emm', index=False, sep="=", header=False)

    # run DSSD
    timespent = time()
    os.chdir("./otheralgorithms/DSSD/bin")
    call(["emc64-mt-modified.exe"])
    # call(["dssd64.exe"])
    os.chdir("../../../")
    timespent = time() - timespent
    os.remove("./otheralgorithms/DSSD/data/datasets/tmp/tmp.arff")

    # read output files
    auxfiles = [path for path in os.listdir('./otheralgorithms/DSSD/xps/dssd/')]
    generated_xp = './otheralgorithms/DSSD/xps/dssd/' + auxfiles[-1]  # last one
    timestamp = generated_xp.split('-')[1]
    # find transaction ids of subgroups
    generated_xp_subsets_path = generated_xp + '/subsets'
    all_generated_subgroups_files = [generated_xp_subsets_path + '/' + x
                                     for x in os.listdir(generated_xp_subsets_path)]
    # find descriptions of subgroups
    if algorithmname == "top-k":
        description_files = generated_xp + '/' + "stats1-" + timestamp + ".csv"
    elif algorithmname == "seq-cover":
        description_files = generated_xp + '/' + "stats2-" + timestamp + ".csv"
    elif algorithmname == "DSSD":
        description_files = generated_xp + '/' + "stats3-" + timestamp + ".csv"

    # count number of items per subgroup
    descriptions = read_csvfile(description_files)
    #columnames, typevar, limits = info4prediction(df.iloc[:, :-number_targets], number_targets)
    #patterns4prediction = make_patterns4prediction(descriptions, columnames, typevar, limits)
    # Test dataset
    # nrows_test = Y_test.shape[0]
    # bitsets_subgroups = findbitsets(patterns4prediction,X_test,Y_test)

    nitems = []
    for row in descriptions[1:]:
        # count items
        nitems.append(1 + row[0].count("&&"))

    subgroup_sets_support = []
    subgroup_sets_support_bitset = []
    support_union = set()
    nb_subgroups = 0
    rules_supp = []
    for subgroup_file in all_generated_subgroups_files:
        aux_subgroup = read_csvfile(subgroup_file)[2:]
        subgroup_biset = [row[0] for row in aux_subgroup]
        subgroup_index = set(i for i, x in enumerate(subgroup_biset) if x == '1')
        subgroup_sets_support.append(subgroup_index)
        subgroup_sets_support_bitset.append(indexes2bitset(subgroup_index))
        support = len(subgroup_index)
        rules_supp.append(support)
        nb_subgroups += 1

    return nitems, subgroup_sets_support_bitset, timespent