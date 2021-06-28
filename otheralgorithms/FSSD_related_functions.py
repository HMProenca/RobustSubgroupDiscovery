import itertools
from os.path import splitext, basename
from time import time
from otheralgorithms.FSSD.FSSDCODE import find_top_k_subgroups_general_precall
import numpy as np

from data.csvProcessing import readCSVwithHeader
from RSD.util.bitset_operations import indexes2bitset


# does not run "sonar", "german", "adult"
def run_FSSD_wrapper(dataset, attributes, class_attribute, types, depthmax):
    offset = 0
    nb_attributes = len(attributes)
    timebudget = 3600
    top_k = 1000
    wanted_label = dataset[0]["class"]
    attributes = attributes[offset:offset + nb_attributes]
    types = types[offset:offset + nb_attributes]
    timespent = time()
    pattern_setoutput, pattern_union_info, top_k_returned, header_returned = \
        find_top_k_subgroups_general_precall(dataset, attributes, types, class_attribute, \
                                             wanted_label, top_k, 'fssd', False, timebudget, depthmax)
    timespent = time() - timespent

    # print (top_k_returned[-1])
    range_attributes = []
    for ia, a in enumerate(attributes):
        colvals = [row[a] for row in dataset]
        if types[ia] == "numeric":
            maxval = max(colvals)
            minval = min(colvals)
            range_attributes.append([minval, maxval])
        elif types[ia] == "nominal":
            range_attributes.append(list(set(colvals)))
        elif types[ia] == "simple":
            range_attributes.append(list(set(colvals)))

    c_values = list(set([row["class"] for row in dataset]))
    count_cl = [0 for c in c_values]
    for row in dataset:
        for ic, c in enumerate(c_values):
            if row["class"] == c:
                count_cl[ic] += 1

    subgroup_sets = []
    items = []
    rules_supp = []
    nitems = []
    for pat in pattern_setoutput:
        # items
        nitemsaux = 0
        for ia, a in enumerate(attributes):
            # print(pat[0][ia])
            print("pattern: " + str(set(pat[0][ia])) + "  range: " + str(set(range_attributes[ia])))
            if not set(pat[0][ia]) >= set(range_attributes[ia]):
                nitemsaux += 1
        nitems.append(nitemsaux)
        subgroup_index = pat[1]["support_full"]
        aux_supp = [0 for c in c_values]
        for idx in subgroup_index:
            for ic, c in enumerate(c_values):
                if dataset[idx]["class"] == c:
                    aux_supp[ic] += 1
        rules_supp.append(aux_supp)
        subgroup_sets.append(indexes2bitset(subgroup_index))

    return nitems, subgroup_sets, timespent

