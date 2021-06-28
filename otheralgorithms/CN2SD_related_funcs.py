#######################################################################################
#  CN2-SD
##############################################
from otheralgorithms.FSSD.FSSDCODE import transform_dataset, transform_dataset_to_attributes
from otheralgorithms.FSSD.util.csvProcessing import writeCSVwithHeader, readCSVwithHeader
import Orange
from copy import deepcopy
from time import time

from RSD.util.bitset_operations import indexes2bitset


def run_CN2SD_wrapper(dataset,attributes,types,class_attribute,beam_width,depthmax, quality):
    wanted_label = dataset[0]["class"]
    # dataset,header=readCSVwithHeader(file,numberHeader=[a for a,t in zip(attributes,types) if t=='numeric'],delimiter=delimiter)
    new_dataset = deepcopy(dataset)
    new_dataset, positive_extent, negative_extent, alpha_ratio_class, _ = transform_dataset(dataset, attributes,
                                                                                            class_attribute,
                                                                                            wanted_label)
    new_dataset.insert(0, {a: 'c' if t == 'numeric' else 'd' for a, t in
                           list(zip(attributes, types)) + [('class', 'class')]})
    new_dataset.insert(1, {a: '' if a != 'class' else 'class' for a in attributes + ['class']})
    writeCSVwithHeader(new_dataset, './otheralgorithms/tmpForOrange.csv', selectedHeader=attributes + ['class'], delimiter='\t',
                       flagWriteHeader=True)
    data = Orange.data.Table('./otheralgorithms/tmpForOrange.csv')
    # print(data)
    timespent = time()
    # ordered! CN2SDLearner
    learner = Orange.classification.rules.CN2SDUnorderedLearner()

    if quality == 'entropy':
        learner.rule_finder.quality_evaluator = Orange.classification.rules.EntropyEvaluator()
    elif quality == 'wracc':
        learner.rule_finder.quality_evaluator =Orange.classification.rules.WeightedRelativeAccuracyEvaluator()
    # learner = Orange.classification.rules.CN2SDLearner()
    learner.gamma = 0.
    # learner.evaluator = "Evaluator_Entropy"
    learner.rule_finder.search_algorithm.beam_width = beam_width

    # continuous value space is constrained to reduce computation time

    learner.rule_finder.search_strategy.constrain_continuous = True

    # found rules must cover at least 15 examples
    learner.rule_finder.general_validator.min_covered_examples = max(int(15), 1.)

    # learner.rule_finder.general_validator.min_covered_examples = max(int(float(len(positive_extent))/10),1.)

    # found rules may combine at most 2 selectors (conditions)
    learner.rule_finder.general_validator.max_rule_length = depthmax

    classifier = learner(data)
    timespent = time() - timespent

    del classifier.rule_list[-1]
    top_quality = []
    # import inspect
    # inspect.getmembers(learner, lambda a:not(inspect.isroutine(a)))
    # inspect.getmembers(row, lambda a:not(inspect.isroutine(a)))
    subgroup_sets = []
    rules_supp = []
    nitems = []
    for i, row in enumerate(classifier.rule_list):
        s = str(row)
        nitems.append(1 + s.count("AND"))
        subgroup_biset = row.covered_examples
        subgroup_index= set(i for i, x in enumerate(subgroup_biset) if x == True)
        subgroup_sets.append(indexes2bitset(subgroup_index))
        rules_supp.append(row.curr_class_dist.tolist())

    return nitems, subgroup_sets, timespent