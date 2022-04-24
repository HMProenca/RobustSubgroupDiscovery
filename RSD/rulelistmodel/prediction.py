from RSD.rulelistmodel.categoricalmodel.prediction_categorical import point_value_categorical
from RSD.rulelistmodel.gaussianmodel.prediction_gaussian import point_value_gaussian
from RSD.rulelistmodel.rulesetmodel import RuleSetModel
import pandas as pd
import numpy as np
from functools import reduce

from math import log2


point_value_estimation = {
    "gaussian" : point_value_gaussian,
    "categorical": point_value_categorical
}

def predict_rulelist(X : pd.DataFrame, rulelist: RuleSetModel):
    if X is not pd.DataFrame: Exception('X needs to be a DataFrame')
    n_predictions = X.shape[0]
    n_targets = rulelist.default_rule_statistics.number_targets
    instances_covered = np.zeros(n_predictions, dtype=bool)
    predictions = np.empty((n_predictions,n_targets),dtype=object)
    for subgroup in rulelist.subgroups:
        instances_subgroup = ~instances_covered &\
                             reduce(lambda x,y: x & y, [item.activation_function(X).values for item in subgroup.pattern])
        predictions[instances_subgroup,:] = point_value_estimation[rulelist.target_model](subgroup.statistics)
        instances_covered |= instances_subgroup

    # default rule
    predictions[~instances_covered, :] = point_value_estimation[rulelist.target_model](rulelist.default_rule_statistics)
    if n_targets == 1:
        predictions = predictions.flatten()
    return predictions

def swkl_subgroup_discovery(X : pd.DataFrame, Y:pd.DataFrame, rulelist: RuleSetModel):
    """ Compute the Sum of Weighted Kullback-Leibler divergence

    TODO: it only works for single target variable
    """
    if X is not pd.DataFrame: Exception('X needs to be a DataFrame')
    n_predictions = X.shape[0]
    n_targets = rulelist.default_rule_statistics.number_targets
    instances_covered = np.zeros(n_predictions, dtype=bool)
    predictions = np.empty((n_predictions,n_targets),dtype=object)
    swkl = 0
    for subgroup in rulelist.subgroups:
        instances_subgroup = reduce(lambda x,y: x & y, [item.activation_function(X).values for item in subgroup.pattern])
        instances_subgroup_in_list = ~instances_covered & instances_subgroup
        for target in rulelist.default_rule_statistics.number_classes.keys():
            swkl += wkl_nominal(Y[instances_subgroup],target, subgroup.statistics, rulelist.default_rule_statistics)
        instances_covered |= instances_subgroup_in_list

    # default rule does not require swkl as it
    return swkl

def wkl_nominal(Y:pd.DataFrame,target: str, subgroup_statistics, defaultrule_statsitics):

    kl= 0
    epsilon = 0.5 # adding Jeffreys prior for unseen data
    counts_subgroup = subgroup_statistics.usage_per_class[target]
    usage_subgroup = subgroup_statistics.usage
    n_classes = subgroup_statistics.number_classes[target]

    prob_default_classes = defaultrule_statsitics.prob_per_classes[target]

    n_points = Y[target].shape[0]
    for val in Y[target].values:
        prob_subgroup = (counts_subgroup.get(val)+epsilon)/(usage_subgroup + n_classes*epsilon)
        prob_default = prob_default_classes.get(val)
        kl += prob_subgroup*log2(prob_subgroup/prob_default)

    wkl = n_points*kl
    return wkl