# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 16:19:15 2020

@author: Hugo Manuel Proenca
"""

from reproducibility.hyperparameters.hyperparameter_experiments import run_hyper_exps


###############################################################################
# maximum search depth experiment
###############################################################################
# user configuration
delim = ','
task_name = "discovery"
hyperparameter = "max_depth"
maxdepth_list = [1,2,3,4,5,6,7,8,9,10,11]


###############################################################################
# Beam size experiment
###############################################################################
# user configuration

"""
Nominal target data
"""
list_datasets = ["sonar","haberman","breastCancer","australian","TicTacToe","german","chess","mushrooms","magic",
                 "adult","iris","balance","CMC","page-blocks","nursery","automobile","glass","dermatology","kr-vs-k",
                 "abalone"]
target_type = "categorical"
run_hyper_exps(hyperparameter, maxdepth_list, list_datasets, task_name, target_type)

"""
Numeric target data
"""
list_datasets = ["baseball","autoMPG8","dee","ele-1","forestFires","concrete",\
                "treasury","wizmir","abalone","puma32h","ailerons","elevators",\
                "bikesharing","california","house"]
target_type = "gaussian"
run_hyper_exps(hyperparameter, maxdepth_list, list_datasets, task_name, target_type)