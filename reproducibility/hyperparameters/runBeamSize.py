# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 16:10:16 2020

@author: Hugo Manuel Proenca
"""

from reproducibility.hyperparameters.hyperparameter_experiments import run_hyper_exps

###############################################################################
# Beam size experiment
###############################################################################
# user configuration
delim = ','
task_name = "discovery"
beamsize_list = [1,3,10,100,300,1000]
hyperparameter = "beam_width"

"""
Nominal target data
"""
list_datasets = ["sonar","haberman","breastCancer","australian","TicTacToe","german","chess","mushrooms","magic",
                 "adult","iris","balance","CMC","page-blocks","nursery","automobile","glass","dermatology","kr-vs-k",
                 "abalone"]
target_type = "categorical"
run_hyper_exps(hyperparameter, beamsize_list, list_datasets, task_name, target_type)

"""
Numeric target data
"""
list_datasets = ["baseball","autoMPG8","dee","ele-1","forestFires","concrete",\
                "treasury","wizmir","abalone","puma32h","ailerons","elevators",\
                "bikesharing","california","house"]
target_type = "gaussian"
run_hyper_exps(hyperparameter, beamsize_list, list_datasets, task_name, target_type)
