###############################################################################
#
#
#                           Redo paper experiments experiments
#
#
###############################################################################
"""
Run our proposed algorithm Robust Subgroup Discoverer (RSD) to test for generalisation
List of algorithms:
- RSD

"""


# univariate nominal target
from otheralgorithms.runGenericSSD import runOtherSSDalgorithms
target = "single-nominal"

exec(open("./reproducibility/runRSD-single-nominal-generalisation.py").read()) # RSD experiments

#########################################################################
# univariate numeric target
target = "single-numeric"

list_datasets = ["baseball","autoMPG8","dee","ele-1","forestFires","concrete",\
                "treasury","wizmir","abalone","puma32h","ailerons","elevators",\
                "bikesharing","california","house"]

exec(open("./reproducibility/runRSD-single-numeric-generalisation.py").read()) # univariate numeric target

