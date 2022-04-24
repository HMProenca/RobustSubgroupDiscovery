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
exec(open("./reproducibility/runRSD-single-nominal-generalisation.py").read()) # RSD experiments

#########################################################################
# univariate numeric target
#exec(open("./reproducibility/runRSD-single-numeric-generalisation.py").read()) # univariate numeric target

