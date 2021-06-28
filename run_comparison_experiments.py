###############################################################################
#
#
#                           Redo paper experiments experiments
#
#
###############################################################################
"""
Run our proposed algorithm Robust Subgroup Discoverer (RSD) and all the other subgroup set discovery algorithms
List of algorithms:
- RSD
- Top-k (from the DSSD software)
- sequencial-covering or seq-cover (from the DSSD software)
- DSSD (from the DSSD software)
- CN2-SD (from Orange software)
- FSSD
- MCTS4DM

"""


# univariate nominal target
from otheralgorithms.runGenericSSD import runOtherSSDalgorithms
target = "single-nominal"
list_datasets = ["sonar","haberman","breastCancer","australian","TicTacToe","german","chess","mushrooms","magic",
                 "adult","iris","balance","CMC","page-blocks","nursery","automobile","glass","dermatology","kr-vs-k",
                 "abalone"]
exec(open("./reproducibility/runRSD-single-nominal.py").read()) # RSD experiments
runOtherSSDalgorithms(target, 'top-k', list_datasets,arff_algorithm=True)
runOtherSSDalgorithms(target, 'seq-cover', list_datasets,arff_algorithm=True)
runOtherSSDalgorithms(target, 'DSSD', list_datasets,arff_algorithm=True)
runOtherSSDalgorithms(target, 'CN2SD-entro', list_datasets,arff_algorithm=False)
runOtherSSDalgorithms(target, 'CN2SD-wracc', list_datasets,arff_algorithm=False)
#runOtherSSDalgorithms(task, 'FSSD', list_datasets,arff_algorithm=False)
#runOtherSSDalgorithms(task, 'MCTS4DM', list_datasets,arff_algorithm=True)

#########################################################################
# univariate numeric target
target = "single-numeric"

list_datasets = ["baseball","autoMPG8","dee","ele-1","forestFires","concrete",\
                "treasury","wizmir","abalone","puma32h","ailerons","elevators",\
                "bikesharing","california","house"]

exec(open("./reproducibility/runRSD-single-numeric.py").read()) # univariate numeric target
runOtherSSDalgorithms(target, 'top-k', list_datasets,arff_algorithm=True)
runOtherSSDalgorithms(target, 'seq-cover', list_datasets,arff_algorithm=True)

#########################################################################
# multivariate nominal targets
target = "multi-nominal"
list_datasets = ['birds','CAL500','Corel5k','emotions','flags','genbase','mediamill','scene','yeast']

exec(open("./reproducibility/runRSD-multi-nominal.py").read())
runOtherSSDalgorithms(target, "top-k", list_datasets)
runOtherSSDalgorithms(target, 'seq-cover', list_datasets)

#########################################################################
# multivariate numeric targets
target = "multi-numeric"
list_datasets = ['andro','atp1d','atp7d','edm','enb','jura','oes10','oes97','osales','rf1','rf2','scm1d','scm20d',
                 'scpf','sf1','sf2','slump','wq']
exec(open("./reproducibility/runRSD-multi-numeric.py").read()) # multivariate numeric targets
runOtherSSDalgorithms(target, "top-k", list_datasets)
runOtherSSDalgorithms(target, 'seq-cover', list_datasets)
