import numpy as np
import scipy as sp
from sklearn.ensemble import GradientBoostingClassifier
import sys
import pickle

sys.path.append('..')
from tl_fairness.tlfair.metrics import *
from tl_fairness.tlfair.superlearner import *
from tl_fairness.tlfair.knncmi import *
from tl_fairness.experiments.cmi_helper import *

weights = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
sizes = [500, 750, 1000, 1750, 2500, 3750, 5000, 7500, 10000]
n_truth = 1000000
sims = 100
rng = np.random.default_rng()
results = pd.DataFrame()
truth_dict = dict()

for i in range(len(weights)):
    c = weights[i]
    truth = cmi_ground_truth(
        c = c,
        d = 3,
        n = n_truth,
        rng = rng
    )
    truth_dict[c] = truth
    for j in range(len(sizes)):
        s = sizes[j]
        r = cmi_coverage_sim(
            n = s,
            c = c,
            ground_truth = truth,
            rng = rng,
            sims = sims
        )
        here = pd.DataFrame(
            {
                "sample_size" : [s],
                "c" : [c],
                "error" : [r[1]],
                "coverage" : [r[0]]
            }
        )
        results = pd.concat([results,here])

results.to_csv('cmi_coverage.csv')

compare_results = pd.DataFrame()

for i in range(len(sizes)):
    s = sizes[i]
    res = cmi_compare(
        n = s,
        params = weights,
        repeats = 10,
        rng = rng
    )
    compare_results = pd.concat([compare_results,res])

compare_results.to_csv('cmi_compare.csv')
with open('truth_dict.pkl', 'wb') as f:
    pickle.dump(truth_dict, f)