import numpy as np
import scipy as sp
from sklearn.ensemble import GradientBoostingClassifier
import sys
import pickle

sys.path.append('..')
from tlfair.metrics import *
from tlfair.superlearner import *
from tlfair.knncmi import *
from experiments.cmi_helper import *

weights = [0, 1, 2, 3, 4]
sizes = [500, 1000, 2500, 5000]
n_truth = 10000
sims = 20
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
        results = results.append(here)

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
    compare_results = compare_results.append(res)

compare_results.to_csv('cmi_compare.csv')
pickle.dump(truth_dict, "truth_dict.pkl")