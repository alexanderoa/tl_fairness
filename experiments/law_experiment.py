import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
import sys
import pickle

sys.path.append('..')
from tl_fairness.tlfair.metrics import *
from tl_fairness.tlfair.superlearner import *

law = pd.read_csv(
    "https://raw.githubusercontent.com/tailequy/fairness_dataset/refs/heads/main/experiments/data/law_school_clean.csv"
)

target = np.array(law[['pass_bar']]).ravel()
data = law.copy().drop(columns=['pass_bar'])
race_encoder = LabelEncoder().fit(data['race'])
data['race'] = race_encoder.transform(data['race'])

xtr, xte, ytr, yte = train_test_split(data, target, test_size=0.40, random_state=123)
gtr = np.array(xtr[['race']]).ravel()
gte = np.array(xte[['race']]).ravel()

xtr = xtr.drop(columns = ['race'])
xte = xte.drop(columns = ['race'])

metrics = [
    parity,
    prob_parity,
    opportunity,
    prob_opportunity,
    cmi
]
metric_title = [
    'parity',
    'prob_parity',
    'opportunity',
    'prob_opp',
    'cmi'
    ]
results = dict()
results['inference'] = dict()
results['importance'] = dict()
print('Beginning Law Experiment')
for i in range(len(metrics)):
    metric = metrics[i]
    title = metric_title[i]
    if title == 'cmi':
        outcome = HistGradientBoostingClassifier()
    else:
        outcome = SuperLearnerClassifier()
    
    inference = metric(
        xtr = xtr,
        xte = xte,
        ytr = ytr,
        yte = yte,
        gtr = gtr,
        gte = gte,
        outcome = outcome,
        propensity = SuperLearnerClassifier(),
    )
    importance = perm_importance(
        xtr = xtr,
        xte = xte,
        ytr = ytr,
        yte = yte,
        gtr = gtr,
        gte = gte,
        outcome = HistGradientBoostingClassifier(),
        propensity= HistGradientBoostingClassifier(),
        metric = metrics[i],
        n_samples = 1000
    )
    results['inference'][title] = inference
    results['importance'][title] = importance
    
with open('law_results.pkl', 'wb') as f:
    pickle.dump(results, f)