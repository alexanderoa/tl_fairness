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

adult = pd.read_csv(
    "https://raw.githubusercontent.com/socialfoundations/folktables/main/adult_reconstruction.csv"
)
thres = 50000
data = adult.copy()

target = (data["income"] > thres).astype(int)
data = data.drop(columns=['income'])

xtr, xte, ytr, yte = train_test_split(data, target, test_size=0.40, random_state=123)
gtr = (xtr['gender'] == 'Male').astype(np.int8)
gte = (xte['gender'] == 'Male').astype(np.int8)

continuous = ["hours-per-week", "age", "capital-gain", "capital-loss"]
categorical = []
label_encoders = {}

xtr = xtr.drop(columns=['education-num', 'gender'])
xte = xte.drop(columns=['education-num', 'gender'])

for col in xtr.columns:
    if col in continuous:
        continue
    categorical.append(col)
    enc = LabelEncoder()
    label_encoders[col] = enc.fit(xtr[col])
    xtr[col] = enc.transform(xtr[col])
    xte[col] = enc.transform(xte[col])

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
        metric = metrics[i]
    )
    results['inference'][title] = inference
    results['importance'][title] = importance

with open('adult_results.pkl', 'wb') as f:
    pickle.dump(results, f)