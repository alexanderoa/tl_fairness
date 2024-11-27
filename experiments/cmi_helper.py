import numpy as np
import scipy as sp
from sklearn.ensemble import GradientBoostingClassifier
import sys

sys.path.append('..')
from tl_fairness.tlfair.metrics import *
from tl_fairness.tlfair.superlearner import *
from tl_fairness.tlfair.knncmi import *

def cmi_sim(
    n,
    c,
    d=3,
    rng=None,
    sep=False
    ):

    n = n //2 #sample splitting
    if rng is None:
        # np.random.seed(123)
        rng = np.random.default_rng()
    z = rng.normal(size=(2*n,d))
    beta = np.array([1,1,1])
    c_prob = c*rng.uniform(size=2*n)
    x_prob = (c_prob + rng.uniform(size=2*n) + 1/(1+np.exp(-z@beta)))/(c+2)
    y_prob = (c_prob + rng.uniform(size=2*n) + 1/(1+np.exp(-z@beta)))/(c+2)
    x = (x_prob > 0.5).astype(np.int8)
    y = (y_prob > 0.5).astype(np.int8)

    

    label = np.zeros(shape=(2*n,)).astype(np.int8)
    label[np.intersect1d(np.where(x==0), np.where(y==0))] = 0
    label[np.intersect1d(np.where(x==1), np.where(y==0))] = 1
    label[np.intersect1d(np.where(x==0), np.where(y==1))] = 2
    label[np.intersect1d(np.where(x==1), np.where(y==1))] = 3

    if sep:
        fn = cmi_separate
    else:
        fn = cmi
        
    res = fn(
        xtr = z[:n,:],
        xte = z[n:,:], 
        ytr = x[:n], 
        yte = x[n:],
        gtr = y[:n],
        gte = y[n:],
        outcome = GradientBoostingClassifier(),
        propensity=None
    )
    return res

def knncmi_sim(
    n,
    c,
    d=3,
    rng=None):
    if rng is None:
        np.random.seed(123)
        rng = np.random.default_rng()
    z = rng.normal(size=(n,d))
    beta = np.array([1,1,1])
    c_prob = c*rng.uniform(size=n)
    x_prob = (c_prob + rng.uniform(size=n) + 1/(1+np.exp(-z@beta)))/(c+2)
    y_prob = (c_prob + rng.uniform(size=n) + 1/(1+np.exp(-z@beta)))/(c+2)

    x = (x_prob > 0.5).astype(np.int8)
    y = (y_prob > 0.5).astype(np.int8)

    data = pd.DataFrame(
        data = {
            'x' : x,
            "y" : y,
            "z1" : z[:,0],
            'z2' : z[:,1],
            'z3' : z[:,2]
        }
    )
    return knncmi(['x'], ['y'], ['z1', 'z2', 'z3'], k=7, data=data)

def cmi_coverage_sim(
    n,
    c,
    ground_truth,
    sims=100,
    fn = cmi_sim,
    rng=None):

    if rng is None:
        rng = np.random.default_rng(123)
    coverage = np.zeros(sims)
    error = 0
    for i in range(sims):
        res = fn(n=n, c=c, rng=rng)
        error += (res[0] - ground_truth)
        if (res[1][0] <= ground_truth) and (res[1][1] >= ground_truth):
            coverage[i] = 1
    return np.mean(coverage), error/sims

def cmi_ground_truth(
    c,
    d,
    n,
    rng):
    z = rng.normal(size=(n,d))
    beta = np.ones(d)
    res = []
    for i in range(n):
        inside = n
        zi = z[i,:]
        
        c_prob = c*rng.uniform(size=inside)
        x_prob = (c_prob + rng.uniform(size=inside) + 1/(1+np.exp(-z@beta)))/(c+2)
        y_prob = (c_prob + rng.uniform(size=inside) + 1/(1+np.exp(-z@beta)))/(c+2)

        x = (x_prob > 0.5).astype(np.int8)
        y = (y_prob > 0.5).astype(np.int8)
        

        p1 = np.mean((x==1)*(y==1)) * np.log(np.mean((x==1)*(y==1)) / (np.mean(x==1)*np.mean(y==1)))
        p2 = np.mean((x==0)*(y==1)) * np.log(np.mean((x==0)*(y==1)) / (np.mean(x==0)*np.mean(y==1)))
        p3 = np.mean((x==1)*(y==0)) * np.log(np.mean((x==1)*(y==0)) / (np.mean(x==1)*np.mean(y==0)))
        p4 = np.mean((x==0)*(y==0)) * np.log(np.mean((x==0)*(y==0)) / (np.mean(x==0)*np.mean(y==0)))
        res.append(p1+p2+p3+p4)

    return(np.mean(res))

def cmi_compare(
    n,
    repeats = 1,
    params = [0.5, 1, 1.25, 1.5, 1.75, 2, 2.5, 3],
    rng = None):

    if rng is None:
        rng = np.random.default_rng()
    
    df = pd.DataFrame()
    for i in range(len(params)):
        cmi_res = []
        sep_res = []
        knn_res = []
        for _ in range(repeats):
            res = cmi_sim(
                c = params[i],
                n = n,
                rng= rng
            )
            cmi_res.append(res[0])

            res = cmi_sim(
                c = params[i],
                n = n,
                rng= rng,
                sep = True
            )
            sep_res.append(res[0])

            res = knncmi_sim(
                c = params[i],
                n = n,
                rng=rng
            )
            knn_res.append(res)

        data = pd.DataFrame(
            {
                "sample size" : [n] * 3,
                "type": ["TL", "TL-sep", "KNN"],
                "c" : [params[i]] * 3,
                "mean" : [np.mean(cmi_res), np.mean(sep_res), np.mean(knn_res)],
                "bottom_five": [np.quantile(cmi_res, 0.05), np.quantile(sep_res, 0.05), np.quantile(knn_res, 0.05)],
                "top_five" : [np.quantile(cmi_res, 0.95), np.quantile(sep_res, 0.95), np.quantile(knn_res, 0.95)]
            }
        )
        df = df.append(data)
    return df