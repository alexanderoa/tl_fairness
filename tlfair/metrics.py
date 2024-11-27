import numpy as np
import pandas as pd
import random
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

def parity(
    xtr,
    xte, 
    ytr, 
    yte,
    gtr,
    gte,
    outcome,
    propensity=None):

    outcome = outcome.fit(xtr, ytr)
    phi0 = -1/np.mean(gte==0) * outcome.predict(xte.iloc[np.where(gte==0)[0],:])
    phi1 = 1/np.mean(gte==1) * outcome.predict(xte.iloc[np.where(gte==1)[0],:])
    phi = np.hstack([phi0, phi1])
    est = np.mean(phi)
    eif = phi - np.mean(phi)
    ci = (est - 1.96*np.sqrt(np.var(eif)/len(eif)), est + 1.96*np.sqrt(np.var(eif)/len(eif)))
    return est, ci

def prob_parity(
    xtr,
    xte, 
    ytr, 
    yte,
    gtr,
    gte,
    outcome,
    propensity=None):
    
    outcome = outcome.fit(xtr, ytr)
    propensity = propensity.fit(xtr,gtr)
    m_probs = propensity.predict_proba(xte)
    f_probs = outcome.predict_proba(xte)[:,1]
    phi0 = -1/(np.mean(gte==0)) * (m_probs[:,0]*((yte==1) -f_probs) + (gte==0)*f_probs)
    phi1 = 1/(np.mean(gte==1)) * (m_probs[:,1]*((yte==1) -f_probs) + (gte==1)*f_probs)
    phi = phi0 + phi1
    est = np.mean(phi)
    eif = phi - (np.mean(phi1) + np.mean(phi0))
    ci = (est - 1.96*np.sqrt(np.var(eif)/len(eif)), est + 1.96*np.sqrt(np.var(eif)/len(eif)))
    return est, ci

def cmi(
    xtr,
    xte, 
    ytr, 
    yte,
    gtr,
    gte,
    outcome,
    propensity=None):

    label = np.zeros(shape=(len(gtr),)).astype(np.int8)
    label[np.intersect1d(np.where(gtr==0), np.where(ytr==0))] = 0
    label[np.intersect1d(np.where(gtr==1), np.where(ytr==0))] = 1
    label[np.intersect1d(np.where(gtr==0), np.where(ytr==1))] = 2
    label[np.intersect1d(np.where(gtr==1), np.where(ytr==1))] = 3
    
    outcome = CalibratedClassifierCV(outcome, cv=3).fit(xtr, label)

    est_vec = np.zeros(len(gte))
    lte = np.zeros(shape=(len(gte),)).astype(np.int8)
    lte[np.intersect1d(np.where(gte==0), np.where(yte==0))] = 0
    lte[np.intersect1d(np.where(gte==1), np.where(yte==0))] = 1
    lte[np.intersect1d(np.where(gte==0), np.where(yte==1))] = 2
    lte[np.intersect1d(np.where(gte==1), np.where(yte==1))] = 3
    tol = 1e-2
    preds = outcome.predict_proba(xte)
    for i in range(len(preds)):
        numerator = preds[i, lte[i]]
        if lte[i] == 0:
            denominator = (preds[i,0]+preds[i,2]) * (preds[i,0]+preds[i,1])
        elif lte[i] == 1:
            denominator = (preds[i,1]+preds[i,3]) * (preds[i,0]+preds[i,1])
        elif lte[i] == 2:
            denominator = (preds[i,0]+preds[i,2]) * (preds[i,2]+preds[i,3])
        elif lte[i] == 3:
            denominator = (preds[i,1]+preds[i,3]) * (preds[i,2]+preds[i,3])
        est_vec[i] = np.log((numerator)/(denominator))
    est = np.mean(est_vec)
    eif = (est_vec - est)
    ci = (est - 1.96*np.sqrt(np.var(eif)/len(eif)), est + 1.96*np.sqrt(np.var(eif)/len(eif)))
    return est, ci

def cmi_separate(
    xtr,
    xte, 
    ytr, 
    yte,
    gtr,
    gte,
    outcome,
    propensity=None):

    label = np.zeros(shape=(len(gtr),)).astype(np.int8)
    label[np.intersect1d(np.where(gtr==0), np.where(ytr==0))] = 0
    label[np.intersect1d(np.where(gtr==1), np.where(ytr==0))] = 1
    label[np.intersect1d(np.where(gtr==0), np.where(ytr==1))] = 2
    label[np.intersect1d(np.where(gtr==1), np.where(ytr==1))] = 3
    
    outcome = CalibratedClassifierCV(outcome, cv=3).fit(xtr, label)
    base = LogisticRegression(solver='liblinear')
    y_model = CalibratedClassifierCV(base, cv=3).fit(xtr, ytr)
    base = LogisticRegression(solver='liblinear')
    g_model = CalibratedClassifierCV(base, cv=3).fit(xtr, gtr)

    est_vec = np.zeros(len(gte))
    lte = np.zeros(shape=(len(gte),)).astype(np.int8)
    lte[np.intersect1d(np.where(gte==0), np.where(yte==0))] = 0
    lte[np.intersect1d(np.where(gte==1), np.where(yte==0))] = 1
    lte[np.intersect1d(np.where(gte==0), np.where(yte==1))] = 2
    lte[np.intersect1d(np.where(gte==1), np.where(yte==1))] = 3
    tol = 1e-2
    preds = outcome.predict_proba(xte)
    ypreds = y_model.predict_proba(xte)
    gpreds = g_model.predict_proba(xte)
    for i in range(len(preds)):
        numerator = preds[i, lte[i]]
        if lte[i] == 0:
            denominator = gpreds[i,0] * ypreds[i,0]
        elif lte[i] == 1:
            denominator = gpreds[i,1] * ypreds[i,0]
        elif lte[i] == 2:
            denominator = gpreds[i,0] * ypreds[i,1]
        elif lte[i] == 3:
            denominator = gpreds[i,1] * ypreds[i,1]
        est_vec[i] = np.log((numerator)/(denominator))
    est = np.mean(est_vec)
    eif = (est_vec - est)
    ci = (est - 1.96*np.sqrt(np.var(eif)/len(eif)), est + 1.96*np.sqrt(np.var(eif)/len(eif)))
    return est, ci

def opportunity(
    xtr,
    xte, 
    ytr, 
    yte,
    gtr,
    gte,
    outcome,
    propensity=None):

    outcome = outcome.fit(xtr, ytr)

    yg1 = np.all(
        np.array([gte==1, yte==1]),
        axis = 0
        )
    yg0 = np.all(
        np.array([gte==0, yte==1]),
        axis = 0
        )
    phi0 = -1/np.mean(yg0) * outcome.predict(xte.iloc[np.where(yg0)[0],:])
    phi1 = 1/np.mean(yg1) * outcome.predict(xte.iloc[np.where(yg1)[0],:])
    phi = np.hstack([phi0, phi1])
    est = np.sum(phi)/gte.shape[0]
    eif = phi - est
    ci = (est - 1.96*np.sqrt(np.var(eif)/len(eif)), est + 1.96*np.sqrt(np.var(eif)/len(eif)))
    return est, ci

def prob_opportunity(
    xtr,
    xte, 
    ytr, 
    yte,
    gtr,
    gte,
    outcome,
    propensity=None):
    yg_tr = np.zeros(shape=(len(gtr),)).astype(np.int8)
    yg_tr[np.intersect1d(np.where(gtr==0), np.where(ytr==0))] = 0
    yg_tr[np.intersect1d(np.where(gtr==1), np.where(ytr==0))] = 1
    yg_tr[np.intersect1d(np.where(gtr==0), np.where(ytr==1))] = 2
    yg_tr[np.intersect1d(np.where(gtr==1), np.where(ytr==1))] = 3

    outcome = outcome.fit(xtr, ytr)
    propensity = propensity.fit(xtr, yg_tr)

    yg1 = np.all(
        np.array([gte==1, yte==1]),
        axis = 0
        )
    yg0 = np.all(
        np.array([gte==0, yte==1]),
        axis = 0
        )
    props = propensity.predict_proba(xte)
    preds = outcome.predict_proba(xte)[:,1]
    phi0 = -1/np.mean(yg0) * (props[:,2]*((yte==1) - preds) + yg0 * preds)
    phi1 = 1/np.mean(yg1) * (props[:,3]* ((yte==1) - preds) + yg1 * preds)
    phi = phi0 + phi1
    est = np.mean(phi)
    eif = est - phi
    ci = (est - 1.96*np.sqrt(np.var(eif)/len(eif)), est + 1.96*np.sqrt(np.var(eif)/len(eif)))
    return est, ci


def perm_importance(
    xtr,
    xte, 
    ytr,
    yte,
    gtr, 
    gte,
    metric,
    outcome,
    propensity,
    n_samples = 10,
    rng = None
    ):
    if rng is None:
        rng = np.random.default_rng()

    seq = xtr.columns
    seen = set()
    perms = []
    while len(perms) < n_samples:
        perm = tuple(rng.choice(seq, size=len(seq), replace=False))
        if perm not in seen:
            perms.append(perm)
            seen.add(perm)

    values = []
    importance = {}
    for col in seq:
        importance[col] = 0

    for perm in perms:
        prev = 0
        v = []
        for i in range(len(perm)):
            htr = xtr[list(perm[:i+1])]
            hte = xte[list(perm[:i+1])]
            res = metric(
                xtr = htr,
                xte = hte,
                ytr = ytr,
                yte = yte,
                gtr = gtr,
                gte = gte,
                outcome = outcome,
                propensity = propensity
                )
            est = res[0]
            importance[perm[i]] += (est-prev) / n_samples
            prev = est
            v.append(est)
        values.append(v)
    
    return importance, values, perms