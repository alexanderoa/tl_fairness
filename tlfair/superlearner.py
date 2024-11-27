import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, KFold

#adapted from https://machinelearningmastery.com/super-learner-ensemble-in-python/
class SuperLearnerClassifier():
    def __init__(self, models=None, folds=10) -> None:
        if models is None:
            self.models = [
                LogisticRegression(solver='liblinear'),
                HistGradientBoostingClassifier(),
                KNeighborsClassifier(),
                SVC(probability=True),
                RandomForestClassifier(),
                MLPClassifier()
            ]
        self.folds = folds
        pass
    
    def fit_base_models(self, x, y):
        for m in self.models:
            m.fit(x,y)
    
    def kfold_predictions(self, x, y):
        meta_x = []
        meta_y = []
        kfold = KFold(n_splits=self.folds, shuffle=True)
        for tr_idx, te_idx in kfold.split(x):
            fold_ypreds = []
            xtr, xte = np.array(x)[tr_idx], np.array(x)[te_idx]
            ytr, yte = np.array(y)[tr_idx], np.array(y)[te_idx]
            meta_y.extend(yte)

            for m in self.models:
                m.fit(xtr, ytr)
                ypreds = m.predict_proba(xte)[:,1]
                fold_ypreds.append(ypreds.reshape((len(ypreds),1)))
            
            meta_x.append(np.hstack(fold_ypreds))
        return np.vstack(meta_x), np.array(meta_y)
            

    def fit(self, x, y):
        meta_x, meta_y = self.kfold_predictions(x, y)
        self.lm = LogisticRegression(solver='liblinear').fit(meta_x, meta_y)
        self.fit_base_models(x,y)
        return self

    def predict(self, x):
        meta_x = []
        for m in self.models:
            ypreds = m.predict_proba(x)[:,1]
            meta_x.append(ypreds.reshape((len(ypreds),1)))
        meta_x = np.hstack(meta_x)
        return self.lm.predict(meta_x)
    
    def predict_proba(self, x):
        meta_x = []
        for m in self.models:
            ypreds = m.predict_proba(x)[:,1]
            meta_x.append(ypreds.reshape((len(ypreds),1)))
        meta_x = np.hstack(meta_x)
        return self.lm.predict_proba(meta_x)
    