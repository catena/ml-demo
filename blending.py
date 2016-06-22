# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 17:56:31 2016

@author: vishnu
"""

from __future__ import division
import numpy as np
from sklearn.datasets import make_classification
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

def blend(clfs, blend_clf, n_folds, X, y, X_submission):
    np.random.seed(1010)
    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))
    skf = StratifiedKFold(y, n_folds=5, shuffle=True)
    for j, clf in enumerate(clfs):
        print(j, clf)
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
        for i, (train_idx, test_idx) in enumerate(skf):
            print("Fold", i)
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            clf.fit(X_train, y_train)
            dataset_blend_train[test_idx, j] = clf.predict_proba(X_test)[:,1]
            dataset_blend_test_j[:,i] = clf.predict_proba(X_submission)[:,1]
        score = log_loss(y, dataset_blend_train[:,j])
        print('logloss:', score)
        dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)
    print("Blending.")
    blend_clf.fit(dataset_blend_train, y)
    y_submission = blend_clf.predict_proba(dataset_blend_test)[:,1]
    return y_submission


X, y = make_classification(n_samples=10000, n_features=20, n_informative=2,
                       n_redundant=10, flip_y=0.5, random_state=1010)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    stratify=y,
                                                    random_state=1810)

clfs = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
        ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
        GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)]

blend_clf = LogisticRegression()

y_preds = blend(clfs, blend_clf, 5, X_train, y_train, X_test)
score = log_loss(y_test, y_preds)
print('blend logloss:', score)


