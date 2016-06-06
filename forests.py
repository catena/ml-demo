# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 18:46:11 2016

@author: vishnu
"""

from sklearn.cross_validation import cross_val_score
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

def eval_score(clf, name, verbose=0):
    X, y = make_classification(n_samples=10000, n_features=20, n_informative=2,
                               n_redundant=10, flip_y=0.5, random_state=1010)
    scores = cross_val_score(clf, X, y, scoring='roc_auc', cv=5, 
                             verbose=verbose)
    print(name, "roc_auc: {0:.6f} +/- {1:.6f}".format(scores.mean(), 
          scores.std()))

#rf roc_auc: 0.758168 +/- 0.008232
clf_rf = RandomForestClassifier(n_estimators=1000, criterion='entropy',
                                max_depth=22, min_samples_leaf=65, 
                                max_features='sqrt', bootstrap=True, 
                                n_jobs=-1, random_state=880)

# et roc_auc: 0.755174 +/- 0.007803
clf_et = ExtraTreesClassifier(n_estimators=1000, criterion='gini',
                              max_depth=10, min_samples_leaf=1,
                              max_features='sqrt', bootstrap=False,
                              n_jobs=-1, random_state=1011)

models = [(clf_rf, 'rf'), (clf_et, 'et')]

for (clf, name) in models:
    eval_score(clf, name)













