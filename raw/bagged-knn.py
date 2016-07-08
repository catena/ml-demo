# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 16:44:46 2016

@author: vishnu
"""

from sklearn.datasets import make_classification
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score

def eval_score(clf, name, verbose=0):
    X, y = make_classification(n_samples=10000, n_features=20, n_informative=2,
                               n_redundant=10, flip_y=0.5, random_state=1010)
    scores = cross_val_score(clf, X, y, scoring='roc_auc', cv=5, 
                             verbose=verbose)
    print(name, "roc_auc: {0:.6f} +/- {1:.6f}".format(scores.mean(), 
          scores.std()))
                               
knc = KNeighborsClassifier(n_neighbors=55, weights='distance', 
                           metric='minkowski', p=1, n_jobs=-1)
#knn classifier roc_auc: 0.750158 +/- 0.007638
eval_score(knc, 'knn classifier')

bagged_knc10 = BaggingClassifier(knc, n_estimators=10, n_jobs=-1, 
                                 random_state=2233)
#bag of 10 knc roc_auc: 0.751811 +/- 0.007779
eval_score(bagged_knc10, 'bag of 10 knc')

bagged_knc20 = BaggingClassifier(knc, n_estimators=20, n_jobs=-1, 
                                 random_state=2233)
#bag of 20 knc roc_auc: 0.751938 +/- 0.007041
eval_score(bagged_knc20, 'bag of 20 knc')