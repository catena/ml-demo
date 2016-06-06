# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 12:58:30 2016

@author: vishnu
"""

from sklearn.cross_validation import cross_val_score
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb

X, y = make_classification(n_samples=10000, n_features=20, n_informative=2,
                           n_redundant=10, flip_y=0.5, random_state=1010)

# Adaboost classifier                           
adc = AdaBoostClassifier(n_estimators=1000, learning_rate=0.01, 
                         random_state=1010)
score_adc = cross_val_score(adc, X, y, scoring='roc_auc', cv=5, 
                         verbose=0)
#adc roc_auc: 0.751203 +/- 0.006307
print("adc", "roc_auc: {0:.6f} +/- {1:.6f}".format(score_adc.mean(), 
      score_adc.std()))
      
# Extreme Gradient Boosting
dtrain = xgb.DMatrix(X, y)
params = {'eta': 0.1,
          'max_depth': 6,
          'min_child_weight': 1,
          'subsample': 0.9,
          'colsample_bylevel': 0.4,
          'objective': 'binary:logistic',
          'eval_metric': 'auc'}
# [13]    train-auc:0.833153+0.00260341   test-auc:0.760512+0.00776352
xgb.cv(params, dtrain, num_boost_round=1000, nfold=5, stratified=True,
       early_stopping_rounds=10, verbose_eval=True)
