# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 16:58:26 2016

@author: vishnu
"""

# gaussian naive bayes
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.cross_validation import train_test_split

X, y = make_classification(n_samples=10000, n_features=20, n_informative=2,
                           n_redundant=10, flip_y=0.5, random_state=1010)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    stratify=y,
                                                    random_state=1022)

# Gaussian Naive Bayes                                                   
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Accuracy : %f" % (y_pred == y_test).mean())

# Bernoulli Naive Bayes
bnb = BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
bnb.fit(X_train, y_train)
y_pred = bnb.fit(X_train, y_train).predict(X_test)
print("Accuracy : %f" % (y_pred == y_test).mean())