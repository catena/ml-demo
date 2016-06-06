# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 19:23:18 2016

@author: vishnu
"""

from sklearn import linear_model
clf = linear_model.Lars(n_nonzero_coefs=1)
clf.fit([[-1, 1], [0, 0], [1, 1]], [-1.1111, 0, -1.1111])
print(clf.coef_)