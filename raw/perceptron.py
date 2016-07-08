# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 12:31:32 2016

@author: vishnu
"""

import lasagne
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.datasets import make_classification
from lasagne.nonlinearities import softmax, tanh
from lasagne.layers import InputLayer, DenseLayer, get_output
from lasagne.updates import sgd, apply_momentum

np.random.seed(0)

# ignore warnings
warnings.filterwarnings('ignore', module='lasagne')

# create data
n_classes=4
X, y = make_classification(n_features=2, n_redundant=0, n_classes=n_classes,
                           n_clusters_per_class=1)

X = X.astype(theano.config.floatX)
y = y.astype('int32')
plt.scatter(X[:,0], X[:,1], c=y)    # color encodes class

# create perceptron layers
l_in = InputLayer(shape=X.shape)
l_hidden = DenseLayer(l_in, num_units=10, nonlinearity=tanh)
l_output = DenseLayer(l_hidden, num_units=n_classes, nonlinearity=softmax)
net_output = get_output(l_output)

# optimization objective
true_output = T.ivector('true_output')
loss = lasagne.objectives.categorical_crossentropy(net_output, true_output)
loss = loss.mean()

# optimize using sgd
params = lasagne.layers.get_all_params(l_output)
updates_sgd = sgd(loss, params, learning_rate=0.1)
updates = apply_momentum(updates_sgd, params, momentum=0.9)

# train
train = theano.function([l_in.input_var, true_output], loss, updates=updates)
get_output = theano.function([l_in.input_var], net_output)
for epoch in range(100):
    loss = train(X, y)
    print("Epoch %d: Loss %g" % (epoch + 1, loss))

# plot accuracy
y_pred = np.argmax(get_output(X), axis=1)
score = np.mean(y == y_pred)
plt.scatter(X[:,0], X[:,1], c=(y != y_pred), cmap=plt.cm.gray_r)
plt.title('Accuracy: {0:.6f}'.format(score))













