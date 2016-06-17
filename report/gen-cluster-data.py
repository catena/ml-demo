# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 12:13:55 2016

@author: vishnu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

np.random.seed(0)

n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5, 
                                      noise=.05, random_state=1010)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05,
                                  random_state=1011)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=1012)

datasets = [noisy_circles, noisy_moons, blobs]
plt.figure(figsize=(12, len(datasets) * 1.25))
for i_ds, ds in enumerate(datasets):
    plt.subplot(1, 3, i_ds + 1)
    plt.scatter(ds[0][:,0], ds[0][:,1], c=ds[1])

noisy_circles_df = pd.DataFrame(np.column_stack(noisy_circles), 
                                columns=['x1','x2','y'])
noisy_moons_df = pd.DataFrame(np.column_stack(noisy_moons), 
                              columns=['x1','x2','y'])
blobs_df = pd.DataFrame(np.column_stack(blobs),
                        columns=['x1', 'x2', 'y'])

noisy_circles_df.to_csv('../data/noisy_circles.csv', index=False)
noisy_moons_df.to_csv('../data/noisy_moons.csv', index=False)
blobs_df.to_csv('../data/blobs.csv', index=False)

