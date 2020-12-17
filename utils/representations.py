# -*- coding: utf-8 -*-

# In[1]:
import numpy as np
import pandas as pd
import matrixprofile as mp

from pyts.image import GramianAngularField, RecurrencePlot
from pyts.approximation import PiecewiseAggregateApproximation, SymbolicAggregateApproximation, SymbolicFourierApproximation
from pyts.transformation import ROCKET
from matplotlib import pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
# # Define classes for representation methods

# Here we define custom classes when necessary for the representation methods we will use inside pipelines during cross validation.
#
# See corresponding modules documentation for documentation.
#
# Pyts : https://pyts.readthedocs.io/
#
# MatrixProfile : https://matrixprofile.docs.matrixprofile.org/

# In[2]:

#Gramian natively use PAA, reccurence don't,
#that's why you'll see calls to PAA inside the Recurrence class but not in the Gramian

class Gramian_transform(BaseEstimator, TransformerMixin):
    def __init__(self, img_size=128, flatten=False, method='s'):
        self.img_size = img_size
        self.flatten = flatten
        self.method = method
        self.cmap = plt.get_cmap('jet')
        self.transformer = None

    def transform(self, X, y=None):
        if type(X[0]) == pd.core.series.Series:
            X = np.asarray([x.values for x in X])

        X = np.asarray([self.transformer.transform(x.reshape(1,-1)) for x in X if x.shape[0] >= self.img_size])
        if self.flatten == True:
            X = X.reshape(X.shape[0], X.shape[2])
        else:
            X = X.reshape(X.shape[0], self.img_size, self.img_size, 1)
            X = self.cmap(X)[:,:,:,:,0:3].reshape(X.shape[0],self.img_size, self.img_size,3)
        return X

    def fit(self, X, y=None):
        self.transformer = GramianAngularField(image_size=self.img_size,
                                               method=self.method,
                                               flatten=self.flatten)
        self.transformer.fit(X)
        return self

class Recurrence_transform(BaseEstimator, TransformerMixin):
    def __init__(self, output_size=128, dimension=1, time_delay=6, flatten=False):
        self.output_size = output_size
        self.flatten=flatten
        self.dimension = dimension
        self.time_delay = time_delay
        self.cmap = plt.get_cmap('jet')
        self.transformer = None

    def transform(self, X, y=None):
        if type(X[0]) == pd.core.series.Series:
            X = np.asarray([x.values for x in X])

        X = np.asarray([self.approximator.transform(x.reshape(1,-1))for x in X if x.shape[0] >= self.output_size])
        X = np.asarray([self.transformer.transform(x) for x in X if x.shape[0]])
        if self.flatten == True:
            X = X.reshape(X.shape[0], X.shape[2])
        else:
            X = X.reshape(X.shape[0], self.output_size, self.output_size, 1)
            X = self.cmap(X)[:,:,:,:,0:3].reshape(X.shape[0],self.output_size, self.output_size,3)
        return X

    def fit(self, X, y=None):
        self.approximator = PiecewiseAggregateApproximation(output_size=self.output_size,
                                                                  window_size=None,
                                                                  overlapping=False)
        self.approximator.fit(X)
        self.transformer = RecurrencePlot(dimension=self.dimension,
                                          time_delay=self.time_delay,
                                          flatten=self.flatten)
        self.transformer.fit(X)
        return self

class PiecewiseApproximation_transform(BaseEstimator, TransformerMixin):
    def __init__(self, output_size=1000, overlapping=False, window_size=None):
        self.output_size = output_size
        self.overlapping = overlapping
        self.window_size = window_size
        self.transformer = None

    def transform(self, X, y=None):
        if type(X[0]) == pd.core.series.Series:
            X = np.asarray([x.values for x in X])

        X = np.asarray([self.transformer.transform(x.reshape(1,-1)) for x in X if x.shape[0] >= self.output_size])
        X = X.reshape(X.shape[0], X.shape[2], X.shape[1])
        return X

    def fit(self, X, y=None):
        self.transformer = PiecewiseAggregateApproximation(output_size=self.output_size,
                                                           window_size=self.window_size,
                                                           overlapping=self.overlapping)
        self.transformer.fit(X)
        return self

class SymbolicAggregate_transform(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins=5, strategy='uniform', alphabet='ordinal'):
        self.n_bins = n_bins
        self.strategy = strategy
        self.alphabet = alphabet
        self.transformer = None

    def transform(self, X, y=None):
        X = np.asarray([self.transformer.transform(x.reshape(1,-1)).astype(float) if np.max(x) - np.min(x) != 0 else np.zeros((1,x.shape[0])) for x in X])
        X = X.reshape(X.shape[0], X.shape[2], X.shape[1])
        return X

    def fit(self, X, y=None):
        self.transformer = SymbolicAggregateApproximation(n_bins=self.n_bins,
                                                          strategy=self.strategy,
                                                          alphabet=self.alphabet)
        self.transformer.fit(X)
        return self

class SymbolicFourrier_transform(BaseEstimator, TransformerMixin):
    def __init__(self, n_coefs=10, n_bins=5, strategy='uniform', drop_sum=True,
                 anova=True, norm_mean=False, norm_std=False, alphabet='ordinal'):
        self.n_coefs = n_coefs
        self.n_bins = n_bins
        self.strategy = strategy
        self.alphabet = alphabet
        self.drop_sum = drop_sum
        self.anova = anova
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.transformer = None

    def transform(self, X, y=None):
        X = np.asarray([self.transformer.transform(x.reshape(1,-1)).astype(float) if np.max(x) - np.min(x) != 0 else np.zeros((1,x.shape[0])) for x in X])
        X = X.reshape(X.shape[0], X.shape[2], X.shape[1])
        return X

    def fit(self, X, y):
        self.transformer = SymbolicFourierApproximation(n_coefs=self.n_coefs, n_bins=self.n_bins,
                                                        strategy=self.strategy, alphabet=self.alphabet,
                                                        drop_sum=self.drop_sum, anova=self.anova,
                                                        norm_mean=self.norm_mean, norm_std=self.norm_std)
        X = X.reshape(X.shape[0],X.shape[1])
        self.transformer.fit(X,y)
        return self


class MatrixProfile_transform():
    def __init__(self, window_size=0.15):
        self.window_size = window_size

    def transform(self, X, y=None):
        if type(X[0]) == pd.core.series.Series:
            X = np.asarray([x.values for x in X])
        X = np.asarray([mp.compute(x.reshape(-1),windows=x.shape[0]*self.window_size)['mp'].reshape(1,-1) for x in X])
        X = X.reshape(X.shape[0], X.shape[2], X.shape[1])
        return X

    def fit(self, X, y=None):
        return self

class ROCKET_transform(BaseEstimator, TransformerMixin):
    def __init__(self, n_kernels=20000, kernel_sizes=(5,7,9,11), flatten=False):
        self.flatten = flatten
        self.n_kernels = n_kernels
        self.kernel_sizes = kernel_sizes
        self.transformer = None

    def transform(self, X, y=None):
        X = X.reshape(X.shape[0],X.shape[1])
        X = self.transformer.transform(X)
        if self.flatten:
            X = X.reshape(X.shape[0], X.shape[1])
        else:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        return X

    def fit(self, X, y=None):
        self.transformer = ROCKET(n_kernels=self.n_kernels, kernel_sizes=self.kernel_sizes)
        X = X.reshape(X.shape[0],X.shape[1])
        self.transformer.fit(X)
        return self
