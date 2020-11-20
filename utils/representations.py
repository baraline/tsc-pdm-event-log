# -*- coding: utf-8 -*-
# In[1]:
import numpy as np
import pandas as pd
import matrixprofile as mp

from pyts.image import GramianAngularField, RecurrencePlot
from pyts.approximation import PiecewiseAggregateApproximation, SymbolicAggregateApproximation, SymbolicFourierApproximation
from pyts.transformation import ROCKET
from matplotlib import pyplot as plt

# # Define classes for representation methods

# Here we define custom classes when necessary for the representation methods we will use inside pipelines during cross validation. 
# 
# See corresponding modules documentation for documentation.
# 
# Pyts : https://pyts.readthedocs.io/
# 
# MatrixProfile : https://matrixprofile.docs.matrixprofile.org/
# 
# sktime : https://sktime.org/index.html
# 
# sklearn : https://scikit-learn.org/stable/index.html

# In[2]:

#Gramian natively use PAA, reccurence don't, 
#that's why you'll see calls to PAA inside the Recurrence class but not in the Gramian

class Gramian_transform:
    def __init__(self, img_size=128, flatten=False, method='s'):
        self.img_size = img_size
        self.flatten=flatten
        self.cmap = plt.get_cmap('jet')
        self.transformer = GramianAngularField(image_size=img_size,
                                               method=method,
                                               flatten=flatten)
    def transform(self,X):
        if type(X[0]) == pd.core.series.Series:
            X = np.asarray([x.values for x in X])
        
        X = np.asarray([self.transformer.transform(x.reshape(1,-1)) for x in X if x.shape[0] >= self.img_size])
        if self.flatten == True:
            X = X.reshape(X.shape[0], X.shape[2])
        else:
            X = X.reshape(X.shape[0], self.img_size, self.img_size, 1)
            X = self.cmap(X)[:,:,:,:,0:3].reshape(X.shape[0],self.img_size, self.img_size,3)
        return X
    
    def set_params(self, **params):
        return self.transformer.set_params(**params)
    
    def fit_transform(self,X,y):
        return self.transform(X)

class Recurrence_transform:
    def __init__(self, output_size=128, dimension=1, time_delay=6, flatten=False):
        self.output_size = output_size
        self.flatten=flatten
        self.cmap = plt.get_cmap('jet')
        self.approximator = PiecewiseAggregateApproximation(output_size=output_size,
                                                                  window_size=None, 
                                                                  overlapping=False)
        self.transformer = RecurrencePlot(dimension=dimension,
                                          time_delay=time_delay,
                                          flatten=flatten)
    def transform(self,X):
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

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            if parameter == 'output_size':
                self.approximator.set_params(**{parameter: value})
                setattr(self, parameter, value)
            elif parameter in ['dimension','time_delay']:
                self.transformer.set_params(**{parameter: value})
            else:
                setattr(self, parameter, value)
        return self
    
    def fit_transform(self,X,y):
        return self.transform(X)

class PiecewiseApproximation_transform:
    def __init__(self, output_size=1000, overlapping=False, window_size=None):
        self.output_size = output_size
        self.transformer = PiecewiseAggregateApproximation(output_size=output_size, 
                                                           window_size=window_size,
                                                           overlapping=overlapping)
    def transform(self,X):
        if type(X[0]) == pd.core.series.Series:
            X = np.asarray([x.values for x in X])
            
        X = np.asarray([self.transformer.transform(x.reshape(1,-1)) for x in X if x.shape[0] >= self.output_size])
        X = X.reshape(X.shape[0], X.shape[2], X.shape[1])
        return X
    
    def set_params(self, **params):
        return self.transformer.set_params(**params)
    
    def fit_transform(self,X,y):
        return self.transform(X)
        
class SymbolicAggregate_transform:
    def __init__(self, n_bins=7, strategy='uniform', alphabet='ordinal'):
        self.transformer = SymbolicAggregateApproximation(n_bins=n_bins, strategy=strategy,
                                                          alphabet=alphabet)
        
    def set_params(self, **params):
        return self.transformer.set_params(**params)
    
    def transform(self, X):
        X = np.asarray([self.transformer.transform(x.reshape(1,-1)).astype(float) if np.max(x) - np.min(x) != 0 else np.zeros((1,x.shape[0])) for x in X])
        X = X.reshape(X.shape[0], X.shape[2], X.shape[1])
        return X
    
    def fit_transform(self,X,y):
        return self.transform(X)
    
class SymbolicFourrier_transform:
    def __init__(self, n_coefs=20, n_bins=7, strategy='uniform', drop_sum=False,
                 anova=True, norm_mean=True, norm_std=False, alphabet='ordinal'):
        self.transformer = SymbolicFourierApproximation(n_coefs=n_coefs, n_bins=n_bins,
                                                        strategy=strategy, alphabet=alphabet,
                                                        drop_sum=drop_sum, anova=anova,
                                                        norm_mean=norm_mean, norm_std=norm_std)
    def transform(self,X):
        X = np.asarray([self.transformer.transform(x.reshape(1,-1)).astype(float) if np.max(x) - np.min(x) != 0 else np.zeros((1,x.shape[0])) for x in X])         
        X = X.reshape(X.shape[0], X.shape[2], X.shape[1])
        return X
    
    def set_params(self, **params):
        return self.transformer.set_params(**params)
    
    def fit_transform(self,X,y):
        X = X.reshape(X.shape[0],X.shape[1])
        self.transformer.fit(X,y)
        return self.transform(X)
    
    
class MatrixProfile_transform:
    def __init__(self, window_size=0.075):
        self._window_size=window_size
        
    def transform(self, X):
        if type(X[0]) == pd.core.series.Series:
            X = np.asarray([x.values for x in X])
        X = np.asarray([mp.compute(x.reshape(-1),windows=x.shape[0]*self._window_size)['mp'].reshape(1,-1) for x in X])        
        X = X.reshape(X.shape[0], X.shape[2], X.shape[1])
        return X
    
    def fit_transform(self,X,y):
        return self.transform(X)
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
class ROCKET_transform:
    def __init__(self, n_kernels=15000, kernel_sizes=(5,7,9), flatten=False):
        self.flatten = flatten
        self.transformer = ROCKET(n_kernels=n_kernels, kernel_sizes=kernel_sizes)
        
    def set_params(self, **params):
        return self.transformer.set_params(**params)
    
    def transform(self,X):
        X = X.reshape(X.shape[0],X.shape[1])
        X = self.transformer.transform(X)
        if self.flatten:
            X = X.reshape(X.shape[0], X.shape[1])
        else:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        return X
    
    def fit_transform(self,X,y):
        X = X.reshape(X.shape[0],X.shape[1])
        self.transformer.fit(X)
        return self.transform(X)

