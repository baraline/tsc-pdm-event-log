# -*- coding: utf-8 -*-
# In[1]:
import numpy as np
from pyts.classification import BOSSVS
from pyts.classification import KNeighborsClassifier as KNeighborsClassifierTS
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sktime.classification.interval_based import TimeSeriesForest
from sktime.utils.data_container import concat_nested_arrays as cna
from sktime.classification.frequency_based import RandomIntervalSpectralForest


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



# In[10]:

"""
#This section is left commented so you have no trouble running the script without Tensorflow/GPU
#If you have error during cross validation, you can try to make the class ResNetV2
# inherit the tensorflow.keras KerasClassifier wrapper, it can fix some issues.
# Don't forget to uncomment pipelines in CV_scripts aswell.

from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

class ResNetV2:
    def __init__(self, loss='binary_crossentropy', pooling='avg', optimizer=Adam(lr=1e-4)):
        self.loss = loss
        self.pooling = pooling
        self.optimizer = optimizer
        self.model = None

    def init_model(self, input_shape):
        model = ResNet50V2(
            include_top=True,
            weights=None,
            input_shape=input_shape,
            pooling=self.pooling,
            classes=1,
            classifier_activation="sigmoid",
        )
        model.compile(optimizer=self.optimizer, loss=self.loss, weighted_metrics=['accuracy'])
        self.model = model

    def fit(self, X, y, epochs=1000, batch_size=32, return_hist=False, el_patience=70, verbose=0, val_size=0.1):
        self.init_model((X.shape[1], X.shape[2], X.shape[3]))
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size)
        el = EarlyStopping(monitor='val_accuracy', patience=el_patience, restore_best_weights=True, mode='max')
        cw = compute_class_weight('balanced',np.unique(y_train), y_train)
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val,y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=[el],
            shuffle=True,
            class_weight={0:cw[0],1:cw[1]}
        )
        if return_hist:
            return history
    
    def predict(self, X):
        return np.array([x>0.5 for x in self.model.predict(X)]).astype(int)
"""
    


class RISE:
    def __init__(self, min_length=5, n_estimators=300):
        self.estimator = RandomIntervalSpectralForest(n_estimators=n_estimators, min_interval=min_length)

    def _sktime_format(self,X,y):
        # X : (n_instance, n_timestamp, n_features)
        X, y = cna(X.reshape(X.shape[2],X.shape[0],X.shape[1])), np.asarray(y)
        return X, y
    
    def set_params(self, **parameters):
        self.estimator.set_params(**parameters)
        return self
    
    def _sktime_format_X(self,X):
        # X : (n_instance, n_timestamp, n_features)
        return cna(X.reshape(X.shape[2],X.shape[0],X.shape[1]))
    
    def fit(self,X,y):
        X, y = self._sktime_format(X,y)
        self.estimator.fit(X,y)

    def predict(self,X):
        X = self._sktime_format_X(X)
        return self.estimator.predict(X)
    
    def predict_proba(self,X):
        X = self._sktime_format(X)
        return self.estimator.predict_proba(X)

class Random_Forest:
    def __init__(self, n_estimators=300, max_depth=None, max_features=0.75, max_samples=0.75,
            ccp_alpha=0.0225, class_weight="balanced_subsample"):
        self.estimator = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                max_features=max_features, max_samples=max_samples,
                                                ccp_alpha=ccp_alpha,class_weight=class_weight)
        
    def set_params(self, **params):
        return self.estimator.set_params(**params)
    
    def fit(self,X,y):
        X = np.asarray([x.astype(np.float32) for x in X])
        self.estimator.fit(X,y)
    
    def predict(self,X):
        X = np.asarray([x.astype(np.float32) for x in X])
        return self.estimator.predict(X)
    
    def predict_proba(self,X):
        X = np.asarray([x.astype(np.float32) for x in X])
        return self.estimator.predict_proba(X)

class KNN_classif:
    def __init__(self, n_neighbors=9, weights='distance',p=2):
        self.estimator = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)
            
    def set_params(self, **params):
        return self.estimator.set_params(**params)
        
    def fit(self,X,y):       
        self.estimator.fit(X,y)
    
    def predict(self,X):
        return self.estimator.predict(X)
    
    def predict_proba(self,X):
        return self.estimator.predict_proba(X)

class TimeSeries_Forest:
    def __init__(self, n_estimators=300,  min_interval=3):
        
        self.estimator = TimeSeriesForest(n_estimators=n_estimators,
                                          min_interval=3) 
        
    def set_params(self, **params):
        return self.estimator.set_params(**params)
    
    def _sktime_format(self,X,y):
        # X : (n_instance, n_timestamp, n_features)
        X, y = cna(X.reshape(X.shape[2],X.shape[0],X.shape[1])), np.asarray(y)
        return X, y
    
    def _sktime_format_X(self,X):
        # X : (n_instance, n_timestamp, n_features)
        return cna(X.reshape(X.shape[2],X.shape[0],X.shape[1]))
    
    def fit(self,X,y):
        X, y = self._sktime_format(X,y)
        self.estimator.fit(X,y)
        
    def predict(self,X):
        X = self._sktime_format_X(X)
        return self.estimator.predict(X)
    
    def predict_proba(self,X):
        X = self._sktime_format_X(X)
        return self.estimator.predict_proba(X)
        
    
class SVM_classif:
    def __init__(self, C=10, kernel='rbf', degree=2, gamma='scale'):
        self.estimator = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma,
                             cache_size=500, class_weight='balanced')
            
    def set_params(self, **params):
        return self.estimator.set_params(**params)
        
    def fit(self,X,y):       
        self.estimator.fit(X,y)
    
    def predict(self,X):
        return self.estimator.predict(X)
    
    def predict_proba(self,X):
        return self.estimator.predict_proba(X)

class Ridge_classif:
    def __init__(self, alpha=10.0, normalize=False, copy_X=True, max_iter=None, tol=0.001,
                 class_weight='balanced'):
        self.estimator = RidgeClassifier(alpha=alpha, normalize=normalize, copy_X=copy_X,
                                         max_iter=max_iter, tol=tol, class_weight=class_weight)
    
    def set_params(self, **params):
        return self.estimator.set_params(**params)

    def fit(self,X,y):       
        self.estimator.fit(X,y)
    
    def predict(self,X):
        return self.estimator.predict(X)
    
    def predict_proba(self,X):
        return self.estimator.predict_proba(X)     

class KNN_TS_classif:
    def __init__(self, n_neighbors=9, weights='distance', p=2):
        self.estimator = KNeighborsClassifierTS(n_neighbors=n_neighbors, weights=weights, p=p)
    
    def _format(self,X,y):
        return X.reshape(X.shape[0],X.shape[1]), np.asarray(y)
        
    def set_params(self, **params):
        return self.estimator.set_params(**params)
        
    def _format_X(self,X):
        return X.reshape(X.shape[0],X.shape[1])
    
    def fit(self,X,y):       
        X, y = self._format(X,y)
        self.estimator.fit(X,y)
    
    def predict(self,X):
        X = self._format_X(X)
        return self.estimator.predict(X)
    
    def predict_proba(self,X):
        X = self._format_X(X)
        return self.estimator.predict_proba(X)
        

class BOSSVS_classif:
    def __init__(self, word_size=9, n_bins=7, window_size=0.2, window_step=1,
                 anova=True, drop_sum=False, norm_mean=False, norm_std=False,
                 strategy='uniform', alphabet=None):
        self.estimator = BOSSVS(word_size=word_size, n_bins=n_bins,
                                window_size=window_size, window_step=window_step,
                                anova=anova, drop_sum=drop_sum,
                                norm_mean=norm_mean, norm_std=norm_std,
                                strategy=strategy, alphabet=alphabet)
    def set_params(self, **params):
        return self.estimator.set_params(**params)
    
    def _format(self,X,y):
        # X : (n_instance, n_timestamp, n_features)
        return X.reshape(X.shape[0],X.shape[1]), np.asarray(y)
        
    def _format_X(self,X):
        # X : (n_instance, n_timestamp, n_features)
        return X.reshape(X.shape[0],X.shape[1])
    
    def fit(self,X,y):       
        X, y = self._format(X,y)
        self.estimator.fit(X,y)
    
    def predict(self,X):
        X = self._format_X(X)
        return self.estimator.predict(X)
    
    def predict_proba(self,X):
        X = self._format_X(X)
        return self.estimator.predict_proba(X)
