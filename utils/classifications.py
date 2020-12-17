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
from sktime.utils.data_container import _concat_nested_arrays as cna
from sktime.classification.frequency_based import RandomIntervalSpectralForest
from sklearn.base import BaseEstimator, ClassifierMixin

# # Define classes for classification methods

# Here we define custom classes when necessary for the classification methods we will use inside pipelines during cross validation.
#
# See corresponding modules documentation for documentation.
#
# Pyts : https://pyts.readthedocs.io/
#
# sktime : https://sktime.org/index.html
#
# sklearn : https://scikit-learn.org/stable/index.html



# In[10]:

"""
# This section is left commented so you have no trouble running the script without Tensorflow/GPU
# While using ResNet, if you have error during cross validation, you can try to make the class ResNetV2
# inherit the tensorflow.keras KerasClassifier wrapper, it can fix some issues.
# Don't forget to uncomment pipelines using ResNet in CV_scripts aswell.

from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight


class ResNetV2(BaseEstimator, ClassifierMixin):
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
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
        self.model = model

    def fit(self, X, y, epochs=1500, batch_size=32, return_hist=False, el_patience=100, verbose=0, val_size=0.1):
        self.init_model((X.shape[1], X.shape[2], X.shape[3]))
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size)
        el = EarlyStopping(monitor='val_loss', patience=el_patience, restore_best_weights=True, mode='min')

        self.model.fit(
            X_train, y_train,
            validation_data=(X_val,y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=[el],
            shuffle=True
        )
        return self

    def predict(self, X):
        return np.array([x>0.5 for x in self.model.predict(X)]).astype(int)

    def predict_proba(self,X):
        return self.model.predict(X)

#Depending on your sktime_dl version, this might throw import errors, see Readme for a fix.
from sktime_dl.deeplearning.inceptiontime._classifier import InceptionTimeClassifier


class InceptionTime(BaseEstimator, ClassifierMixin):
    def __init__(self, depth=18, nb_filters=32, bottleneck_size=32):
        self.model = None
        self.depth = depth
        self.nb_filters = nb_filters
        self.bottleneck_size = bottleneck_size

    def fit(self, X, y, epochs=1500, batch_size=32,
            el_patience=100, verbose=False, val_size=0.1):
        self.model = InceptionTimeClassifier(verbose=verbose, depth=self.depth,
                                             nb_filters=self.nb_filters, bottleneck_size=self.bottleneck_size,
                                             callbacks=[EarlyStopping(monitor='val_loss', patience=el_patience,
                                                                      restore_best_weights=True, mode='min')])

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size)
        self.model.fit(X_train, y_train, validation_X=X_val,validation_y=y_val)
        return self

    def predict(self, X):
        return np.array([x>0.5 for x in self.model.predict(X)]).astype(int)

    def predict_proba(self,X):
        return self.model.predict(X)

"""

class SktimeEstimator:
    def _sktime_format(self,X,y):
        # X : (n_instance, n_timestamp, n_features)
        X, y = self._sktime_format_X(X), np.asarray(y)
        return X, y

    def _sktime_format_X(self,X):
        # X : (n_instance, n_timestamp, n_features)
        return cna(X.reshape(X.shape[2],X.shape[0],X.shape[1]))

class PytsEstimator:
    def _format(self,X,y):
        return self._format_X(X), np.asarray(y)

    def _format_X(self,X):
        return X.reshape(X.shape[0],X.shape[1])


class RISE(BaseEstimator, ClassifierMixin, SktimeEstimator):
    def __init__(self, min_length=5, n_estimators=300):
        self.min_length = min_length
        self.n_estimators = n_estimators
        self.estimator = None

    def fit(self,X,y):
        X, y = self._sktime_format(X,y)
        self.estimator = RandomIntervalSpectralForest(n_estimators=self.n_estimators,
                                                      min_interval=self.min_length)
        self.estimator.fit(X,y)
        return self

    def predict(self,X):
        X = self._sktime_format_X(X)
        return self.estimator.predict(X)

    def predict_proba(self,X):
        X = self._sktime_format(X)
        return self.estimator.predict_proba(X)


class Random_Forest(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=300, max_depth=None, max_samples=0.75,
            ccp_alpha=0.0225, class_weight="balanced_subsample"):
        self.n_estimators=n_estimators
        self.max_depth=max_depth
        self.max_samples=max_samples
        self.ccp_alpha=ccp_alpha
        self.class_weight=class_weight
        self.estimator = None

    def fit(self, X, y):
        X = np.asarray([x.astype(np.float32) for x in X])
        self.estimator = RandomForestClassifier(n_estimators=self.n_estimators,
                                                max_depth=self.max_depth,
                                                max_samples=self.max_samples,
                                                ccp_alpha=self.ccp_alpha,
                                                class_weight=self.class_weight)

        self.estimator.fit(X,y)
        return self

    def predict(self,X):
        X = np.asarray([x.astype(np.float32) for x in X])
        return self.estimator.predict(X)

    def predict_proba(self,X):
        X = np.asarray([x.astype(np.float32) for x in X])
        return self.estimator.predict_proba(X)

class KNN_classif(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=7, weights='distance',p=2):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        self.estimator = None

    def fit(self,X,y):
        self.estimator = KNeighborsClassifier(n_neighbors=self.n_neighbors,
                                              weights=self.weights, p=self.p)
        self.estimator.fit(X,y)
        return self

    def predict(self,X):
        return self.estimator.predict(X)

    def predict_proba(self,X):
        return self.estimator.predict_proba(X)

class TimeSeries_Forest(BaseEstimator, ClassifierMixin, SktimeEstimator):
    def __init__(self, n_estimators=300,  min_interval=5):
        self.n_estimators = n_estimators
        self.min_interval = min_interval
        self.estimator = None

    def fit(self,X,y):
        X, y = self._sktime_format(X,y)
        self.estimator = TimeSeriesForest(n_estimators=self.n_estimators,
                                          min_interval=self.min_interval)

        self.estimator.fit(X,y)
        return self

    def predict(self,X):
        X = self._sktime_format_X(X)
        return self.estimator.predict(X)

    def predict_proba(self,X):
        X = self._sktime_format_X(X)
        return self.estimator.predict_proba(X)


class SVM_classif(BaseEstimator, ClassifierMixin):
    def __init__(self, C=10, kernel='rbf', degree=2, gamma='scale',
                 cache_size=500, class_weight='balanced'):
            self.C = C
            self.kernel = kernel
            self.degree = degree #Not used with RBF
            self.gamma = gamma
            self.cache_size = cache_size
            self.class_weight = class_weight
            self.estimator = None

    def fit(self,X,y):
        self.estimator = SVC(C=self.C, kernel=self.kernel, degree=self.degree,
                             gamma=self.gamma, cache_size=self.cache_size,
                             class_weight=self.class_weight)
        self.estimator.fit(X,y)
        return self

    def predict(self,X):
        return self.estimator.predict(X)

    def predict_proba(self,X):
        return self.estimator.predict_proba(X)

class Ridge_classif(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=10.0, normalize=False, copy_X=True, max_iter=None, tol=0.001,
                 class_weight='balanced'):
        self.alpha = alpha
        self.normalize = normalize
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.class_weight = class_weight
        self.estimator = None

    def set_params(self, **params):
        return self.estimator.set_params(**params)

    def fit(self,X,y):
        self.estimator = RidgeClassifier(alpha=self.alpha, normalize=self.normalize,
                                         copy_X=self.copy_X, max_iter=self.max_iter,
                                         tol=self.tol, class_weight=self.class_weight)
        self.estimator.fit(X,y)
        return self

    def predict(self,X):
        return self.estimator.predict(X)

    def predict_proba(self,X):
        return self.estimator.predict_proba(X)

class KNN_TS_classif(BaseEstimator, ClassifierMixin, PytsEstimator):
    def __init__(self, n_neighbors=7, weights='distance', p=2):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        self.estimator = None

    def fit(self,X,y):
        X, y = self._format(X,y)
        self.estimator = KNeighborsClassifierTS(n_neighbors=self.n_neighbors,
                                                weights=self.weights, p=self.p)
        self.estimator.fit(X,y)
        return self

    def predict(self,X):
        X = self._format_X(X)
        return self.estimator.predict(X)

    def predict_proba(self,X):
        X = self._format_X(X)
        return self.estimator.predict_proba(X)

class BOSSVS_classif(BaseEstimator, ClassifierMixin, PytsEstimator):
    def __init__(self, word_size=5, n_bins=5, window_size=0.15, window_step=0.01,
                 anova=True, drop_sum=False, norm_mean=False, norm_std=False,
                 strategy='quantile', alphabet=None,smooth_idf=True):
        self.word_size = word_size
        self.n_bins = n_bins
        self.window_size = window_size
        self.window_step = window_step
        self.anova = anova
        self.drop_sum = drop_sum
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.strategy = strategy
        self.alphabet = alphabet
        self.smooth_idf = smooth_idf
        self.estimator = None

    def fit(self,X,y):
        X, y = self._format(X,y)
        self.estimator = BOSSVS(word_size=self.word_size, n_bins=self.n_bins,
                                window_size=self.window_size, window_step=self.window_step,
                                anova=self.anova, drop_sum=self.drop_sum,
                                norm_mean=self.norm_mean, norm_std=self.norm_std,
                                strategy=self.strategy, alphabet=self.alphabet,
                                smooth_idf=self.smooth_idf)

        self.estimator.fit(X,y)
        return self

    def predict(self,X):
        X = self._format_X(X)
        return self.estimator.predict(X)

    def predict_proba(self,X):
        X = self._format_X(X)
        return self.estimator.predict_proba(X)