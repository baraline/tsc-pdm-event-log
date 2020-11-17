# # Imports

# In[2]:
# All in one script to maximize portability

import pandas as pd
import numpy as np
import warnings
import matrixprofile as mp

from os import listdir, mkdir
from os.path import isfile, join, exists
from datetime import timedelta

from pyts.image import GramianAngularField, RecurrencePlot
from pyts.approximation import PiecewiseAggregateApproximation, SymbolicAggregateApproximation, SymbolicFourierApproximation
from pyts.transformation import ROCKET
from pyts.classification import BOSSVS
from pyts.classification import KNeighborsClassifier as KNeighborsClassifierTS
from pyts.preprocessing import MinMaxScaler as MinMaxScalerTS
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import FeatureUnion
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import f1_score, balanced_accuracy_score, make_scorer   
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import MinMaxScaler

from sktime.classification.interval_based import TimeSeriesForest
from sktime.utils.data_container import concat_nested_arrays as cna
from sktime.classification.frequency_based import RandomIntervalSpectralForest

# # Params
# 

# In[3]:

#Base path, all necessary folders are supposed to be contained in this one.
base_path = r"/home/prof/guillaume/"

#Path to the life cycles CSV files.
dataset_path = base_path+r"datasets/"

#Path where to export the results of cross validation  
result_path = base_path+r"results/"

#If not None, CSV files containing data used by the TS-CHIEF java program will be outputed
TSCHIEF_path = dataset_path+r"TSCHIEF/"

#If True, perform cross validation of all defined pipelines
do_cross_validation = True

#If not None, will output results of cross validation as a latex file (csv results are still exported)
produce_latex = base_path+'results.tex'

#Separator used when procuding csv outputs
csv_separator = ';'

#Size of the predicitve padding
predictive_padding_hours = 48

#Extend the infected interval to cover restart process
extended_infected_interval_hours = 24

#Size of the PAA transform output
size=1000

#Number of cross validation splits
n_splits=10

if dataset_path is not None and not exists(dataset_path):
    mkdir(dataset_path)
if result_path is not None and not exists(result_path):
    mkdir(result_path)
if TSCHIEF_path is not None and not exists(TSCHIEF_path):
    mkdir(TSCHIEF_path)
    
# # Import data

# In this experiment, we consider life cycle data coming from X ATMs in France using the same protocol. Only life of at least that seven days are considered.
# 
# CSV files are formatted as follow : `Cycle_{}_{}_{}.csv` with in that order : ATM id, life cycle id, state in place of brackets

# In[4]:


def process_cycle(file_name, path, predictive_interval, infected_interval):
    """
    Read a csv file containing the information of a life cycle and apply predictive and infected intervals

    Parameters
    ----------
    file_name : str
        The name of the life cycle csv file to process
    
    path : str
        The full path to the dataset repository
        
    predictive_interval : int
        Predictive interval to apply in hours
                
    infected_interval : int
        Infected interval to apply in hours

    Returns
    -------
    A Couple (x,y) with x a Pandas DataFrame containing the log data of a life cycle
    and y the class of the life cycle
    """
    _ ,atm_id, cycle_id, y = file_name.split('.')[0].split('_')
    y = 0 if y == 'LIVE' else 1
    print("Processing ATM N°{}, cycle N°{}".format(atm_id,cycle_id),end='\r')
    #Read File
    data = pd.read_csv(path+file_name)
    data['date'] = pd.to_datetime(data['date'])
    
    #Apply Predictive interval
    date = pd.Timestamp((data.iloc[-1]['date'] - timedelta(hours=predictive_interval)))
    data.drop(data[data['date'] >= date].index,axis=0,inplace=True)
    #Apply infected interval
    if data.shape[0] > 0:
        date = pd.Timestamp((data.iloc[0]['date'] + timedelta(hours=infected_interval)))
        data.drop(data[data['date'] <= date].index,axis=0,inplace=True)
        #Reset index
        data.reset_index(drop=True,inplace=True)
        if data.shape[0] > 0:
            return (data, y)
        else:
            return None
    else:
        return None

file_list = [f for f in listdir(dataset_path) if (isfile(join(dataset_path, f)) and f.endswith('.csv'))]
life_cycles = np.asarray([process_cycle(file_name, dataset_path,
                             predictive_padding_hours,
                             extended_infected_interval_hours) for file_name in file_list])

print('\nData Loaded')
# # Define data representation functions

# In[5]:


codes = []
for x in [data[0]['cod_evt'].unique() for data in life_cycles if data is not None]:
    codes.extend(x) 
codes = np.unique(codes) #Unique event codes present in the data, in increasing order 

# In[6]:

##

def get_R1_dict(codes):
    return {x : i for i,x in enumerate(codes)}

def get_R2_dict(codes):
    return {x : int(x[2:]) for x in codes}


def get_R3_dict(codes, spacing=200):
    OK_codes = ["GA01000","GA02000","GA03000","GA04000","GA05000","GA06000","GA07000","GA10000","GA10011",
           "GA10015","GA10020","GA10024","GA10031","GA10500","GA11000","GA11500","GA12000",
           "GA12500","GA13000","GA13500","GA14000","GA14500","GA15000","GA15100","GA15200",
           "GA17000","GA17002","GA20000","GA21000"]
    OK_codes = [x for x in codes if x in OK_codes]
    
    WAR_codes = ["GA02002","GA02003","GA02005","GA02006","GA02007","GA02008","GA03002","GA03003","GA03004",
                "GA03005","GA03006","GA04002","GA04003","GA04004","GA04005","GA04006","GA04006","GA04007",
                "GA05002","GA05003","GA06002","GA07002","GA07003","GA08001","GA08002","GA08003","GA10013",
                "GA10017","GA10018","GA11002","GA11003","GA11004","GA12002","GA12003","GA12004","GA13002",
                "GA13003","GA13004","GA14002","GA14003","GA14004","GA11504","GA12504","GA13504","GA14504",
                "GA15002","GA15003","GA15101","GA15102","GA15103","GA15201","GA15202","GA15203","GA19002",
                 "GA19003","GA19004","GA19005","GA20002","GA20003","GA20004","GA20005","GA21001","GA21002",
                 "GA21003"]
    WAR_codes = [x for x in codes if x in WAR_codes]
    
    KO_codes = ["GA01001","GA02001","GA03001","GA04001","GA05001","GA06001","GA07001","GA10001","GA10012",
               "GA10016","GA10021","GA10025","GA10032","GA10501","GA11001","GA12001",
               "GA13001","GA14001","GA15001","GA15102","GA15202",
               "GA17001","GA17003","GA20001","GA21004"]
    KO_codes = [x for x in codes if x in KO_codes]
        
    R_codes = ["GA40000","GA41000","GA42000"]
    R_codes = [x for x in codes if x in R_codes]

    I_codes = ["GA30000"]
    I_codes = [x for x in codes if x in I_codes]
    
    if set(OK_codes + WAR_codes + KO_codes + R_codes + I_codes) != set(codes):
        warnings.warn("get_R3_dict : Following codes did not fit in the OK/KO/WAR paradigm or were not related to hardware events and were discarded:\n{}".format(set(codes)-set(OK_codes + WAR_codes + KO_codes  + R_codes + I_codes)))

    k = 0
    dict_codes = {}
    for cods in [KO_codes,WAR_codes,I_codes,OK_codes,R_codes]:
        for i, code in enumerate(cods):
            dict_codes.update({code:k+i})
        k+=spacing
    return dict_codes
    
def get_R4_dict(codes):
    vals = np.arange(codes.shape[0])
    np.random.shuffle(vals)
    return {code : vals[i] for i,code in enumerate(codes)}

def apply_code_dict(df, code_dic, code_column='cod_evt'):
    mask = df[~df[code_column].isin(list(code_dic.keys()))].index
    if mask.shape[0] > 0:
        df = df.drop(mask,axis=0)
        df = df.reset_index(drop=True)
    df[code_column] = df[code_column].apply(lambda x: code_dic[x])
    return df

# In[7]:
# # Define classes for transformation and models

# Here we define custom classes when necessary for the transformations and model we will use inside pipelines during cross validation. 
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


# ## Transformations

# In[8]:


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


# ## Models


# In[10]:


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


# # Define pipelines

# We now define the pipelines that we will use for crossvalidation

# In[11]:


pipeline_dict = {}
#FLATTENED IMAGE CLASSIFIERS

pipeline_dict.update({"PAA Gramian Flat RF":make_pipeline(Gramian_transform(flatten=True),
                                                          Random_Forest())})


pipeline_dict.update({"PAA Recurrence Flat RF":make_pipeline(Recurrence_transform(flatten=True),
                                                             Random_Forest())})


pipeline_dict.update({"PAA Gramian Flat SVM":make_pipeline(Gramian_transform(flatten=True),
                                                           MinMaxScaler(),
                                                           SelectFromModel(ExtraTreesClassifier(n_estimators=300, class_weight="balanced_subsample"), threshold=0.000001),
                                                           SVM_classif())})


pipeline_dict.update({"PAA Recurrence Flat SVM":make_pipeline(Recurrence_transform(flatten=True),
                                                              MinMaxScaler(),
                                                              SelectFromModel(ExtraTreesClassifier(n_estimators=300, class_weight="balanced_subsample"), threshold=0.000001),
                                                              SVM_classif())})
                                                              
pipeline_dict.update({"PAA Gramian Flat KNN":make_pipeline(Gramian_transform(flatten=True),
                                                           MinMaxScaler(),
                                                           SelectFromModel(ExtraTreesClassifier(n_estimators=300, class_weight="balanced_subsample"), threshold=0.000001),
                                                           KNN_classif())})

pipeline_dict.update({"PAA Recurrence Flat KNN":make_pipeline(Recurrence_transform(flatten=True),
                                                              MinMaxScaler(),
                                                              SelectFromModel(ExtraTreesClassifier(n_estimators=300, class_weight="balanced_subsample"), threshold=0.000001),
                                                              KNN_classif())})

pipeline_dict.update({"PAA Gramian Flat Ridge":make_pipeline(Gramian_transform(flatten=True),
                                                             MinMaxScaler(),
                                                           SelectFromModel(ExtraTreesClassifier(n_estimators=300, class_weight="balanced_subsample"), threshold=0.000001),
                                                           Ridge_classif())})


pipeline_dict.update({"PAA Recurrence Flat Ridge":make_pipeline(Recurrence_transform(flatten=True),
                                                                MinMaxScaler(),
                                                              SelectFromModel(ExtraTreesClassifier(n_estimators=300, class_weight="balanced_subsample"), threshold=0.000001),
                                                              Ridge_classif())})

#TIME SERIE CLASSIFIERS + PAA
pipeline_dict.update({"PAA TSRF":make_pipeline(PiecewiseApproximation_transform(output_size=size),
                                               TimeSeries_Forest())})


pipeline_dict.update({"PAA BOSSVS":make_pipeline(PiecewiseApproximation_transform(output_size=size),
                                                 BOSSVS_classif())})


pipeline_dict.update({"PAA KNN":make_pipeline(PiecewiseApproximation_transform(output_size=size),
                                              KNN_TS_classif())})


pipeline_dict.update({"PAA RISE":make_pipeline(PiecewiseApproximation_transform(output_size=size),
                                               RISE())})


#TIME SERIE CLASSIFIERS + PAA + SAX

pipeline_dict.update({"PAA SAX TSRF":make_pipeline(PiecewiseApproximation_transform(output_size=size),
                                                   SymbolicAggregate_transform(),
                                                   TimeSeries_Forest())})



pipeline_dict.update({"PAA SAX BOSSVS":make_pipeline(PiecewiseApproximation_transform(output_size=size),
                                                   SymbolicAggregate_transform(),
                                                   BOSSVS_classif())})



pipeline_dict.update({"PAA SAX KNN":make_pipeline(PiecewiseApproximation_transform(output_size=size),
                                                  SymbolicAggregate_transform(),
                                                  KNN_TS_classif())})



pipeline_dict.update({"PAA SAX RISE":make_pipeline(PiecewiseApproximation_transform(output_size=size),
                                                   SymbolicAggregate_transform(),
                                                   RISE())})


#TIME SERIE CLASSIFIERS + PAA + SFA
pipeline_dict.update({"PAA SFA TSRF":make_pipeline(PiecewiseApproximation_transform(output_size=size),
                                                   SymbolicFourrier_transform(),
                                                   TimeSeries_Forest())})

#BOSSVS natively perform SFA on input so no point in testing it here

pipeline_dict.update({"PAA SFA KNN":make_pipeline(PiecewiseApproximation_transform(output_size=size),
                                                  SymbolicFourrier_transform(),
                                                  KNN_TS_classif())})

#RISE apply techniques such as power spectrum and autocorrelation that are supposed to be applied in the time domain.
#SFA use Fourrier transform (DFT) and they binning with MCB, the result of this operation is not in the time domain anymore.



#TIME SERIE CLASSIFIERS + PAA + MATRIX PROFILE

pipeline_dict.update({"PAA MP TSRF":make_pipeline(PiecewiseApproximation_transform(output_size=size),
                                                   MatrixProfile_transform(),
                                                   TimeSeries_Forest())})



pipeline_dict.update({"PAA MP BOSSVS":make_pipeline(PiecewiseApproximation_transform(output_size=size),
                                                   MatrixProfile_transform(),
                                                   BOSSVS_classif())})



pipeline_dict.update({"PAA MP KNN":make_pipeline(PiecewiseApproximation_transform(output_size=size),
                                                  MatrixProfile_transform(),
                                                  KNN_TS_classif())})

pipeline_dict.update({"PAA MP RISE":make_pipeline(PiecewiseApproximation_transform(output_size=size),
                                                  MatrixProfile_transform(),
                                                  RISE())})




#Rocket transform
pipeline_dict.update({"PAA ROCKET RF":make_pipeline(PiecewiseApproximation_transform(output_size=size),
                                                    ROCKET_transform(flatten=True),
                                                    Random_Forest())})

pipeline_dict.update({"PAA ROCKET SVM":make_pipeline(PiecewiseApproximation_transform(output_size=size),
                                                     ROCKET_transform(flatten=True),
                                                     MinMaxScaler(),
                                                     SelectFromModel(ExtraTreesClassifier(n_estimators=300, class_weight="balanced_subsample"), threshold=0.000001),
                                                     SVM_classif())})

pipeline_dict.update({"PAA ROCKET KNN":make_pipeline(PiecewiseApproximation_transform(output_size=size),
                                                     ROCKET_transform(flatten=True),
                                                     MinMaxScaler(),
                                                     SelectFromModel(ExtraTreesClassifier(n_estimators=300, class_weight="balanced_subsample"), threshold=0.000001),
                                                     KNN_classif())})

pipeline_dict.update({"PAA ROCKET Ridge":make_pipeline(PiecewiseApproximation_transform(output_size=size),
                                                     ROCKET_transform(flatten=True),
                                                     MinMaxScaler(),
                                                     SelectFromModel(ExtraTreesClassifier(n_estimators=300, class_weight="balanced_subsample"), threshold=0.000001),
                                                     Ridge_classif())})

pipeline_dict.update({"PAA MP ROCKET RF":make_pipeline(PiecewiseApproximation_transform(output_size=size),
                                                       MatrixProfile_transform(), 
                                                       ROCKET_transform(flatten=True),
                                                       Random_Forest())})



pipeline_dict.update({"PAA MP ROCKET SVM":make_pipeline(PiecewiseApproximation_transform(output_size=size),
                                                        MatrixProfile_transform(),
                                                        ROCKET_transform(flatten=True),
                                                        MinMaxScaler(),
                                                        SelectFromModel(ExtraTreesClassifier(n_estimators=300, class_weight="balanced_subsample"), threshold=0.000001),
                                                        SVM_classif())})

pipeline_dict.update({"PAA MP ROCKET KNN":make_pipeline(PiecewiseApproximation_transform(output_size=size),
                                                        MatrixProfile_transform(),
                                                        ROCKET_transform(flatten=True),
                                                        MinMaxScaler(),
                                                        SelectFromModel(ExtraTreesClassifier(n_estimators=300, class_weight="balanced_subsample"), threshold=0.000001),
                                                        KNN_classif())})

pipeline_dict.update({"PAA MP ROCKET Ridge":make_pipeline(PiecewiseApproximation_transform(output_size=size),
                                                        MatrixProfile_transform(),
                                                        ROCKET_transform(flatten=True),
                                                        MinMaxScaler(),
                                                        SelectFromModel(ExtraTreesClassifier(n_estimators=300, class_weight="balanced_subsample"), threshold=0.000001),
                                                        Ridge_classif())})

pipeline_dict.update({"PAA SAX ROCKET RF":make_pipeline(PiecewiseApproximation_transform(output_size=size),
                                                       SymbolicAggregate_transform(), 
                                                       ROCKET_transform(flatten=True),
                                                       Random_Forest())})

pipeline_dict.update({"PAA SAX ROCKET Ridge":make_pipeline(PiecewiseApproximation_transform(output_size=size),
                                                       SymbolicAggregate_transform(), 
                                                       ROCKET_transform(flatten=True),
                                                       MinMaxScaler(),
                                                       SelectFromModel(ExtraTreesClassifier(n_estimators=300, class_weight="balanced_subsample"), threshold=0.000001),
                                                       Ridge_classif())})

pipeline_dict.update({"PAA SAX ROCKET SVM":make_pipeline(PiecewiseApproximation_transform(output_size=size),
                                                        SymbolicAggregate_transform(),
                                                        ROCKET_transform(flatten=True),
                                                        MinMaxScaler(),
                                                        SelectFromModel(ExtraTreesClassifier(n_estimators=300, class_weight="balanced_subsample"), threshold=0.000001),
                                                        SVM_classif())})

pipeline_dict.update({"PAA SAX ROCKET KNN":make_pipeline(PiecewiseApproximation_transform(output_size=size),
                                                        SymbolicAggregate_transform(),
                                                        ROCKET_transform(flatten=True),
                                                        MinMaxScaler(),
                                                        SelectFromModel(ExtraTreesClassifier(n_estimators=300, class_weight="balanced_subsample"), threshold=0.000001),
                                                        KNN_classif())})

pipeline_dict.update({"PAA MP Gramian + Recurrence RF":make_pipeline(PiecewiseApproximation_transform(output_size=size),
    MatrixProfile_transform(),
    FeatureUnion([
        ("gramian",Gramian_transform(flatten=True)),
        ("recurrence",Recurrence_transform(flatten=True))
        ]),
    Random_Forest())})


pipeline_dict.update({"PAA MP Gramian + Recurrence SVM":make_pipeline(PiecewiseApproximation_transform(output_size=size),
    MatrixProfile_transform(),
    FeatureUnion([
        ("gramian",Gramian_transform(flatten=True)),
        ("recurrence",Recurrence_transform(flatten=True))
        ]),
    MinMaxScaler(),
    SelectFromModel(ExtraTreesClassifier(n_estimators=300, class_weight="balanced_subsample"), threshold=0.000001),
    SVM_classif())})

pipeline_dict.update({"PAA MP Gramian + Recurrence KNN":make_pipeline(PiecewiseApproximation_transform(output_size=size),
    MatrixProfile_transform(),
    FeatureUnion([
        ("gramian",Gramian_transform(flatten=True)),
        ("recurrence",Recurrence_transform(flatten=True))
        ]),
    MinMaxScaler(),
    SelectFromModel(ExtraTreesClassifier(n_estimators=300, class_weight="balanced_subsample"), threshold=0.000001),
    KNN_classif())})                                                        

pipeline_dict.update({"PAA Gramian + Recurrence RF":make_pipeline(PiecewiseApproximation_transform(output_size=size),
    FeatureUnion([
        ("gramian",Gramian_transform(flatten=True)),
        ("recurrence",Recurrence_transform(flatten=True))
        ]),
    Random_Forest())})


pipeline_dict.update({"PAA Gramian + Recurrence SVM":make_pipeline(PiecewiseApproximation_transform(output_size=size),
    FeatureUnion([
        ("gramian",Gramian_transform(flatten=True)),
        ("recurrence",Recurrence_transform(flatten=True))
        ]),
    MinMaxScaler(),
    SelectFromModel(ExtraTreesClassifier(n_estimators=300, class_weight="balanced_subsample"), threshold=0.000001),
    SVM_classif())})

pipeline_dict.update({"PAA Gramian + Recurrence KNN":make_pipeline(PiecewiseApproximation_transform(output_size=size),
    FeatureUnion([
        ("gramian",Gramian_transform(flatten=True)),
        ("recurrence",Recurrence_transform(flatten=True))
        ]),
    MinMaxScaler(),
    SelectFromModel(ExtraTreesClassifier(n_estimators=300, class_weight="balanced_subsample"), threshold=0.000001),
    KNN_classif())})

#ROCKET on SFA is not efficient, rocket can already extract frequency based features due to the nature of convolutional kernels.
         
"""
#This section is left commented so you have no trouble running the script without Tensorflow/GPU
#If you have error during cross validation, you can try to make the class ResNetV2
# inherit the tensorflow.keras KerasClassifier wrapper, it can fix some issues.

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

		

pipeline_dict.update({"PAA Gramian ResNet50V2":make_pipeline(Gramian_transform(flatten=True),
                                                             ResNetV2())})


pipeline_dict.update({"PAA Recurrence ResNet50V2":make_pipeline(Recurrence_transform(flatten=True),
																ResNetV2())})
"""

		 
print('Pipelines initialised')

# In[12]:

# Critical Failure index (CFI). As True Negatives implies that no maintenance is schedule (so no business impact),
# this measure indicate how many maintenance operation we "missed" (False Negatives) plus how many we did
# while it was not necessary to do so (False Positives). Then those two variables are summed and 
# divided by their sum plus the number of successful prediction (True Positives). 
# In short, the closer to 0, the more the system is "business" efficient.
def CFI(y_test, y_pred):
    if type(y_test) == list:
        y_test = np.asarray(y_test)
    if type(y_pred) == list:
        y_pred = np.asarray(y_pred)
    tp = len(list(set(np.where(y_test == 1)[0]) &  set(np.where(y_pred == 1)[0])))
    fp = len(list(set(np.where(y_test == 0)[0]) &  set(np.where(y_pred == 1)[0])))
    fn = len(list(set(np.where(y_test == 1)[0]) &  set(np.where(y_pred == 0)[0])))
    return (fp+fn)/(tp+fp+fn) if (tp+fp+fn) > 0 else 1

def report(pipeline, r ,b_accs, cfis, f1s, decimals=3):
    print("\n------ REPORT FOR {} on {} ------".format(pipeline,r))
    print("-- Balanced accuracy {}(+/-{}) ------".format(np.around(np.mean(b_accs),decimals=decimals),np.around(np.std(b_accs),decimals=decimals)))
    print("-- Critical failure index {}(+/-{}) ------".format(np.around(np.mean(cfis),decimals=decimals),np.around(np.std(cfis),decimals=decimals)))
    print("-- F1 score {}(+/-{}) ------".format(np.around(np.mean(f1s),decimals=decimals),np.around(np.std(f1s),decimals=decimals)))


# In[13]:


df_res = pd.DataFrame(columns=['name','representation','balanced accuracy mean', 'CFI mean', 'F1 score mean', 'Fit time mean','Score time mean'
                               'balanced accuracy std', 'CFI std', 'F1 score std','Fit time std','Score time std'])


# In[14]:
    

print('Cross Validation')
order = {0:'R1',1:'R2',2:'R3',3:'R4'}

print("A total of {} runs will be launched".format(len(pipeline_dict)*n_splits*len(order)))

for i_r, dic_func in enumerate([get_R1_dict, get_R2_dict, get_R3_dict, get_R4_dict]):
    X = np.asarray([apply_code_dict(x.copy(deep=True),dic_func(codes))['cod_evt'].values for x in life_cycles[:,0] if x is not None],dtype=object)
    y = np.asarray([x[1] for x in life_cycles if x is not None]).astype(int)
    
    idx = np.where([x.shape[0]>=size for x in X])[0]
    X = X[idx]
    y = y[idx]
    if TSCHIEF_path is not None:
        skf = StratifiedKFold(n_splits=n_splits)
        
        paa = PiecewiseApproximation_transform(size)
        X_paa = paa.transform(X)
        y_paa = y
        
        df = pd.DataFrame(data = {i: x.reshape(-1) for i,x in enumerate(X_paa)}).transpose()
        df[size]=y_paa
        df = df.astype(np.float32)
        i_split=0
        for train_idx, test_idx in skf.split(X,y):
            df.loc[train_idx].to_csv(TSCHIEF_path+'data_Train_{}_{}_{}.csv'.format(size, i_split, order[i_r]),index=False,header=False)
            df.loc[test_idx].to_csv(TSCHIEF_path+'data_Test_{}_{}_{}.csv'.format(size, i_split, order[i_r]),index=False,header=False)
            i_split+=1


    if do_cross_validation:
        for pipeline in pipeline_dict:   
            try:
                cv = cross_validate(pipeline_dict[pipeline],X, y, cv=n_splits, n_jobs=-1,
                                     scoring={'b_a':make_scorer(balanced_accuracy_score),
                                              'cfi':make_scorer(CFI),
                                              'f1':make_scorer(f1_score)})   

            except Exception as e:
                print(e)
                df_res = pd.concat([df_res,pd.DataFrame({'name':[pipeline],
                                                         'representation':[order[i_r]],
                                                         'balanced accuracy mean':[pd.NA],
                                                         'CFI mean':[pd.NA],
                                                         'F1 score mean':[pd.NA],
                                                         'Fit time mean':[pd.NA],
                                                         'Score time mean':[pd.NA],
                                                         'balanced accuracy std':[pd.NA],
                                                         'CFI std':[pd.NA],
                                                         'F1 score std':[pd.NA],
                                                         'Fit time std':[pd.NA],
                                                         'Score time std':[pd.NA]})
                                                        ])
            else:
                report(pipeline, order[i_r], cv['test_b_a'], cv['test_cfi'], cv['test_f1'])
                df_res = pd.concat([df_res,pd.DataFrame({'name':[pipeline],
                                                         'representation':[order[i_r]],
                                                         'balanced accuracy mean':[np.mean(cv['test_b_a'])],
                                                         'CFI mean':[np.mean(cv['test_cfi'])],
                                                         'F1 score mean':[np.mean(cv['test_f1'])],
                                                         'Fit time mean':[np.mean(cv['fit_time'])],
                                                         'Score time mean':[np.mean(cv['score_time'])],
                                                         'balanced accuracy std':[np.std(cv['test_b_a'])],
                                                         'CFI std':[np.std(cv['test_cfi'])],
                                                         'F1 score std':[np.std(cv['test_f1'])],
                                                         'Fit time std':[np.std(cv['fit_time'])],
                                                         'Score time std':[np.std(cv['score_time'])]})
                                                         ])

# In[15]:

    df_res.to_csv(result_path+'cv_results.csv',sep=csv_separator, index=False)
    if produce_latex is not None:
        df_dict = {'name':df_res['name'].unique()}
        for col in ['balanced accuracy','CFI','F1 score','Fit time','Score time']:
            for r in ['R1', 'R2', 'R3','R4']:
                df_dict.update({col+' '+r:(df_res[df_res['representation']==r][col + ' mean'].astype(str).str[0:5] + '(+/- '+df_res[df_res['representation']==r][col+' std'].astype(str).str[0:5]+')').reset_index(drop=True)})
    df_latex = pd.DataFrame(df_dict)
    df_latex.to_csv(result_path+'cv_results_latex.csv',sep=csv_separator, index=False)
    latex_str = df_latex.sort_values(by=['CFI R3'],ascending=True).to_latex(index=False)
	with open(produce_latex, 'w') as f:
		f.write(latex_str)

