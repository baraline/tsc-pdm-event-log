# # Imports

# In[2]:

import pandas as pd
import numpy as np
import warnings

from os import listdir, mkdir
from os.path import isfile, join, exists
from datetime import timedelta


#You should be able to ignore import * warnings safely, otherwise import
# each class name of both files. 
from utils.representations import *
from utils.classifications import *

from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import FeatureUnion
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import f1_score, balanced_accuracy_score, make_scorer   
from sklearn.preprocessing import MinMaxScaler


# # Params
# 

# In[3]:

#Base path, all necessary folders are supposed to be contained in this one.
base_path = r"!! REPLACE BY YOUR PATH !!"

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

#Extend the infected interval to cover restart process, setting it to 0 will introduce bias.
extended_infected_interval_hours = 24

#Size of the PAA transform output
size=1000

#Number of cross validation splits
n_splits=10

# Number of process to launch in parallel for cross validation of each pipeline.
# Set to None if you don't have the setup to allow such speedups. 
n_cv_jobs=None

if dataset_path is not None and not exists(dataset_path):
    mkdir(dataset_path)
if result_path is not None and not exists(result_path):
    mkdir(result_path)
if TSCHIEF_path is not None and not exists(TSCHIEF_path):
    mkdir(TSCHIEF_path)
    
# # Import data

# In this experiment, we consider life cycle data coming from ATMs. Only life cycle of at least that seven days are considered.
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

# # Define data encoding functions

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


# # Define pipelines

# We now define the pipelines that we will use for crossvalidation

# In[11]:

pipeline_dict = {}
#FLATTENED IMAGE

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



#PAA + ROCKET
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


#PAA + MATRIX PROFILE + ROCKET
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

#PAA + SAX + ROCKET
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

#ROCKET on SFA is not efficient, rocket can already extract frequency based features due to the nature of convolutional kernels.

#PAA + MP + STACKED FLAT IMAGES
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

#PAA + STACKED FLAT IMAGES
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
         
"""
#This section is left commented so you have no trouble running the script without Tensorflow/GPU
#If you have error during cross validation, you can try to make the class ResNetV2
# inherit the tensorflow.keras KerasClassifier wrapper, it can fix some issues.
        

pipeline_dict.update({"PAA Gramian ResNet50V2":make_pipeline(Gramian_transform(flatten=True),
                                                             ResNetV2())})


pipeline_dict.update({"PAA Recurrence ResNet50V2":make_pipeline(Recurrence_transform(flatten=True),
                                                                ResNetV2())})
"""

         
print('Pipelines initialized')

# In[12]:

# Critical Failure Index (CFI). As True Negatives implies that no maintenance is scheduled (so no business impact),
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
                cv = cross_validate(pipeline_dict[pipeline],X, y, cv=n_splits, n_jobs=n_cv_jobs,
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

