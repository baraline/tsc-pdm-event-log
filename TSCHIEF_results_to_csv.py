import pandas as pd
import numpy as np
from pathlib import Path
from os import listdir
from sklearn.metrics import f1_score, balanced_accuracy_score

base_path = 'results/TSCHIEF/'
out_path =  'results/TSCHIEF/'

def CFI(y_test, y_pred):
    if type(y_test) == list:
        y_test = np.asarray(y_test)
    if type(y_pred) == list
        y_pred = np.asarray(y_pred)
    tp = len(list(set(np.where(y_test == 1)[0]) &  set(np.where(y_pred == 1)[0])))
    fp = len(list(set(np.where(y_test == 0)[0]) &  set(np.where(y_pred == 1)[0])))
    fn = len(list(set(np.where(y_test == 1)[0]) &  set(np.where(y_pred == 0)[0])))
    return (fp+fn)/(tp+fp+fn) if (tp+fp+fn) > 0 else 1

run_list = [f for f in listdir(base_path)]

results_path = np.asarray([[f for f in list(Path(base_path+path+'/').rglob("*.csv")) if f.is_file()] for path in run_list]).reshape(-1)
df_dict = {'name':['TS-CHIEF']}

scorer_dict = {'balanced accuracy':balanced_accuracy_score,
               'CFI':CFI,
               'F1 score':f1_score}

for R in ['R1','R2','R3','R4']:
    R_path = [r for r in results_path if R+'.csv' in str(r)]
    result_dict = {'balanced accuracy':[],
                   'CFI':[],
                   'F1 score':[]}
    for result in R_path:
        df = pd.read_csv(result)
        y_true = df['actual_label']
        y_test = df['predicted_label']
        
        result_dict['balanced accuracy'].append(balanced_accuracy_score(y_true,y_test))
        result_dict['CFI'].append(CFI(y_true,y_test))
        result_dict['F1 score'].append(f1_score(y_true,y_test))
        
    for col in scorer_dict:
        df_dict.update({col+' '+R: str(np.mean(result_dict[col]))[0:5] + '(+/- '+str(np.std(result_dict[col]))[0:5]+')'})

df_res = pd.DataFrame(df_dict)
df_res.to_csv(out_path+'TS-CHIEF_results.csv',index=False)