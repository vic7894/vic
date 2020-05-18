#%%
import os 
import pandas as pd
import numpy as np
os.chdir('c:\\Users\\SA\\python\\練習py')
from feature_selector import  FeatureSelector
path='c:\\Users\\SA\\python\\Transaction Prediction'
os.chdir(path)
df=pd.read_csv('train.csv')
y=df.iloc[:,1]
x=df.iloc[:,2:]
print(df.head(5))
print(x.shape)
print(y.shape)
df.head()
# %%
tt=pd.DataFrame()
def miss_data(data):
    total=data.isnull().sum()
    types=[]
    for col in data :
        dtype=str(data[col].dtype)
        types.append(dtype)
    tt['Total']=total
    tt['Types']=types
    return (np.transpose(tt))

miss_data(df)
# %%
fs=FeatureSelector(x,y)
fs.identify_missing (missing_threshold=0.6)
fs.record_missing.head()
fs.plot_missing()

# %%
fs.identify_collinear(correlation_threshold=0.9)
fs.record_collinear.head()

# %%
fs.identify_zero_importance(task = 'classification',  
                            n_iterations = 10, 
                             early_stopping = False)
# list of zero importance features
zero_importance_features = fs.ops['zero_importance']

# %%
fs.identify_low_importance(cumulative_importance=0.99)
fs.record_low_importance.head()
fs.plot_feature_importances(100)
#%%
train_removed=fs.remove(methods='all')
X_clean=train_removed

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(y , palette='Set3')

# %%
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.ensemble import  RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import log_loss,f1_score,make_scorer,roc_auc_score
#%%
#X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
score=make_scorer(roc_auc_score)

model=[RandomForestClassifier(),GradientBoostingClassifier(),xgb.XGBClassifier(),lgb.LGBMClassifier()]
model_mean=[]
model_std=[]
for i in model:
    cross_score=cross_val_score(i,x,y,scoring=score,n_jobs=2)
    model_mean.append(np.mean(cross_score))
    model_std.append(np.std(cross_score))
model_results=pd.DataFrame({'models':['RandomForestClassifier','GradientBoostingClassifier','xgb','lgb'],
                            'model_mean':model_mean,
                            'model_std':model_std})

model_results


# %%
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss,f1_score,make_scorer,roc_auc_score
param = {
    'bagging_freq': 5,
    'bagging_fraction': 0.4,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.05,
    'learning_rate': 0.01,
    'max_depth': -1,  
    'metric':'auc',
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 13,
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'binary', 
    'verbosity': 1
}
clf=lgb.LGBMClassifier( bagging_freq= 5,
     bagging_fraction=0.4,
     boost_from_average=False,
     boost= 'gbdt',
     feature_fraction= 0.05,
     learning_rate= 0.01,
     max_depth= -1,  
     metric='auc',
     min_data_in_leaf= 80,
     min_sum_hessian_in_leaf= 10.0,
     num_leaves= 13,
     num_threads= 8,
     tree_learner= 'serial',
     objective='binary', 
     verbosity= 1)

# params_test1={
#     'bagging_freq':[4,5,6],
#     'bagging_fraction':[0.3,0.4,0.5],
#     'learning_rate': [0.1,0.01,0.001],
#     'min_data_in_leaf':[ 70,80,90],
#     'num_leaves':[11,12,13,14]
# }
params_test1={   
    'learning_rate': [0.1,0.01],
    'num_leaves':[20,100]
}
gsearch1 = GridSearchCV(estimator=clf, param_grid=params_test1, scoring=make_scorer(roc_auc_score), cv=5, verbose=2, n_jobs=3)
gsearch1.fit(x,y)
#%%
#print(gsearch1.score)
print(gsearch1.best_params_)
print(gsearch1.best_score_)

# %%
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
cv= StratifiedKFold(n_splits=3,random_state=42)
model=lgb.LGBMClassifier( bagging_freq= 5,
     bagging_fraction=0.3,
     boost_from_average=False,
     boost= 'gbdt',
     feature_fraction= 0.05,
     learning_rate= 0.1,
     max_depth= -1,  
     metric='auc',
     min_data_in_leaf= 70,
     min_sum_hessian_in_leaf= 10.0,
     num_leaves= 14,
     num_threads= 8,
     tree_learner= 'serial',
     objective='binary', 
     verbosity= 1)
x=x.values
y=y.values
scores=[]
scores2=[]
for train_idx, test_idx in cv.split(x, y):
    x_train=x[train_idx]
    x_test=x[test_idx]
    y_train=y[train_idx]
    y_test=y[test_idx]
    
    model.fit(x_train,y_train)
    y_proba = model.predict_proba(x_test)
    y_pred = np.zeros(y_proba.shape[0])
    y_pred[y_proba[:,1] >= 0.166] = 1
    
    score = roc_auc_score(y_test, y_pred)
    score2 = accuracy_score(y_test, y_pred)
    print(score)
    print(score2)
    scores.append(score)
    scores2.append(score2)

print(np.round(np.mean(scores),4))
print(np.round(np.std(scores), 4))

print(np.round(np.mean(scores2),4))
print(np.round(np.std(scores2), 4))

# %%
