#%%
import os 
import pandas as pd
import numpy as np
os.chdir('c:\\Users\\SA\\python\\練習py')
from feature_selector import  FeatureSelector
path='c:\\Users\\SA\\python\\Santander Value Prediction Challenge'
os.chdir(path)
df=pd.read_csv('train.csv')
y=df.iloc[:,1]
x=df.iloc[:,2:]

print(x.shape)
print(y.shape)
df.head()
#%%
import matplotlib.pyplot as plt
import seaborn as sns
fig,ax=plt.subplots(3,1,figsize=(5,6),squeeze=False)
ax[0][0].hist(y,bins=100)

ax[1][0].hist(np.log1p(y),bins=100)# np.log(1+y.values)

sns.set_style("whitegrid")
ax[2][0] = sns.violinplot(x=np.log1p(y))

plt.xlabel('count')
plt.ylabel('target')
plt.show()
print(np.log1p(y).describe())
y=np.log1p(y)
#%%
constant_train = x.loc[:, (x == 0).all()].columns.tolist()
print('Number of constant columns in the train set:', len(constant_train))

# %%
fs=FeatureSelector(x,y)
fs.identify_missing (missing_threshold=0.6)
fs.record_missing.head()
fs.plot_missing()

# %%
fs.identify_collinear(correlation_threshold=0.9)
fs.record_collinear.head()

# %%
fs.identify_zero_importance(task = 'regression', 
                            eval_metric = 'rmse', 
                            n_iterations = 10, 
                            early_stopping = True)
# list of zero importance features
zero_importance_features = fs.ops['zero_importance']
#%%
fs.identify_low_importance(cumulative_importance=0.99999)
fs.record_low_importance.head()
fs.plot_feature_importances(50)

# %%
train_removed=fs.remove(methods='all')
X_clean=train_removed
# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error,make_scorer
#%%
scores=make_scorer(mean_squared_error)
models=[RandomForestRegressor(n_estimators=200,max_depth=3,verbose=2,random_state=42)
,GradientBoostingRegressor(random_state=42),lgb.LGBMRegressor(random_state=42),xgb.XGBRFRegressor(random_state=42)]
model_mean=[]
model_std=[]
#%%
for i in models :
    cross_score=cross_val_score(i,X_clean,y,scoring=scores,n_jobs=-1,cv=4)
    cross_score=np.sqrt(cross_score)
    model_mean.append(np.mean(cross_score))
    model_std.append(np.std(cross_score))
model_results=pd.DataFrame({'models':['RandomForestRegressor','GradientBoostingRegressor','lgb','xgb'],
                            'model_mean':model_mean,
                            'model_std':model_std})

model_results

#%%
params = {
    'boosting_type': 'gbdt', 
    'objective': 'regression', 

    'learning_rate': 0.1, 
    'num_leaves': 50,                #小於2^max_depth
    'max_depth': 6,

    'subsample': 0.8, 
    'colsample_bytree': 0.8, 
    }

data_train = lgb.Dataset(X_clean, y, silent=True)
cv_results = lgb.cv(
    params, data_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True, metrics='rmse',
    early_stopping_rounds=100, verbose_eval=50, show_stdv=True, seed=0)

print('best n_estimators:', len(cv_results['rmse-mean']))
print('best cv score:', cv_results['rmse-mean'][-1])
# %%
from sklearn.model_selection import GridSearchCV
#%%
clf = lgb.LGBMRegressor(objective='regression',num_leaves=50,
                              learning_rate=0.1, n_estimators=92, max_depth=6,
                              metric='rmse', 
                              bagging_fraction = 0.8,feature_fraction = 0.8 ) 


params_test1={
    'max_depth': range(3,8,2),
    'num_leaves':range(50, 170, 30)
}
gsearch1 = GridSearchCV(estimator=clf, param_grid=params_test1, scoring='neg_mean_squared_error', cv=5, verbose=2, n_jobs=4)
gsearch1.fit(x,y)
for i in ['mean_test_score', 'std_test_score', 'params']:
        print(i," : ",gsearch1.cv_results_[i])
print(gsearch1.best_params_) 
print(gsearch1.best_score_)

# %%
clf = lgb.LGBMRegressor(objective='regression',num_leaves=80,
                              learning_rate=0.1, n_estimators=92, max_depth=7,
                              metric='rmse', min_child_samples=21,
                              bagging_fraction = 0.8,feature_fraction = 0.8 ) 


params_test3={
    'min_child_samples': [18, 19, 20, 21, 22],
    'min_child_weight':[0.001, 0.002]
}
gsearch3 = GridSearchCV(estimator=clf, param_grid=params_test3, scoring='neg_mean_squared_error', cv=5, verbose=2, n_jobs=4)
gsearch3.fit(x,y)

for i in ['mean_test_score', 'std_test_score', 'params']:
        print(i," : ",gsearch3.cv_results_[i])
print(gsearch3.best_params_) 
print(gsearch3.best_score_)

# %%
clf = lgb.LGBMRegressor(objective='regression',num_leaves=80,
                              learning_rate=0.1, n_estimators=92, max_depth=7,
                              metric='rmse', min_child_samples=21,min_child_weight=0.001,
                               ) 

params_test4={
    'feature_fraction': [0.5, 0.6, 0.7, 0.8, 0.9],
    'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 1.0]
}
gsearch4 = GridSearchCV(estimator=clf, param_grid=params_test4, scoring='neg_mean_squared_error', cv=5, verbose=2, n_jobs=4)
gsearch4.fit(x,y)

for i in ['mean_test_score', 'std_test_score', 'params']:
        print(i," : ",gsearch4.cv_results_[i])
print(gsearch4.best_params_) 
print(gsearch4.best_score_)

# %%
clf = lgb.LGBMRegressor(objective='regression',num_leaves=80,
                              learning_rate=0.1, n_estimators=92, max_depth=7,
                              metric='rmse', min_child_samples=21,min_child_weight=0.001,
                               ) 

params_test4={
    'feature_fraction': [0.72,0.75,0.78, 0.8,0.82,0.85,0.88],
    'bagging_fraction': [0.6, 0.8]
}
gsearch5 = GridSearchCV(estimator=clf, param_grid=params_test4, scoring='neg_mean_squared_error', cv=5, verbose=2, n_jobs=4)
gsearch5.fit(x,y)

for i in ['mean_test_score', 'std_test_score', 'params']:
        print(i," : ",gsearch5.cv_results_[i])
print(gsearch5.best_params_) 
print(gsearch5.best_score_)


# %%
clf = lgb.LGBMRegressor(objective='regression',num_leaves=80,
                              learning_rate=0.1, n_estimators=92, max_depth=7,
                              metric='rmse', min_child_samples=21,min_child_weight=0.001,
                              feature_fraction=0.8,bagging_fraction=0.6
                               ) 

params_test6={
    'reg_alpha': [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5],
    'reg_lambda': [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5]
}
gsearch6 = GridSearchCV(estimator=clf, param_grid=params_test6, scoring='neg_mean_squared_error', cv=5, verbose=2, n_jobs=4)
gsearch6.fit(x,y)
#%%
for i in ['mean_test_score', 'std_test_score', 'params']:
        print(i," : ",gsearch6.cv_results_[i])
print(gsearch6.best_params_) 
print(gsearch6.best_score_)

# %%
params = {
    'boosting_type': 'gbdt', 
    'objective': 'regression', 

    'learning_rate': 0.01, 
    'num_leaves': 80,                
    'max_depth': 7,
    'min_data_in_leaf': 21,
    'min_sum_hessian_in_leaf':0.001,

    'subsample': 0.8, 
    'colsample_bytree': 0.6,
    'reg_alpha': 0.5, 'reg_lambda': 0.03 
    }

data_train = lgb.Dataset(X_clean, y, silent=True)
cv_results = lgb.cv(params, data_train, num_boost_round=10000, nfold=5, stratified=False, shuffle=True, metrics='rmse', 
                 early_stopping_rounds=50, verbose_eval=100, show_stdv=True)

print('best n_estimators:', len(cv_results['rmse-mean']))
print('best cv score:', cv_results['rmse-mean'][-1])


# %%

cross_score1=cross_val_score(lgb.LGBMRegressor(objective='regression',num_leaves=80,
                              learning_rate=0.01, n_estimators=813, max_depth=7,
                              metric='rmse', min_child_samples=21,min_child_weight=0.001,
                              feature_fraction=0.8,bagging_fraction=0.6,reg_alpha=0.5,reg_lambda=0.03)
                              ,X_clean,y,scoring=scores,n_jobs=-1,cv=5)
cross_score1=np.sqrt(cross_score1)
mean=np.mean(cross_score1)
std=np.std(cross_score1)
print(mean,'+',std)


# %%
cross_score1=cross_val_score(lgb.LGBMRegressor(objective='regression',num_leaves=200,
                              learning_rate=0.01, n_estimators=813, max_depth=-1,bagging_freq=4,
                              metric='rmse', min_child_samples=21,min_child_weight=10,
                              feature_fraction=0.5,bagging_fraction=0.5,reg_alpha=0.3,reg_lambda=0.1)
                              ,X_clean,y,scoring=scores,n_jobs=-1,cv=5)
cross_score1=np.sqrt(cross_score1)
mean=np.mean(cross_score1)
std=np.std(cross_score1)
print(mean,'+',std)

# %%
