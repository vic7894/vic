#%%
import pandas as pd
import numpy as np 
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
path='C:\\Users\\SA\\python\\csv'
os.chdir(path)
os.getcwd()
df=pd.read_csv('PastYear_V4.csv',index_col=False)
df.rename(columns={'Unnamed: 0':'ID'},inplace=True)
df['ID']=df['ID'].astype('int32')
X=df.iloc[:,1:-1]
y=df.iloc[:,-1]
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
clf=LogisticRegression()
clf.fit(X_train,y_train)
prediction=clf.predict(X_test)
print('準確率:',metrics.accuracy_score(prediction,y_test))

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
from sklearn.ensemble import RandomForestClassifier
RC=RandomForestClassifier(n_estimators=100,random_state=4)
RC.fit(X_train,y_train)
prediction=RC.predict(X_test)
print('準確率:',metrics.accuracy_score(prediction,y_test))

# %%
from sklearn.feature_selection import SelectKBest, f_regression,chi2
from sklearn.pipeline import make_pipeline
chi2_filter=SelectKBest(chi2,k=3)
X_2=chi2_filter.fit_transform(X,y)
f_regression_filter=SelectKBest(f_regression,k=3)
X_3=f_regression_filter.fit_transform(X,y)
for i in range(17):
    print(X.columns[i],'CHI2:',chi2_filter.pvalues_[i]<=0.01,
    'ANOVA:',f_regression_filter.pvalues_[i]<=0.01)

# %%
from sklearn.feature_selection import RFE
clf=LogisticRegression()
rfe=RFE(estimator=clf,n_features_to_select=2,step=1)
rfe.fit(X,y)
for i in range(17):
    print(X.columns[i],'Rank:',rfe.ranking_[i])

# %%
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
RC=RandomForestClassifier(n_estimators=100,random_state=4)
rfecv=RFECV(estimator=RC,step=1,cv=StratifiedKFold(2),scoring='accuracy')
rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)

import matplotlib.pyplot as plt
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

# %%
# PCA 降維
from sklearn.decomposition import PCA
pca=PCA(n_components=3)
pca.fit(X)

print("Explained Variance: %s" % pca.explained_variance_ratio_)
#print(pca.explained_variance_ratio_)
print(pca.components_)

# %%
#透過模型來挑選特徵
#L1-based特征选择
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
svm=LinearSVC(C=0.01,penalty='l1',dual=False)
svm=svm.fit(X,y)

model=SelectFromModel(svm,prefit=True)
X_svm = model.transform(X)
X_svm.shape

# %%
#透過樹模型挑選特徵
from sklearn.ensemble import ExtraTreesClassifier
etc=ExtraTreesClassifier(n_estimators=100)
etc=etc.fit(X,y)
print(etc.feature_importances_)
model2 = SelectFromModel(etc,prefit=True)
X_etc = model2.transform(X)
X_etc.shape

# %%
#使用挑選出最重要兩特徵進行預測
from sklearn.ensemble import RandomForestClassifier
X1=df.iloc[:,1:3]
rfc=RandomForestClassifier(n_estimators=500,random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=.3)
rfc.fit(X_train,y_train)
prediction=rfc.predict(X_test)
print('準確率:',metrics.accuracy_score(prediction,y_test))
# %%
############################################################################################
#featuretools 建立特徵
import featuretools as fe
#建立資料分群
es=fe.EntitySet(id='df_featuretools')
a=['MAX_LOW_TYPE','APPLIED','HELPED','GRANT_YN','ID','CHANGE_TYPE']
b=['AGE','SEX','ABORIGINE','MARRY','CRIP_LEVEL']
c=['GRANT_YN','VETERAN','SETTLED','FOREIGNER','FOREIGNER_CHILD','SINGLE_TYPE','SINGLE_LIVE','SPC_WOM','TRANJOB_YN']

#匯入EntitySet
es.entity_from_dataframe(entity_id='LOWTYPE',dataframe=df[a],index='ID')
es.entity_from_dataframe(entity_id='IDENTITY',dataframe=df[b+c],make_index=True,index='ID_ID')
#es.entity_from_dataframe(entity_id='SPC_IDY',dataframe=df[c],make_index=True,index='SPC_ID')

#建立Relationship
r1=fe.Relationship(es['IDENTITY']['ID_ID'],es['LOWTYPE']['ID'])
#r2=fe.Relationship(es['SPC_IDY']['SPC_ID'],es['LOWTYPE']['ID'])

es=es.add_relationship(r1)
#es=es.add_relationship(r2)
# %%
feature,feature_names=fe.dfs(entityset=es,target_entity='IDENTITY',max_depth=2,verbose=1,n_jobs=1)

col_m=[]
for col in feature:
    if 'CHANGE_TYPE' not in col:
        col_m.append(col)
feature_matrix=feature[col_m]

# %%
#使用feature_selector 篩選特徵
import os
os.chdir('c:\\Users\\SA\\python\\練習py')
from feature_selector import  FeatureSelector
fs = FeatureSelector(data=feature_matrix,labels=y)
#%%
#處理缺失值
fs.identify_missing (missing_threshold=0.6)
fs.record_missing.head()
fs.plot_missing()
# %%
#處理共線性(colliear)
fs.identify_collinear(correlation_threshold=0.8)
fs.record_collinear.head()
fs.plot_collinear()
# %%
#使用lightGBM演算法
fs.identify_zero_importance(task = 'classification', 
                            eval_metric = 'auc', 
                            n_iterations = 10, 
                             early_stopping = False)
# list of zero importance features
zero_importance_features = fs.ops['zero_importance']

# %%
#尋找低貢獻的Feature
#當重要貢獻度的feautures累積超過0.99後，剩下就是低貢獻features
fs.identify_low_importance(cumulative_importance=0.9)
fs.record_low_importance

# %%
#排序找出貢獻高的因子
fs.feature_importances.sort_values(by='cumulative_importance')

# %%
#method可以客製化你想要先去除的
train_removed=fs.remove(methods='all')

# %%
all_to_remove=fs.check_removal()
all_to_remove


# %%
def featureselect(datas,target):
    import os
    os.chdir('c:\\Users\\SA\\python\\練習py')
    from feature_selector import  FeatureSelector
    fs = FeatureSelector(data=datas,labels=target)

    fs.identify_missing (missing_threshold=0.6)
    fs.identify_collinear(correlation_threshold=0.9)
    fs.identify_zero_importance(task = 'classification', 
                                eval_metric = 'auc', 
                                n_iterations = 10, 
                                early_stopping = False)
    fs.identify_low_importance(cumulative_importance=0.9)

    train_removed=fs.remove(methods='all')
    return train_removed
X2=featureselect(datas=X,target=y)

# %%
#將演算法一一帶入，挑選出表現最佳之演算法，音資料預測目標為多元分類
#使用Cohen’s kappa coefficient來當作模型評量分數

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC  #support vector classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import  RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import f1_score, make_scorer,cohen_kappa_score

X_train,X_test,y_train,y_test=train_test_split(X2,y,test_size=0.3,random_state=42)
score=make_scorer(cohen_kappa_score)

model=[LogisticRegression(),SVC(),DecisionTreeClassifier(),KNeighborsClassifier(),RandomForestClassifier(),GradientBoostingClassifier()]
models=[]
model_mean=[]
model_std=[]
for model_fit in model:
    cross_score=cross_val_score(model_fit,X_train,y_train,scoring=score,n_jobs=-1)
    model_mean.append(np.mean(cross_score))
    model_std.append(np.std(cross_score))
model_results=pd.DataFrame({'model':['LogisticRegression','SVC','DecisionTreeClassifier','KNeighborsClassifier',
                                     'RandomForestClassifier','GradientBoostingClassifier'],
                                     'model_mean':model_mean,
                                     'model_std':model_std})
model_results

# %%
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.metrics import f1_score, make_scorer,cohen_kappa_score
# from sklearn.model_selection import cross_val_score,train_test_split

# X_train,X_test,y_train,y_test=train_test_split(X2,y,test_size=0.3,random_state=42)
# score=make_scorer(cohen_kappa_score)

#挑選GradientBoostingClassifer 當模型，透過GridSearchCV挑選最佳參數
loss=['deviance']   #,'exponential'
#Number of tree used in the boosting process
n_estimators=[100,500]
#每個tree的最大深度
max_depth=[2,3]
#每個leaf最少要有幾個sample
min_sample_leaf=[1,2]
#分割一個截點node最少需要多少sample
min_samples_split=[2,4]
#分割一個節點需要的最大features數量
#Maximum number of features to consider for making splits
max_features=['auto','sqrt','log2',None]
hyperparameter_grid={'loss':loss,
                     'n_estimators':n_estimators,
                     'max_depth':max_depth,
                     'min_samples_leaf':min_sample_leaf,
                     'min_samples_split':min_samples_split,
                     'max_features':max_features}
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
model =GradientBoostingClassifier(random_state=42)
#建立RandomSearchCV n_iter=25這過程要跑25次
random_cv=RandomizedSearchCV(estimator=model,param_distributions=hyperparameter_grid,n_iter=5,scoring=score,cv=5,
                             n_jobs=2,verbose=1,return_train_score=True,random_state=42)
random_cv.fit(X_train,y_train)


# %%
#使用stacking方式進行模型擬合
#第一層使用三種模型，第二層則使用LR模型
from vecstack import stacking
modeler=[DecisionTreeClassifier(),RandomForestClassifier(),GradientBoostingClassifier()]

S_X_train,S_X_test=stacking(modeler,X_train,y_train,X_test,regression=False,
                            metric=metrics.log_loss,needs_proba=True,stratified=True,shuffle=True,
                            random_state=42,verbose=2)


# %%
model = LogisticRegression(penalty='l1',C=1,random_state=42)

model = model.fit(S_X_train,y_train)

y_pred=pd.Series(model.predict(S_X_test))
y_pred_proba = model.predict_proba(S_X_test)[:,1]


print("R Square:",metrics.accuracy_score(y_test, model.predict(S_X_test)))
print("kappa:",metrics.cohen_kappa_score(y_test,model.predict(S_X_test)))




# %%
