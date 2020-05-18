#%%
#EX1
from sklearn import svm
from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline

X, y =samples_generator.make_classification(
    n_features=20,n_informative=3,n_redundant=0,n_classes=4,n_clusters_per_class=2
)
#我們將X建立為一個有20個特徵的資料，其中有3種特徵具有目標資訊性，0個特徵是由目標資訊性特徵所產生的線性組合，
# 目標分為4類，而每個分類的目標分布為2個群集。

#使用ANOVA特徵選擇
anova_filter=SelectKBest(f_regression,k=3)
anova_filter.fit_transform(X,y)


clf=svm.SVC(kernel='linear')
anova_svm=make_pipeline(anova_filter,clf)
anova_svm.fit(X,y)
anova_svm.predict(X)

# %%
#EX2 使用REF跌代方式計算模型
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt



digits=load_digits()
X = digits.images.reshape(len(digits.images),-1)
y = digits.target


svc= SVC(kernel='linear',C=1)
rfe = RFE(estimator=svc,n_features_to_select=1,step=1)
rfe.fit(X,y)
ranking=rfe.ranking_.reshape(digits.images[0].shape)


plt.matshow(ranking,cmap=plt.cm.Blues)
plt.colorbar()
plt.title('ranking of pixels with REF')
plt.show()

#%%
#EX3
# 1.以疊代方式計算模型
# 2.以交叉驗證來取得影響力特徵
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification

X,y=make_classification(n_samples=1000,n_features=25,n_informative=3,
                        n_redundant=2,n_repeated=0,n_classes=8,
                        n_clusters_per_class=1,random_state=0)


svm=SVC(kernel='linear')
rfecv=RFECV(estimator=svm,step=1,scoring='accuracy')

rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)

plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

# %%
#EX4
# 使用LASSOCV來篩選特徵，必須皆為連續變數，類似最小平方法
from sklearn.datasets import load_boston
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

boston = load_boston()
X,y=boston['data'],boston['target']

clf=LassoCV()

sfm=SelectFromModel(clf, threshold=0.25)
sfm.fit(X,y)
n_features=sfm.transform(X).shape[1]

while n_features > 2:
    sfm.threshold +=0.1
    X_transform =sfm.transform(X)
    #print(X_transform)
    n_features =X_transform.shape[1]
    #print(n_features)

import numpy as np
plt.title(
    "Features selected from Boston using SelectFromModel with "
    "threshold %0.3f." % sfm.threshold)
feature1 = X_transform[:, 0]
feature2 = X_transform[:, 1]
plt.plot(feature1, feature2, 'r.')
plt.xlabel("Feature number 1")
plt.ylabel("Feature number 2")
plt.ylim([np.min(feature2), np.max(feature2)])
plt.show()

#%%
#EX 5 卡方檢定
# 料標籤為無大小關係的分類，也就是第一類與第二類並無前後大小關係的分類。
# 由於輸入分類器的標籤仍為數值，但數值的大小可能影響分類結果
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, permutation_test_score
from sklearn import datasets

iris =datasets.load_iris()
X = iris.data
y = iris.target
n_classes=np.unique(y).size

#make noisy data 
random=np.random.RandomState(seed=0)
E = random.normal(size=(len(X),2200))
X=np.c_[X,E]

svm=SVC(kernel='linear')
cv=StratifiedKFold(n_splits=2)

score, permutation_scores,pvalue=permutation_test_score(
    svm,X,y,scoring="accuracy",cv=cv,n_permutations=100,n_jobs=2
)
print("Classification score %s (pvalue : %s)" % (score, pvalue))

# View histogram of permutation scores
plt.hist(permutation_scores, 20, label='Permutation scores')
ylim = plt.ylim()

# plt.plot(2 * [score], ylim, '--g', linewidth=3,
#          label='Classification Score'
#          ' (pvalue %s)' % pvalue)
plt.plot(2 * [1. / n_classes], ylim, '--k', linewidth=3, label='Luck')

plt.ylim(ylim)
plt.legend()
plt.xlabel('Score')
plt.show()

# %%
#Ex 6: Univariate Feature Selection
# 選擇過程會畫出每個特徵的 p-value 與其在支持向量機中的權重。
# 可以從圖表中看出主要影響力特徵的選擇會選出具有主要影響力的特徵，
# 並且這些特徵會在支持向量機有相當大的權重。
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets, svm
from sklearn.feature_selection import SelectPercentile, f_classif


iris=datasets.load_iris()
X=iris.data
y=iris.target

E=np.random.uniform(0,0.1,size=(len(X),20))
X=np.hstack((X,E))

selector=SelectPercentile(f_classif,percentile=10)
selector.fit(X,y)
scores = -np.log10(selector.pvalues_)
scores /= scores.max()


plt.figure(1)
plt.clf()
X_indices = np.arange(X.shape[-1])
plt.bar(X_indices - .45, scores, width=.2,
        label=r'Univariate score ($-Log(p_{value})$)', color='g')

clf=svm.SVC(kernel='linear')
clf.fit(X,y)
svm_weights = (clf.coef_ ** 2).sum(axis=0)
svm_weights /= svm_weights.max()

plt.bar(X_indices - .25, svm_weights, width=.2, label='SVM weight', color='r')


clf_selected = svm.SVC(kernel='linear')
clf_selected.fit(selector.transform(X), y)

svm_weights_selected = (clf_selected.coef_ ** 2).sum(axis=0)
svm_weights_selected /= svm_weights_selected.max()

plt.bar(X_indices[selector.get_support()] - .05, svm_weights_selected,
        width=.2, label='SVM weights after selection', color='b')

# %%
#EX7
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_regression, mutual_info_regression

np.random.seed(0)
X = np.random.rand(1000, 3)
y = X[:, 0] + np.sin(6 * np.pi * X[:, 1]) + 0.1 * np.random.randn(1000)

f_test, _ = f_regression(X, y)
f_test /= np.max(f_test)

mi = mutual_info_regression(X, y)
mi /= np.max(mi)

plt.figure(figsize=(15, 5))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.scatter(X[:, i], y)
    plt.xlabel("$x_{}$".format(i + 1), fontsize=14)
    if i == 0:
        plt.ylabel("$y$", fontsize=14)
    plt.title("F-test={:.2f}, MI={:.2f}".format(f_test[i], mi[i]),
              fontsize=16)
plt.show()

# %%
