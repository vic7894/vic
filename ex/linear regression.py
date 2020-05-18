#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#y=ax+b
rng=np.random.RandomState(1)
x=10*rng.rand(50)
y=2*x-5+rng.randn(50)
plt.scatter(x,y)
plt.show()
# %%
#最小平方法
def lr(x,y):
  x=np.concatenate((np.ones((x.shape[0],1)),x[:,np.newaxis]),axis=1)
  y=y[:,np.newaxis]
  beta=np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T,x)),x.T),y)
  return beta
b=lr(x,y)

# %%
x2=np.linspace(0,10,200)
y2=b[0]+b[1]*x2

plt.scatter(x,y,s=100)
plt.plot(x2,y2,'r')
plt.show()

#%%
learning_rate=0.0001
#經過3000次的調整參數
n_iterations=3000

def gd(x,y):
    #隨機定義參數值
    theta = np.random.randn(2,1)
    x=np.concatenate((np.ones((x.shape[0],1)),x[:,np.newaxis]),axis=1)
    y=y[:,np.newaxis]
    for iteration in range(n_iterations):
        scores = np.dot(x,theta)
        #誤差值
        output_error = y-scores
        #x的shape(50,2) output_error的shapeｊ為(50,1) gradients為(2,1)
        gradients = 2*np.dot(x.T,output_error)
        # if iteration%100==1:
        #     print(theta)
        #每次對theta
        theta += learning_rate*gradients
        
        plt.plot(theta[0],theta[1],'o-',color='red')
    print('theta:',theta)
    plt.xlabel('intercept',color='black')
    plt.ylabel('weight',color='black')
    plt.xticks(color='black')
    plt.yticks(color='black')
    plt.show()
gd(x,y)

# %%
#Logistic Regression
import numpy as np
import matplotlib.pyplot as plt

def data() :
    np.random.seed(12)
    number_observation=5000

    x1=np.random.multivariate_normal([0,0],[[1,0.75],[0.75,1]],number_observation)
    x2=np.random.multivariate_normal([1,4],[[1,0.75],[0.75,1]],number_observation)

    X=np.vstack((x1,x2)).astype(np.float32)

    #為了設置color的label
    Y=np.hstack((np.zeros(number_observation),np.ones(number_observation)))
    return X,Y

X,Y=data()

plt.figure(figsize=(10,6))
plt.scatter(X[:,0],X[:,1],c=Y)

# %%
def sigmoid(function):
    return 1/(1+np.exp(-function))

def log_likelihood(features,target,weights):
    scores=np.dot(features,weights)
    l1=np.sum(target*scores-np.log(1+np.exp(scores)))
    return l1


def logistic_regression(features,target,num_steps,learning_rate,add_intercept=False):
    make_plot=[]
    #代表如果是否需要截距項 True的話就會建立之
    if add_intercept:
        intercept=np.ones((features.shape[0],1))
        features=np.hstack((intercept,features))
    
    weights=np.zeros(features.shape[1])
    #開始進行迴圈跑最出最佳參數
    for step in range(num_steps):
        scores=np.dot(features,weights)
        #function套入sigmoid function得1機率值
        predictions=sigmoid(scores)
        #觀看誤差
        output_error_singal=target-predictions
        #gradient
        gradient=np.dot(features.T,output_error_singal)
        weights+=learning_rate*gradient
        
        #print出藉由參數不斷的調整 Loss function不斷在向最小化邁進
        if step%10000==0:
            make_plot.append(log_likelihood(features,target,weights))
    return weights,make_plot

# %%
weights,make_plot=logistic_regression(X,Y,num_steps=300000,learning_rate=5e-5,add_intercept=True)

# %%
plt.plot(make_plot)

# %%
