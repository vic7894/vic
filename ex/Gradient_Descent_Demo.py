#%%
import numpy as np
import matplotlib.pyplot as plt

x_data =[338.,333.,328.,207.,226,25,179,60,208,606]
y_data =[640,633,619,393,428,27,193,66,226,1591]
#y=b+w*x
#%%
b=-120
w=-4
lr=1
step=100000
b_history=[]
w_history=[]

lr_b = 0
lr_w = 0

for i in range(step):

    b_grad=0
    w_grad=0
    for n in range(len(x_data)):
        b_grad += 2*(y_data[n]-(b+w*x_data[n]))*(-1)
        w_grad += 2*(y_data[n]-(b+w*x_data[n]))*(-x_data[n])
        
    lr_b = lr_b + b_grad ** 2
    lr_w = lr_w + w_grad ** 2

    b=b-lr/np.sqrt(lr_b) * b_grad   #Adagrad
    w=w-lr/np.sqrt(lr_w) * w_grad

    b_history.append(b)
    w_history.append(w)

plt.plot(b_history,w_history)
plt.xlim(-200,-100)
plt.ylim(-5,5)
plt.show()




# %%
