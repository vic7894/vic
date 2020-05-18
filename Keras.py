#%%
from keras.layers import Dense, BatchNormalization,Convolution2D,MaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils
from keras.datasets import mnist

def load_data():
    (x_train,y_train),(x_test,y_test)=mnist.load_data()

    number=10000
    x_train=x_train[0:number]
    y_train=y_train[0:number]
    x_train=x_train.reshape(number,28*28)
    x_test=x_test.reshape(x_test.shape[0],28*28)
    x_train=x_train.astype('float32')
    x_test=x_test.astype('float32')
    #label
    y_train=np_utils.to_categorical(y_train,10)
    y_test=np_utils.to_categorical(y_test,10)
    x_train=x_train
    x_test=x_test

    x_train=x_train/255
    x_test=x_test/255
    return(x_train,y_train),(x_test,y_test)
#%%
(x_train,y_train),(x_test,y_test)=load_data()

model =Sequential()

model.add(Dense(input_dim=28*28,units=500,
                activation='relu'))
model.add(Dense(units=64,activation='relu'))
model.add(Dense(units=16,activation='relu'))
model.add(Dense(units=10, activation='softmax'))
#configuration
model.compile(loss='categorical_crossentropy', 
                optimizer='adam',
                metrics=['accuracy'])
#pick the best function
model.fit(x_train,y_train,batch_size=100,nb_epoch=20)

result1=model.evaluate(x_train,y_train)
print('\nTrain Acc',result1[1])

result=model.evaluate(x_test,y_test)
print('\nTest Acc',result[1])

# %%
#CNN
model2=[]
model2.add(Convolution2D(25,3,3,inout_shqpe=(1,28,28)))
model2.add(MaxPooling2D((2,2)))
model2.add(Convolution2D(50,3,3))
model2.add(MaxPooling2D((2,2)))


# %%
