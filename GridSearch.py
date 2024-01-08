import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.layers import Normalization
import matplotlib.pyplot as plt
import os
from tensorboard.plugins.hparams import api as hp

#input csv
df=pd.read_csv("delhi_preprocessed.csv")

#preprocessing
df= df.groupby('date_time', as_index=False, sort=False)['date', 'month','year', 'maxtempC', 'mintempC', 'totalSnow_cm',
       'sunHour', 'uvIndex', 'uvIndex.1', 'moon_illumination', 'DewPointC',
       'FeelsLikeC', 'HeatIndexC', 'WindChillC', 'WindGustKmph', 'cloudcover',
       'humidity', 'precipMM', 'pressure', 'tempC', 'visibility',
       'winddirDegree', 'windspeedKmph'].mean()
df2 = df.drop(['date_time'], axis=1)
df2=df2.dropna().reset_index(drop=True)
df2['precipMM'] = df2['precipMM'].astype(bool)  
pd.set_option('display.max_columns', None)
df2['precipMM'] = df2['precipMM'].astype(int) 

#sampling
dftest=[]
for i in range(1,4018-7): #3662-7
    temp = df2.T.iloc[17,i:i+7].to_numpy() #change the 7 to 2
    tempd = pd.DataFrame(temp.T,columns=['precipMM'])
    tempn = tempd.T 
    dftest.append(tempn)
    i=i+1
dfinal=pd.concat(dftest,ignore_index=True)

#train and test
df_X_train = df2.iloc[0:3660,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]] 
df_X_test = df2.iloc[3660:4010,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]]
df_y_train = dfinal.iloc[0:3660]
df_y_test = dfinal.iloc[3660:4010]

X_train = df_X_train.to_numpy()
X_test = df_X_test.to_numpy()
y_train = df_y_train.to_numpy()
y_test = df_y_test.to_numpy()


#model
size = X_train.shape[1]

dropout1 = [0.13, 0.14, 0.15]
dropout2 = [0.13, 0.14, 0.15]
dense1 = [22, 23, 24]
dense2 = [8, 9, 10]

tf.random.set_seed(1234)
max=0
mins=[0,0,0,0]

for x in dropout1:
    for y in dropout2:
        for z in dense1:
            for l in dense2:
                model = Sequential(
                    [    
                        Normalization(axis=-1,), 
                        tf.keras.Input(shape=(size,)),       
                        Dense(z, activation = 'relu',kernel_constraint=MaxNorm(3)),
                        Dropout(x, input_shape=(size,)),  
                        Dense(l, activation = 'relu',kernel_constraint=MaxNorm(3)),
                        Dropout(y , input_shape=(size,)),
                        # Dense(5, activation = 'relu',kernel_constraint=MaxNorm(3)),
                        # Dropout(0.1 , input_shape=(size,)),
                        Dense(7, activation = 'sigmoid')
                    ], name = "my_model" 
                )
                model.compile(
                    loss=tf.keras.losses.BinaryCrossentropy(),
                    optimizer=tf.keras.optimizers.Adam(0.001),
                )
                history= model.fit(
                    X_train,y_train,
                    epochs=500,
                    batch_size=64,
                )

                #loss fn graph
                # train_loss = history.history['loss']
                # epochs = range(1, len(train_loss) + 1)

                # plt.figure(figsize=(10, 6))
                # plt.plot(epochs, train_loss, 'b', label='Training Loss')
                # plt.title('Loss vs. Epoch')
                # plt.xlabel('Epochs')
                # plt.ylabel('Loss')
                # plt.legend()
                # plt.yscale("log")
                # plt.grid(True, which='both')
                # plt.show()

                #accuracy 
                i=0
                yhat=0
                prediction = model.predict(X_test[0].reshape(1,size))
                prediction[0]
                count = 0
                for i in range(0,len(y_test)):
                    prediction = model.predict(X_test[i].reshape(1,size))
                    for j in range(0,7):
                        if prediction[0][j] >= 0.5:
                            yhat = 1
                        else:
                            yhat = 0
                        if(yhat==y_test[i][j]):
                            count = count + 1

                accuracy = (count/(7*i))*100
                if(accuracy>max):
                    max=accuracy
                    mins=[x,y,z,l]

print("The best accuracy is ", max, " with values", mins[0], mins[1], mins[2], mins[3])
# The best accuracy is  73.02496930004094  with values 0.13 0.13 23 10


