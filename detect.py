import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import math
import matplotlib.pyplot as plt
import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

filename = sys.argv[2]
n = int(sys.argv[4])

df = pd.read_csv(filename, sep = '\t', header=None, index_col = 0)

iterations = 20 #NUMBER OF SERIES FOR TRAINING

first_row = df.iloc[0, :]
trainers_size = int(len(first_row) * 0.70)#TRAINING SET 70%
testers_size = len(first_row) - trainers_size#TEST SET 30%

X_train_2, y_train_2 = [], []

for j in range(0, iterations):
    train = df.iloc[j, 0:trainers_size].values

    train = train.reshape((-1, 1))

    training_set = pd.DataFrame()
    training_set['close'] = pd.DataFrame(train)
    training_set['time'] = np.arange(0, len(train))

    #SCALING
    scaler = StandardScaler()
    scaler = scaler.fit(training_set[['close']])
    training_set['close'] = scaler.transform(training_set[['close']])

    #CREATE DATASET FUNCTION
    def create_dataset(X, y, time_steps=1):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            v = X.iloc[i:(i + time_steps)].values
            Xs.append(v)
            ys.append(y.iloc[i + time_steps])
        return np.array(Xs), np.array(ys)

    TIME_STEPS = 80 #TIME_STEPS
    
    X_train, y_train = create_dataset(training_set[['close']], training_set.close, TIME_STEPS)

    model = keras.Sequential()
    #FIRST LAYE
    model.add(keras.layers.LSTM(units=100, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(keras.layers.Dropout(rate=0.2))
    #SECOND LAYER
    model.add(keras.layers.RepeatVector(n=X_train.shape[1]))
    model.add(keras.layers.LSTM(units=100, return_sequences=True))
    model.add(keras.layers.Dropout(rate=0.2))
    #THIRD LAYER
    model.add(keras.layers.LSTM(units=100, return_sequences=True))
    model.add(keras.layers.Dropout(rate=0.2))
    #FOURTH LAYER
    model.add(keras.layers.LSTM(units=100, return_sequences=True))
    model.add(keras.layers.Dropout(rate=0.2))
    #OUTPUT LAYER
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=X_train.shape[2])))
    model.compile(loss='mae', optimizer='adam')
    
    #APPEND ALL THE X_TRAIN PARTS
    for i in range (0, len(X_train)):
        v = X_train[i]
        X_train_2.append(v)
    #APPEND ALL THE Y_TRAIN PARTS
    for i in range (0, len(y_train)):
        v = y_train[i]
        y_train_2.append(v) 
    
X_train_2 = np.array(X_train_2)
y_train_2 = np.array(y_train_2)

#WE HAVE ALREADY SAVED THE MODEL ---------------------------------------------------------------      
# model.fit(X_train_2, y_train_2, epochs=10, batch_size=32, validation_split=0.1, shuffle=False)  
# model.save("model_B_at_home")
#-----------------------------------------------------------------------------------------------
   
reloaded_model = keras.models.load_model("model_B_at_home")

#PREDICT N TIME SERIES
for j in range(0, n):
    test =  df.iloc[j, trainers_size:].values
    test = test.reshape((-1, 1))
    
    test_set = pd.DataFrame()
    test_set['close'] = pd.DataFrame(test)
    test_set['time'] = np.arange(0, len(test))
    
    #SCALING TEST_SET
    scaler = scaler.fit(test_set[['close']])
    test_set['close'] = scaler.transform(test_set[['close']])
    
    X_train_pred = reloaded_model.predict(X_train_2)
    #CALCULATE TRAIN MAE LOSS
    train_mae_loss = np.mean(np.abs(X_train_pred - X_train_2), axis=1)
    
    THRESHOLD = float(sys.argv[6])

    X_test, y_test = create_dataset(test_set[['close']], test_set.close, TIME_STEPS)
    X_test_pred = reloaded_model.predict(X_test)
    
    #CALCULATE TEST MAE LOSS
    test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)
    
    test_score_df = pd.DataFrame(index=test_set[TIME_STEPS:].index)
    test_score_df['loss'] = test_mae_loss
    test_score_df['threshold'] = THRESHOLD
    test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
    test_score_df['close'] = test_set[TIME_STEPS:].close
    test_score_df['time'] = test_set[TIME_STEPS:].time

    #INVERSE TRANSFORM ANOMALIES
    anomalies = test_score_df[test_score_df.anomaly == True]
    scaler.inverse_transform(anomalies['close'].values.reshape(-1,1))
    
    first_col = anomalies.iloc[:, 3]
    plt.plot(test_set['time'], test_set['close'], color = 'red', label = 'Real Stock Price')
    plt.plot(anomalies.time, first_col,'o', markersize = 2, color='blue', label = 'Anomalies');
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()