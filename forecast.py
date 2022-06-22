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

filename = sys.argv[2]
n = int(sys.argv[4])

df=pd.read_csv(filename, sep = '\t', header=None, index_col = 0)
df.head(5)
    
steps = 80
first_row = df.iloc[0, :]
trainers_size = int(len(first_row) * 0.70)
testers_size = len(first_row) - trainers_size

iterations = 20

X_train_2, y_train_2 = [], []


for j in range(0, iterations):
    training_set = df.iloc[j, :trainers_size].values #TRAINING SET 70% OF THE TIME SERIES
    test_set = df.iloc[j, trainers_size:].values  #TEST SET 30% OF THE TIME SERIES

    training_set = training_set.reshape((-1, 1))
    test_set = test_set.reshape((-1, 1))

    # SCALING TRAINING SET
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    # CREATING A DATA STRUCTURE WITH 80 TIME STEPS AND 1 OUTPUT
    X_train = []
    y_train = []
    for i in range(steps, trainers_size):
        X_train.append(training_set_scaled[i-steps:i, 0])
        y_train.append(training_set_scaled[i, 0])
        
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = Sequential()
    #FIRST LSTM LAYER
    model.add(LSTM(units = 100, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    #SECOND LSTM LAYER
    model.add(LSTM(units = 100, return_sequences = True))
    model.add(Dropout(0.2))
    #THIRD LSTM LAYER
    model.add(LSTM(units = 100))
    model.add(Dropout(0.2))
    #OUTPUT LAYER
    model.add(Dense(units = 1))

    #COMPILING
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
    #APPEND ALL THE X_TRAIN PARTS
    for i in range (0, len(X_train)):
        v = X_train[i]
        X_train_2.append(v)
    #APPEND ALL THE Y_TRAIN PARTS
    for i in range (0, len(y_train)):
        v = y_train[i]
        y_train_2.append(v) 
    
    #IF  N=1 THEN FIT AND PREDICT ONE BY ONE TIME SERIES
    if(n == 1):
        model.fit(X_train, y_train, epochs=10, batch_size=32)
        
        #GETTING THE PREDICTED STOCK
        dataset_train = df.iloc[j, :trainers_size]
        dataset_test = df.iloc[j, trainers_size:]

        dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
        inputs = dataset_total[len(dataset_total) - len(dataset_test) - steps:].values
        inputs = inputs.reshape(-1,1)
        inputs = sc.transform(inputs)
        
        X_test = []
        for i in range(steps, testers_size + steps):
            X_test.append(inputs[i-steps:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        #PREDICT X_TEST
        predicted_stock_price = model.predict(X_test)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)

        #GRAPHS
        num_array = np.array(range(0, testers_size))
        num_array = num_array.reshape((-1, 1))
        
        #REAL STOCL
        plt.plot(num_array, dataset_test.values, color = 'red', label = 'Real Stock Price')
        #PREDICT STOCK
        plt.plot(num_array, predicted_stock_price, color = 'blue', label = 'Predicted Stock Price')
        plt.xticks(np.arange(0, testers_size, 100))
        plt.title('Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()
     
#IF N>1 ONE FIT FOR THE TOTAL AND PREDICT FOR N TIME SERIES  
if(n != 1):
    X_train_2 = np.array(X_train_2)
    y_train_2 = np.array(y_train_2)
    
    #WE HAVE ALREADY SAVED THE MODEL --------------------------
    # model.fit(X_train_2, y_train_2, epochs=10, batch_size=32)
    # model.save("model_A_at_home")
    #----------------------------------------------------------

    #RELOADED MODEL
    reloaded_model = keras.models.load_model("model_A_at_home")
    
    
    #PREDICT N TIME SERIES
    for j in range(0, n):
        
        #GETTING PREDICTED STOCKS
        dataset_train = df.iloc[j, :trainers_size]
        dataset_test = df.iloc[j, trainers_size:]

        dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
        inputs = dataset_total[len(dataset_total) - len(dataset_test) - steps:].values
        inputs = inputs.reshape(-1,1)
        inputs = sc.transform(inputs)
        
        X_test = []
        for i in range(steps, testers_size + steps):
            X_test.append(inputs[i-steps:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        #PREDICT X_TEST
        predicted_stock_price = reloaded_model.predict(X_test)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)

        #GRAPHS
        num_array = np.array(range(0, testers_size))
        num_array = num_array.reshape((-1, 1))

        #REAL STOCKS
        plt.plot(num_array, dataset_test.values, color = 'red', label = 'Real Stock Price')
        #PREDICTED STOCKS
        plt.plot(num_array, predicted_stock_price, color = 'blue', label = 'Predicted Stock Price')
        plt.xticks(np.arange(0, testers_size, 100))
        plt.title('Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()