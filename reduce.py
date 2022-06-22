from base64 import encode
from enum import auto
from importlib import reload
import multiprocessing
import sys
import os
import keras
from sklearn import preprocessing
from tensorflow.python.framework.random_seed import set_seed
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, LSTM, RepeatVector
from keras.models import Model
from keras.models import model_from_json
from keras import regularizers
import datetime
import time
import requests as req
import json
import pandas as pd
import pickle
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
import tensorflow
import numpy

window_length = 10
test_samples = 110

def reproducibleResults(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tensorflow.random.set_seed(seed)
    numpy.random.seed(seed)
    
def plot_examples(stock_input, stock_decoded):
    n = 10  
    plt.figure(figsize=(20, 4))
    for i, idx in enumerate(list(np.arange(0, test_samples, 11))):
        # display original
        ax = plt.subplot(2, n, i + 1)
        if i == 0:
            ax.set_ylabel("Input", fontweight=600)
        else:
            ax.get_yaxis().set_visible(False)
        plt.plot(stock_input[idx])
        ax.get_xaxis().set_visible(False)
        

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        if i == 0:
            ax.set_ylabel("Output", fontweight=600)
        else:
            ax.get_yaxis().set_visible(False)
        plt.plot(stock_decoded[idx])
        ax.get_xaxis().set_visible(False)
    plt.show()
        
       
def plot_history(history):
    plt.figure(figsize=(15, 5))
    ax = plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"])
    plt.title("Train loss")
    ax = plt.subplot(1, 2, 2)
    plt.plot(history.history["val_loss"])
    plt.title("Test loss")
    plt.show()
    
filename = sys.argv[2]

df = pd.read_csv(filename, sep = '\t', header=None, index_col = 0)
id_df = pd.read_csv(filename, sep = '\t', header=None)
id = id_df.iloc[:, 0].values
x_test= []

rows = df.shape[0]
iterations = rows
x_train_total = []
x_test = []
scaled_time_series = []

for j in range(0, iterations): 
    current_row = df.iloc[j, :].values

    x_train = []
        
    df2 = pd.DataFrame()
    df2['price'] = pd.DataFrame(current_row)
    
    #SCALING
    scaler = MinMaxScaler(feature_range = (0, 1))
    x_train_nonscaled = np.array([df2['price'].values[i-window_length:i].reshape(-1, 1) for i in tqdm(range(window_length,len(df2['price']),window_length))])
    x_train = np.array([scaler.fit_transform(df2['price'].values[i-window_length:i].reshape(-1, 1)) for i in tqdm(range(window_length,len(df2['price']), window_length))])
    
    scaled_time_series.append(x_train[:])
    
    x_test.append(x_train[-test_samples:])
    x_train = x_train[:-test_samples]

    x_train = x_train.astype('float32')
    x_test[j] = x_test[j].astype('float32')

    reproducibleResults(123)
    
    #BOTTLENECK 1D CONVOLUTIONAL
    #ENCODER
    input_window = Input(shape=(window_length,1))
    x = Conv1D(32, 3, activation="relu", padding="same")(input_window) # 10 dims
    x = MaxPooling1D(2, padding="same")(x) 
    
    x = Conv1D(32, 3, activation="relu", padding="same")(x)
    x = MaxPooling1D(1, padding="same")(x) 
    
    x = Conv1D(1, 3, activation="relu", padding="same")(x) 
    encoded = MaxPooling1D(2, padding="same")(x)

    encoder = Model(input_window, encoded)

    #DECODER
    x = Conv1D(1, 3, activation="relu", padding="same")(encoded) # 3 dims
    x = UpSampling1D(2)(x) 
    
    x = Conv1D(32, 2, activation='relu')(x) 
    x = UpSampling1D(1)(x) 
    
    x = Conv1D(32, 3, activation='relu', padding='same')(x) 
    x = UpSampling1D(2)(x) # 10 dims
    
    decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(x) # 10 dims

    autoencoder = Model(input_window, decoded)
    # autoencoder.summary()

    encoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    x_train_total.extend(x_train)

x_train_total = np.array(x_train_total)

#WE HAVE ALREADY SAVED THE MODEL FOR THE ENCODER AND THE AUTOENCODER ------------------------------
# history = autoencoder.fit(x_train_total, x_train_total, epochs=500, batch_size=16, shuffle=True)
# autoencoder.save("model_G_at_home")
# encoder.save("model_G_at_home_encoder")
#--------------------------------------------------------------------------------------------------

#RELOADED MODELS
reloaded_model = keras.models.load_model("model_G_at_home")
reloaded_model_encoded = keras.models.load_model("model_G_at_home_encoder")

#PREDICT ALL THE TIME SERIES
for j in range(0, rows): 
    
    #DECODED STOCK PREDICT
    decoded_stocks = reloaded_model.predict(x_test[j])
    #ENCODED STOCK PREDICT
    encoded_stocks = reloaded_model_encoded.predict(scaled_time_series[j])
   
    encoded_stocks = encoded_stocks.reshape(-1,1)
    
    #INVERSE ENCODED STOCK
    encoded_stocks = scaler.inverse_transform(encoded_stocks)
 
    new_csv = pd.DataFrame()
    new_csv[id[j]] = pd.DataFrame(encoded_stocks)
    new_csv = new_csv.T
    output_dataset_file = sys.argv[4]
    output_query_file = sys.argv[6]
    
    if(j < 349):#SAVE 349 ENCODED PREDICT SERIES FOR THE OUTPUT DATASET FILE
        new_csv.to_csv(output_dataset_file, sep = "\t", index = True, mode='a', header=False)
    else:#SAVE 10 ENCODED PREDICT SERIES FOR THE OUTPUT QUERY FILE
        new_csv.to_csv(output_query_file, sep = "\t", index = True, mode='a', header=False)
    
    #REAL AND PREDICT PLOTS
    # plot_examples(x_test[j], decoded_stocks)