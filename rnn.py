# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:14:19 2022
#rnns
@author: anukriti
"""
# Recurrent Neural Network
### needsimprovement


# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
import sklearn


# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

X_train = training_set[0:1257]
y_train = training_set[1:1258]
# Feature Scaling


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler() 
training_set_scaled = sc.fit_transform(training_set)



# Creating a data structure with 60 timesteps and 1 output
#X_train = []
#y_train = []
#for i in range(60, 1258):
 #   X_train.append(training_set_scaled[i-60:i, 0])
  #  y_train.append(training_set_scaled[i, 0])
#X_train, y_train = np.array(X_train), np.array(y_train)


# Reshaping
X_train = np.reshape(X_train,(1257,1, 1))

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = keras.models.Sequential([

#### Adding the first LSTM layer and some Dropout regularisation
keras.layers.LSTM(units = 4, return_sequences = True, input_shape = [None, 1]),
keras.layers.Dropout(0.2),

# Adding a second LSTM layer and some Dropout regularisation
#regressor.add(LSTM(units = 4, return_sequences = True))
#regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
#regressor.add(LSTM(units = 4, return_sequences = True))
#regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
#regressor.add(LSTM(units = 4))
#regressor.add(Dropout(0.2))

#### Adding the output layer
keras.layers.Dense(units = 1)])

## Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

## Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)



# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values



inputs = real_stock_price
inputs = sc.transform(inputs)
inputs = np.reshape(inputs,(20,1,1))

import math
from sklearn.metrics import mean_square_error



# Getting the predicted stock price of 2017
#dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
#inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
#inputs = inputs.reshape(-1,1)
#inputs = sc.transform(inputs)
#X_test = []
#for i in range(60, 80):
  #  X_test.append(inputs[i-60:i, 0])
#X_test = np.array(X_test)
#X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()




