import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Check if the GPU has been found
print("You are using TensorFlow version", tf.__version__)
gpus_available = len(tf.config.list_physical_devices("GPU"))
if gpus_available != 0:
    print("{0} GPU available\n".format(gpus_available))
    # print(tf.test.gpu_device_name())  # /device:GPU:0
else:
    print("None GPUS available")
    exit()  # If there is no GPU available we can not continue. We don't want to run our training process on CPU

# Import the data
dataset_train = pd.read_csv("dataset/Google_Stock_Price_Train.csv")
# [rows, columns] => Index start at 0 and the end should be "value - 1"
training_set = dataset_train.iloc[:, 1:2].values  # Column "Open" => we create the np.array with .values

# Feature Scaling. Normalization vs Standardization
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
# Timesteps specify how many previous observations should be considered
x_train = []
y_train = []
for i in range(60, 1258):
    x_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
# Reshaping
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Initialize the RNN for regression
model = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))  # 20% of 50 Dropout
# Second layer
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
# Third layer
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
# Fourth layer
model.add(LSTM(units=50))
model.add(Dropout(0.2))
# Output layer
model.add(Dense(units=1))

# Compiling the RNN
model.compile(optimizer="adam", loss="mean_squared_error")

# Fitting the RNN to the Training Set
model.fit(x_train, y_train, epochs=100, batch_size=32)

# Getting the real stock price of 2017
dataset_test = pd.read_csv("dataset/Google_Stock_Price_Test.csv")
real_stock_price = dataset_test.iloc[:, 1:2].values
# Getting predicted stock price of 2017
dataset_total = pd.concat((dataset_train["Open"], dataset_test["Open"]), axis=0)  # Horizontal Axis = 1, Vertical Axis = 0
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

x_test = []
for i in range(60, 80):
    x_test.append(inputs[i-60:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_stock_price = model.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(real_stock_price, color="red", label="Real Google Stock Price")
plt.plot(predicted_stock_price, color="blue", label="Predicted Google Stock Price")
plt.title("Google Stock Price Predictions")
plt.xlabel("Time")
plt.ylabel("Google Stock Price")
plt.legend()
plt.show()
