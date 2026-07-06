import numpy as np
import pandas as pd  # 1.4.3
import tensorflow as tf  # 2.9.1
import sys
import os
# With this setting we can avoid to see the uncomfortable tensorflow messages
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

# Check if the GPU has been found
print("You are using TensorFlow version", tf.__version__)
gpus_available = len(tf.config.list_physical_devices("GPU"))
if gpus_available != 0:
    print("{0} GPU available\n".format(gpus_available))
else:
    print("None GPUS available")
    sys.exit()  # If there is no GPU available we can not continue. We don't want to run our training process on CPU

# Create de ANN for Regression
dataset = pd.read_excel("dataset.xlsx")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

x_test, x_train, y_test, y_train = train_test_split(x, y, test_size=0.2, random_state=0)

model = Sequential()
model.add(Dense(units=6, activation="relu"))
model.add(Dense(units=6, activation="relu"))
model.add(Dense(units=1, activation=None))

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(x_train, y_train, batch_size=32, epochs=100)

y_pred = model.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
