import sys
import os
# With this setting we can avoid to see the uncomfortable tensorflow messages
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras import Sequential, optimizers
from keras.layers import InputLayer, Dense
from sklearn.metrics import accuracy_score

# Check if the GPU has been found
print("You are using TensorFlow version", tf.__version__)
gpus_available = len(tf.config.list_physical_devices("GPU"))
if gpus_available != 0:
    print("{0} GPU available\n".format(gpus_available))
else:
    print("None GPUS available")
    sys.exit()  # If there is no GPU available we can not continue. We don't want to run our training process on CPU

# Read the data and preprocess it
df = pd.read_csv("data.csv")
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])  # Label Encoder for the "Gender" column
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [1])],
                       remainder="passthrough")  # One Hot Encoder for the "Geography" columns
x = df.iloc[:, 3:-1].values
x = np.array(ct.fit_transform(x))
y = df.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Scale our data for better performance of the model
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Building the ANN for Classification
model = Sequential()
model.add(InputLayer(input_shape=12, name="Input_Layer"))  # Define the input layer with the 12 features of our dataset. This first layer does not have an activation function
model.add(Dense(24, activation="relu", name="Hidden_Layer_1"))  # Define the first hidden layer with the activation function
model.add(Dense(12, activation="relu", name="Hidden_Layer_2"))  # Define the second hidden layer with the activation function
model.add(Dense(1, activation="sigmoid", name="Output_Layer"))  # Define the output layer with one neuron, our output is binary, so it is enough

# Compiling the ANN
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Training the ANN
model.fit(x_train, y_train, batch_size=10, epochs=50)  # Default value fot batch_size is 32

# Start making predictions. With these values, we expect to get a False result.
prediction = model.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))
print("\n" + "Making our first prediction:")
print("Probability ...", prediction)  # We can get the probability with this statement
print("Prediction ....", prediction > 0.5, "\n")  # We can modify the threshold. Now we get a True or False result

# Evaluating the model
y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5)
print(accuracy_score(y_test, y_pred))

