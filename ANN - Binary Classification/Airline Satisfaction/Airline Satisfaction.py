from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from dlgpu_assistant.dl_tools import DlAssistant, DlTools, DlLogger

import os

DlAssistant().tf_warnings()
DlAssistant().find_gpu()

# ----------------------------------------------------------------------------------------------------------------------

dataset = "datasets/train.csv"
features = ['Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class', 'Flight Distance', 'Inflight wifi service',
            'Departure/Arrival time convenient', 'Ease of Online booking', 'Gate location', 'Food and drink',
            'Online boarding', 'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service',
            'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness', 'Departure Delay in Minutes',
            'Arrival Delay in Minutes']
label = ["satisfaction"]
label_encoder_bool = True
label_encoder_list = ['Gender', 'Customer Type', 'Type of Travel', 'satisfaction']
one_hot_encoded_bool = True
one_hot_encoded_list = ["Class"]
x, y = DlTools().data_preprocessing(dataset, features, label, label_encoder_bool, label_encoder_list,
                                    one_hot_encoded_bool, one_hot_encoded_list)

# ----------------------------------------------------------------------------------------------------------------------

# Split the data and scale it
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

sc = preprocessing.StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# ----------------------------------------------------------------------------------------------------------------------

# Create the model
model = Sequential()

# ----------------------------------------------------------------------------------------------------------------------

# Hyperparameters for the structure of the model. Also, specific the directory where the models will be saved
models_dir = "Saved Models"
hidden_layers = 2
neurons_hidden_layer = ["24", "12"]
hidden_layer_af = "relu"
output_layer_af = "sigmoid"
# Model Structure
model = DlTools().model_structure(model, x, y, models_dir, hidden_layers, neurons_hidden_layer, hidden_layer_af, output_layer_af)

# ----------------------------------------------------------------------------------------------------------------------

# Hyperparameters for the compile section
optimizer = "adam"
loss = "binary_crossentropy"
metrics = ["accuracy"]
# Compile the model
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# ----------------------------------------------------------------------------------------------------------------------

# Hyperparameters for the training section
batch_size = 32
epochs = 10
verbose = 1
model_name = "Fit_Process_{0}.hdf5".format(len(os.listdir(models_dir))+1)

# Save the best model during the training process
early_stopping = EarlyStopping(monitor="loss", patience=5, verbose=0, mode="min")
mcp_save = ModelCheckpoint(models_dir + "/" + model_name, save_best_only=True, monitor="loss", mode="min")
reduce_lr_loss = ReduceLROnPlateau(monitor="loss", factor=0.1, patience=5, verbose=1, min_delta=1e-4, mode='min')

# Train the model
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose,
                    callbacks=[early_stopping, mcp_save, reduce_lr_loss], validation_split=0.25).history

# ----------------------------------------------------------------------------------------------------------------------

DlLogger().logger_metrics("TrainingResults.log", dataset, history, model_name, hidden_layers, neurons_hidden_layer,
                          hidden_layer_af, output_layer_af, optimizer, loss, metrics, batch_size, epochs, verbose)
