from dlgpu_assistant.dl_tools import DlAssistant
from keras import models, optimizers, layers
from keras.preprocessing.image import ImageDataGenerator

DlAssistant().find_gpu()
DlAssistant().tf_warnings()

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory("CNN_datasets/dataset/train",
                                                    target_size=(128, 128),
                                                    batch_size=20,
                                                    class_mode="categorical")
validation_generator = test_datagen.flow_from_directory("CNN_datasets/dataset/validation",
                                                        target_size=(128, 128),
                                                        batch_size=20,
                                                        class_mode="categorical")

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(36, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer=optimizers.RMSprop(learning_rate=1e-4), metrics=["acc"])

history = model.fit(train_generator,
                    steps_per_epoch=30,
                    epochs=40,
                    validation_data=validation_generator,
                    validation_steps=10)

model.save('CNN_model.h5')
