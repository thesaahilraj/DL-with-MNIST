from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape)
# o/p (60000,28,28)

# print(x_test.shape)
# o/p (10000,28,28)

x_train = x_train.reshape(-1, 28*28).astype("float32")/255.0
x_test = x_test.reshape(-1, 28*28).astype("float32")/255.0

# print(x_train.shape)
# o/p (60000,784)
# print(x_test.shape)
# o/p (10000,784)

# Create a basic neural network using Sequential API
model = keras.Sequential()
model.add(keras.Input(shape=(784)))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(10))

# Compiling Model
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

# Fitting the model with Data

model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=2)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)

print(model.summary())
