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
"""
Epoch 1/10
1875/1875 - 6s - loss: 0.1840 - accuracy: 0.9437
Epoch 2/10
1875/1875 - 6s - loss: 0.0776 - accuracy: 0.9758
Epoch 3/10
1875/1875 - 6s - loss: 0.0553 - accuracy: 0.9827
Epoch 4/10
1875/1875 - 9s - loss: 0.0405 - accuracy: 0.9865
Epoch 5/10
1875/1875 - 7s - loss: 0.0354 - accuracy: 0.9882
Epoch 6/10
1875/1875 - 7s - loss: 0.0274 - accuracy: 0.9914
Epoch 7/10
1875/1875 - 8s - loss: 0.0233 - accuracy: 0.9926
Epoch 8/10
1875/1875 - 6s - loss: 0.0214 - accuracy: 0.9930
Epoch 9/10
1875/1875 - 6s - loss: 0.0185 - accuracy: 0.9944
Epoch 10/10
1875/1875 - 6s - loss: 0.0162 - accuracy: 0.9948
"""

model.evaluate(x_test, y_test, batch_size=32, verbose=2)
"""
313/313 - 0s - loss: 0.0795 - accuracy: 0.9815
"""

print(model.summary())
"""
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 512)               401920
_________________________________________________________________
dense_1 (Dense)              (None, 256)               131328
_________________________________________________________________
dense_2 (Dense)              (None, 10)                2570
=================================================================
Total params: 535,818
"""
