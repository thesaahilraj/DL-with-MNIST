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

# Create a basic neural network using Functional API
inputs = keras.Input(shape=(784))
x = layers.Dense(512, activation='relu', name='First_layer')(inputs)
x = layers.Dense(256, activation='relu', name='Second_layer')(x)
output = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=output)

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
1875/1875 - 6s - loss: 1.5733 - accuracy: 0.8906
Epoch 2/10
1875/1875 - 6s - loss: 1.5057 - accuracy: 0.9560
Epoch 3/10
1875/1875 - 6s - loss: 1.4987 - accuracy: 0.9625
Epoch 4/10
1875/1875 - 6s - loss: 1.4935 - accuracy: 0.9677
Epoch 5/10
1875/1875 - 6s - loss: 1.4916 - accuracy: 0.9696
Epoch 6/10
1875/1875 - 6s - loss: 1.4899 - accuracy: 0.9711
Epoch 7/10
1875/1875 - 6s - loss: 1.4884 - accuracy: 0.9727
Epoch 8/10
1875/1875 - 6s - loss: 1.4909 - accuracy: 0.9702
Epoch 9/10
1875/1875 - 6s - loss: 1.4895 - accuracy: 0.9716
Epoch 10/10
1875/1875 - 6s - loss: 1.4886 - accuracy: 0.9725
"""

model.evaluate(x_test, y_test, batch_size=32, verbose=2)
"""
313/313 - 0s - loss: 1.4942 - accuracy: 0.9667
"""

print(model.summary())
"""
Model: "functional_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 784)]             0
_________________________________________________________________
First_layer (Dense)          (None, 512)               401920
_________________________________________________________________
Second_layer (Dense)         (None, 256)               131328
_________________________________________________________________
dense (Dense)                (None, 10)                2570
=================================================================
Total params: 535,818
Trainable params: 535,818
Non-trainable params: 0
_________________________________________________________________
"""
