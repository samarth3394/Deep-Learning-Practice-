# TensorFlow
import tensorflow as tf

print("TensorFlow Version:", tf.__version__)

a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])

print("TensorFlow Result:", (a + b).numpy())


# Keras
import keras

print("\nKeras Version:", keras.__version__)

model = keras.Sequential([
    keras.layers.Input(shape=(5,)),
    keras.layers.Dense(10, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])

print("\nKeras Model:")
model.summary()


# PyTorch
import torch

print("\nPyTorch Version:", torch.__version__)

x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])

print("PyTorch Result:", x + y)


# Theano
print("\nTheano:")
print("Theano is an older deep learning framework and is deprecated.")