import tensorflow as tf
import matplotlib.pyplot as plt
import optuna
from tensorflow import keras as keras
from keras import layers as tfl
import numpy as np

Inputs = keras.Input(shape=(16,16,3))
X = tfl.Conv2D(4, (2,2))(Inputs)
X = tfl.Flatten()(X)
X = tfl.Dense(512, 'relu')(X)
Y = tfl.Dense(1024, 'relu')(X)
EncGenerator = keras.Model(inputs = Inputs, outputs = Y)

print(EncGenerator.summary())