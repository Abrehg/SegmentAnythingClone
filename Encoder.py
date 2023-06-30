import tensorflow as tf
import optuna
from tensorflow import keras as keras
from keras import layers as tfl

encLayers = 15
NNLayers = 3
units = 10

def feedForwardNN(baseInput):
    X = baseInput
    for k in range(0, NNLayers):
        X = tfl.Dense(activation= 'relu', units=units)(X)
    return X

def encoderLayer(input):
    X = tfl.MultiHeadAttention(num_heads= 2, )(input)
    X = tfl.Add()([X, input])
    input2 = tfl.LayerNormalization()(X)
    X = feedForwardNN(X)
    X = tfl.Add()([X, input])
    output = tfl.LayerNormalization()(X)
    return output

def encoder(input):
    X = input
    for i in range(0, encLayers):
        X = encoderLayer(X)
    return X