import tensorflow as tf
import optuna
from tensorflow import keras as keras
from keras import layers as tfl

decodeLayers = 7

def decodeLayer(input):
    output = input
    return output

def decoder(input):
    X = input
    for i in range(0, decodeLayers):
        X = decodeLayers(X)
    return X