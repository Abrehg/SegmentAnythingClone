import tensorflow as tf
import optuna
from tensorflow import keras as keras
from keras import layers as tfl

def decodeFormat(encoding, text):
    output = encoding + text
    return output