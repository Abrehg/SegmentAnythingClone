import tensorflow as tf
import optuna
from tensorflow import keras as keras
from keras import layers as tfl
from inputFormatting import formatImg
from Encoder import encoder

def model(image):
    X = formatImg(image)
    X = encoder(X)