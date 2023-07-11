import tensorflow as tf
from tensorflow import keras as keras
from keras import layers as tfl
import numpy as np
from inputFormatting import formatImg

encLayers = 30
NNLayers = 10
units = 1024

#imgFilePath = "Test"
imgFilePath = "/Users/adityaasuratkal/Downloads/ML_Projects/UNet/Data/Indoor Semantic Segmentation/images/vedioDataCollection_July2019_Kent0001.png"
#imgFilePath = "/Users/adityaasuratkal/Downloads/ML_Projects/UNet/Data/ADEChallengeData2016/images/training/ADE_train_00000001.jpg"

fullEncodings, MAEencodings = formatImg(imgFilePath)

input_data = tf.expand_dims(fullEncodings, axis=0)

def feedForwardNN(baseInput):
    X = baseInput
    for k in range(0, NNLayers):
        X = tfl.Dense(activation= 'relu', units=units)(X)
    return X

def encoderLayer(input):
    X = tfl.MultiHeadAttention(num_heads = 16, key_dim = 1024, dropout=0.3)(input, input)
    X = tf.cast(X, dtype=tf.float32)
    input = tf.cast(input, dtype=tf.float32)
    input2 = np.add(X.numpy(), input.numpy())
    input2 = tfl.LayerNormalization(axis = 1)(input2)
    X = feedForwardNN(input2)
    X = tf.cast(X, dtype=tf.float32)
    input2 = tf.cast(input2, dtype=tf.float32)
    X = np.add(X.numpy(), input2.numpy())
    output = tfl.LayerNormalization(axis = 1)(X)
    return output

def encoder(input):
    X = tf.cast(input, dtype=tf.float32)
    for i in range(0, encLayers):
        print(f"Layer {i}")
        X = encoderLayer(X)
    return X

out = encoder(input_data)
fullModel = keras.Model(inputs = input_data, outputs = out)
print(fullModel.summary())
print(out)