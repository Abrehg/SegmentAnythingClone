import tensorflow as tf
import optuna
from tensorflow import keras as keras
from keras import layers as tfl
from inputFormatting import formatImg

encLayers = 30
NNLayers = 10
units = 512

#imgFilePath = "Test"
imgFilePath = "/Users/adityaasuratkal/Downloads/ML_Projects/UNet/Data/Indoor Semantic Segmentation/images/vedioDataCollection_July2019_Kent0001.png"
#imgFilePath = "/Users/adityaasuratkal/Downloads/ML_Projects/UNet/Data/ADEChallengeData2016/images/training/ADE_train_00000001.jpg"

fullEncodings, MAEencodings, MAEpositions = formatImg(imgFilePath)

def feedForwardNN(baseInput):
    X = baseInput
    for k in range(0, NNLayers):
        X = tfl.Dense(activation= 'relu', units=units)(X)
    return X

def encoderLayer(input):
    X = tfl.MultiHeadAttention(num_heads = 16, key_dim = 25, dropout=0.10)(input, input)
    X = tfl.Add()([X, input])
    input2 = tfl.LayerNormalization(axis = 1)(X)
    X = feedForwardNN(X)
    X = tfl.Add()([X, input])
    output = tfl.LayerNormalization(axis = 1)(X)
    return output

def encoder(input):
    X = input
    for i in range(0, encLayers):
        X = encoderLayer(X)
    return X

out = encoder(MAEencodings)
fullModel = keras.Model(inputs = [MAEencodings, MAEpositions], outputs = out)
print(fullModel.summary())
print(out)