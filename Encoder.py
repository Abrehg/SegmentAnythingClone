import tensorflow as tf
import optuna
from tensorflow import keras as keras
from keras import layers as tfl
from inputFormatting import formatImg

encLayers = 25
NNLayers = 3
units = 10

imgFilePath = "Test"
#imgFilePath = "/Users/adityaasuratkal/Downloads/ML_Projects/UNet/Data/Indoor Semantic Segmentation/images/vedioDataCollection_July2019_Kent0001.png"
#imgFilePath = "/Users/adityaasuratkal/Downloads/ML_Projects/UNet/Data/ADEChallengeData2016/images/training/ADE_train_00000001.jpg"

fullEncodings, MAEencodings, MAEpositions = formatImg(imgFilePath)

def feedForwardNN(baseInput):
    X = baseInput
    for k in range(0, NNLayers):
        X = tfl.Dense(activation= 'relu', units=units)(X)
    return X

def encoderLayer(input):
    X = tfl.MultiHeadAttention(num_heads= 16)(input)
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