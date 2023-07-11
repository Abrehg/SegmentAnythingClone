import tensorflow as tf
import optuna
from tensorflow import keras as keras
from keras import layers as tfl
from inputFormatting import formatImg
from FullEncoder import encoder

#imgFilePath = "Test"
imgFilePath = "/Users/adityaasuratkal/Downloads/ML_Projects/UNet/Data/Indoor Semantic Segmentation/images/vedioDataCollection_July2019_Kent0001.png"
#imgFilePath = "/Users/adityaasuratkal/Downloads/ML_Projects/UNet/Data/ADEChallengeData2016/images/training/ADE_train_00000001.jpg"

fullEncodings, MAEencodings = formatImg(imgFilePath)
enc = encoder(MAEencodings)

decodeLayers = 7

def decodeLayer(input):
    output = input
    return output

def decoder(input):
    X = input
    for i in range(0, decodeLayers):
        X = decodeLayers(X)
    return X