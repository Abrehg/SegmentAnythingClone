import tensorflow as tf
from tensorflow import keras
from keras import layers as tfl
import numpy as np
from inputFormatting import formatImg
from inputFormatting import formatTensorFromPath
from FullEncoder import encoder

NNLayers = 7
unitsMid = 10000
unitsOut = 1000

def dummyDecoder():
    model = keras.models.Sequential([
        tfl.Input(shape=(None, 1024)),  
        tfl.GlobalAveragePooling1D()
    ])
    for i in range(0, NNLayers):
        model.add(tfl.Dense(unitsMid, activation='relu'))
    model.add(tfl.Dense(unitsOut, activation='softmax'))
    
    return model


imgFilePath = "/Users/adityaasuratkal/Downloads/ML_Projects/UNet/Data/Indoor Semantic Segmentation/images/vedioDataCollection_July2019_Kent0001.png"
#imgFilePath = "/Users/adityaasuratkal/Downloads/ML_Projects/UNet/Data/ADEChallengeData2016/images/training/ADE_train_00000001.jpg"

fullEncodings, MAEencodings = formatImg(formatTensorFromPath(imgFilePath))
X = encoder(fullEncodings)
decoderOut = dummyDecoder()(X)
print(tf.shape(X))
print(tf.shape(decoderOut))