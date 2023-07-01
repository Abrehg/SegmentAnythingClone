import tensorflow as tf
import matplotlib.pyplot as plt
import optuna
from tensorflow import keras as keras
from keras import layers as tfl
import numpy as np

def formatImg(filePath):
    #loads in image from filepath
    img = tf.io.read_file(filePath)
    tensor = tf.io.decode_image(img, channels=3, dtype=tf.dtypes.uint8)
    tensor = tf.transpose(tensor, perm=[1, 0, 2])
    
    #split into 16x16 patches
    #first figure out if the dimensions are divisible by 16
    dims = tf.shape(tensor).numpy()
    if dims[0] % 16 != 0:
        for i in range(0, 30):
            if (dims[0]+i) % 16 != 0:
                break
    if dims[1] % 16 != 0:
        for j in range(0, 30):
            if (dims[1]+j) % 16 != 0:
                break
    output = tensor
    return output

out = formatImg("/Users/adityaasuratkal/Downloads/ML_Projects/UNet/Data/Indoor Semantic Segmentation/images/vedioDataCollection_July2019_Kent0001.png")