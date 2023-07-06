import tensorflow as tf
import matplotlib.pyplot as plt
import optuna
from tensorflow import keras as keras
from keras import layers as tfl
import numpy as np
from EncoderModel import EncGen


percentRemove = 75
custom_model_path = '/Users/adityaasuratkal/Downloads/GitHub/SegmentAnythingClone/custom_model_weights.h5'
Encoder = EncGen((16, 16, 3))
Encoder.load_weights(custom_model_path)


#takes in a string (filePath) that has a filepath to an image of your choosing in order to create an embedding of the image
def formatImg(filePath):
    #loads in image from filepath
    img = tf.io.read_file(filePath)
    tensor = tf.io.decode_image(img, channels=3, dtype=tf.dtypes.uint8)
    tensor = tf.transpose(tensor, perm=[1, 0, 2])
    
    #split into 16x16 patches
    #first figure out if the dimensions are divisible by 16
    dims = tf.shape(tensor).numpy()
    i = 0
    j = 0
    if dims[0] % 16 != 0:
        while i < 30:
            if (dims[0]+i) % 16 == 0:
                break
            i = i + 1
    if dims[1] % 16 != 0:
        while j < 30:
            if (dims[1]+j) % 16 == 0:
                break
            j = j + 1

    #Then make an array of 16x16 patches
    len = dims[0]+i
    wid = dims[1]+j
    tensor = tf.image.resize_with_pad(tensor, len, wid)
    patches = np.ndarray((1,int(wid/16),16,16,3))
    k = 0
    ctr = 0
    
    while k * 16 < len:
        patchTemp = np.ndarray((1,16,16,3))
        while ctr * 16 < wid:
            patch = tf.image.crop_to_bounding_box(tensor, k * 16, ctr * 16, 16, 16).numpy()
            patch = np.expand_dims(patch, axis = 0)
            patchTemp = np.append(patchTemp,patch, axis = 0)
            ctr = ctr + 1
        patchTemp = np.delete(patchTemp, 1, axis = 0)
        patchTemp = np.expand_dims(patchTemp, axis = 0)
        patches = np.append(patches,patchTemp, axis = 0)
        ctr = 0
        k = k + 1
    patches = np.delete(patches, 1, axis = 0)

    #produce encoding for each part of the image (stored in encodings array)

    dim = np.shape(patches)
    horiz = dim[0]
    vert = dim[1]

    encodings = np.ndarray((horiz, vert,1024))

    for i in range(0, horiz):
        print(i)
        encodings[i][:] = Encoder.predict(patches[i][:][:][:][:])

    #use np.random.rand() function to create a matrix of random values and find all values above a certain percentage (percentRemove variable)

    output = patches
    return output

out = formatImg("/Users/adityaasuratkal/Downloads/ML_Projects/UNet/Data/Indoor Semantic Segmentation/images/vedioDataCollection_July2019_Kent0001.png")
#out = formatImg("/Users/adityaasuratkal/Downloads/ML_Projects/UNet/Data/ADE20K/ADE_frame_00000007/X/ADE_frame_00000007.jpg")