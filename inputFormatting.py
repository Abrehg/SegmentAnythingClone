import tensorflow as tf
import matplotlib.pyplot as plt
import optuna
from tensorflow import keras as keras
from keras import layers as tfl
import numpy as np
from EncoderModel import EncGen
from tf_encodings import add_positional_encoding

percentRemove = 75
custom_model_path = '/Users/adityaasuratkal/Downloads/GitHub/SegmentAnythingClone/custom_model_weights.h5'
Encoder = EncGen((16, 16, 3))
Encoder.load_weights(custom_model_path)


#takes in a string (filePath) that has a filepath to an image of your choosing in order to create an embedding of the image
#returns the original encodings, a vector with 1024 dimension encodings for MAE, and a parallel array with the position (vertical, horizontal) of each patch
def formatImg(filePath):
    #loads in image from filepath (delete if tensor is already inputted to function)
    img = tf.io.read_file(filePath)
    tensor = tf.io.decode_image(img, channels=3, dtype=tf.dtypes.uint8)
    
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
    height = dims[0]+i
    width = dims[1]+j

    tensor = tf.image.resize_with_pad(tensor, height, width)
    patches = np.ndarray((1,int(width/16),16,16,3))
    k = 0
    ctr = 0
    
    while k * 16 < height:
        patchTemp = np.ndarray((1,16,16,3))
        while ctr * 16 < width:
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
    vert = dim[0]
    horiz = dim[1]

    encodings = np.ndarray((vert, horiz, 1024))

    for i in range(0, vert):
        encodings[i][:] = Encoder.predict(patches[i][:][:][:][:])

    #Add positional encodings to the image embeddings (see tf_encodings.py to find out how it works)
    embeddings = add_positional_encoding(encodings)

    #use np.random.rand() function to create a matrix of random values and find all values above a certain percentage (percentRemove variable)
    mask = np.random.rand(vert,horiz,1)
    mask = mask > (percentRemove/100)
    maskedEncodings = np.multiply(embeddings, mask)

    #Making a vector of unmasked embeddings
    MAEencodings = np.ndarray((1, 1024))

    zeroVector = np.zeros(1024)

    dimen = np.shape(maskedEncodings)
    for i in range(0, dimen[0]):
        for j in range(0, dimen[1]):
            if not (np.array_equal(maskedEncodings[i][j][:], zeroVector)):
                Encoding = np.expand_dims(maskedEncodings[i][j], axis = 0)
                MAEencodings = np.append(MAEencodings, Encoding, axis = 0)

    MAEencodings = np.delete(MAEencodings, 1, axis = 0)
    MAEencodings = tf.expand_dims(MAEencodings, axis=0)
    
    #turning the main embeddings into a (Lx1024) array
    finalEmbeddings = np.ndarray((1, 1024))

    dimen = np.shape(embeddings)
    for i in range(0, dimen[0]):
        for j in range(0, dimen[1]):
            Encoding = np.expand_dims(embeddings[i][j], axis = 0)
            finalEmbeddings = np.append(finalEmbeddings, Encoding, axis = 0)

    finalEmbeddings = np.delete(finalEmbeddings, 1, axis = 0)
    finalEmbeddings = tf.expand_dims(finalEmbeddings, axis=0)

    return finalEmbeddings, MAEencodings