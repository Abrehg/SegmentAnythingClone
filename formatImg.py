import tensorflow as tf
import matplotlib.pyplot as plt
import optuna
from tensorflow import keras as keras
from keras import layers as tfl
import numpy as np
from patchModel import EncGen
from tf_encodings import add_positional_encoding

percentRemove = 75
Encoder = EncGen((16, 16, 3))
Encoder.load_weights('/Users/adityaasuratkal/Downloads/GitHub/SegmentAnythingClone/patches_weights.h5')

#takes in a tensor representation of your image in order to create an embedding of the image
#returns the original encodings, a vector with 1024 dimension encodings for MAE, and a parallel array with the position (vertical, horizontal) of each patch
def formatImg(img):
    #split into 16x16 patches
    #first create dimensions that are divisible by 16
    height, width, _ = img.shape
    new_height = ((height + 15) // 16) * 16
    new_width = ((width + 15) // 16) * 16
    tensor = tf.image.resize_with_pad(img, new_height, new_width)

    #Then make an array of 16x16 patches
    patches = tf.image.extract_patches(
        images=tf.expand_dims(tensor, axis=0),
        sizes=[1, 16, 16, 1],
        strides=[1, 16, 16, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    patches = tf.reshape(patches, [-1, 16, 16, 3])

    #produce encoding for each part of the image (stored in encodings array)
    encodings = Encoder.predict(patches)

    #Add positional encodings to the image embeddings
    encodings = tf.reshape(encodings, [(new_height // 16), (new_width // 16), 1024])
    embeddings = add_positional_encoding(encodings)
    finalEmbeddings = tf.expand_dims(embeddings, axis=0)
    MAEencodings = finalEmbeddings

    # Create a mask to remove a percentage of encodings
    embeddings = tf.reshape(embeddings, [-1, 1024])
    mask = np.random.rand(((new_height//16)*(new_width//16)),1) > (percentRemove/100)
    maskedEncodings = np.multiply(embeddings, mask)
    MAEencodings = tf.reshape(maskedEncodings, [(new_height // 16), (new_width // 16), 1024])
    MAEencodings = tf.expand_dims(MAEencodings, axis=0)

    return finalEmbeddings, MAEencodings, [new_height, new_width]

#takes filepath as string and turns image into tensor
def formatTensorFromPath(Filepath):
    img = tf.io.read_file(Filepath)
    tensor = tf.io.decode_image(img, channels=3, dtype=tf.dtypes.uint8)
    return tensor