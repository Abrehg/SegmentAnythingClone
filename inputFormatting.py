import tensorflow as tf
import matplotlib.pyplot as plt
import optuna
from tensorflow import keras as keras
from keras import layers as tfl
import numpy as np

percentRemove = 75

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
    print(tf.shape(tensor))
    patches = [[]]
    k = 0
    ctr = 0
    while k * 16 < len:
        patches.append(tf.image.crop_to_bounding_box(tensor, k * 16, ctr * 16, 16, 16))
        ctr = ctr + 1
        while ctr * 16 < wid:
            print(f"{k*16},{ctr*16}")
            patches[k].append(tf.image.crop_to_bounding_box(tensor, k * 16, ctr * 16, 16, 16))
            #AttributeError: 'tensorflow.python.framework.ops.EagerTensor' object has no attribute 'append' 
            #doesn't make sense since all values are EagerTensors and it still works till 16,32 after fully completing one row
            ctr = ctr + 1
        ctr = 0
        k = k + 1

    print(f"{patches.len()} x {patches[1].len()}")

    #produce encoding for each part of the image

    #use np.random.rand() function to create a matrix of random values and find all values above a certain percentage (percentRemove variable)

    output = patches
    return output

out = formatImg("/Users/adityaasuratkal/Downloads/ML_Projects/UNet/Data/Indoor Semantic Segmentation/images/vedioDataCollection_July2019_Kent0001.png")
print(out)