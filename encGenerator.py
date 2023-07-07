import tensorflow as tf
import matplotlib.pyplot as plt
import optuna
from tensorflow import keras
from keras import layers as tfl
import numpy as np
from keras.datasets import cifar100
from EncoderModel import EncGen


(train_images, train_labels), (test_images, test_labels) = cifar100.load_data()
num_classes = 100
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)

input_shape = (32, 32, 3)
input = tf.keras.Input(shape=input_shape)

x = tf.keras.layers.Conv2D(3, (2, 2), activation='relu', strides=(2, 2))(input)
custom_output = EncGen((16, 16, 3))(x)

x = tf.keras.layers.Dense(100, activation='relu')(custom_output)
output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

FullModel = tf.keras.Model(inputs=input, outputs=output)

FullModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

FullModel.fit(train_images, train_labels, batch_size=32, epochs=100, validation_data=(test_images, test_labels))

custom_model_path = '/Users/adityaasuratkal/Downloads/GitHub/SegmentAnythingClone/custom_model_weights.h5'
FullModel.layers[2].save_weights(custom_model_path)