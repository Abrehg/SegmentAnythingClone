import tensorflow as tf
from tensorflow import keras
from keras import layers as tfl
import numpy as np
from FullEncoder import encoder
from processImageNet import main
import time
import h5py
import os
import multiprocessing

NNLayers = 4
unitsMid = 100
unitsOut = 1000

# Create the encoder model
encoder_model = encoder()

# Obtain the encoder output
encoder_output = encoder_model.output

#decoder architecture
x = tfl.GlobalAveragePooling1D()(encoder_output)
for i in range(NNLayers):
    x = tfl.Dense(unitsMid, activation='relu')(x)
output = tfl.Dense(unitsOut, activation='softmax')(x)

# Create the combined model
combined_model = keras.Model(inputs=encoder_model.input, outputs=output)

# Compile the model
combined_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
#combined_model.summary()

#Find Final Layer Number of Encoder
finalLayer = 0
for i, layer in enumerate(combined_model.layers):
#    print("Layer {}: {}".format(i, layer.name))
    if layer.name == "global_average_pooling1d":
        finalLayer = i-1
        break

#Prepare Imagenet dataset for training
image_data_folder = "/Users/adityaasuratkal/Downloads/ImageNet"
pickle_dump_folder = "/Users/adityaasuratkal/Downloads/ImageNet/compression"

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main(image_data_folder, pickle_dump_folder)

"""
batch_idx = 0
while True:
    filename = os.path.join(pickle_dump_folder, f"{'train'}_batch{batch_idx}.h5")
    if os.path.exists(filename):
        with h5py.File(filename, "r") as file:
            embeddings = tf.convert_to_tensor(file["embeddings"], dtype=tf.float32)
            labels = tf.convert_to_tensor(file["labels"], dtype=tf.float32)

        dataset = tf.data.Dataset.from_tensor_slices((embeddings, labels))
        dataset = dataset.shuffle(buffer_size=len(embeddings)).batch(1024).prefetch(tf.data.AUTOTUNE)
        combined_model.fit(dataset)

        batch_idx = batch_idx + 1
    else:
        break

batch_idx = 0    
while True:
    filename = os.path.join(pickle_dump_folder, f"{'val'}_batch{batch_idx}.h5")
    if os.path.exists(filename):
        with h5py.File(filename, "r") as file:
            embeddings = tf.convert_to_tensor(file["embeddings"], dtype=tf.float32)
            labels = tf.convert_to_tensor(file["labels"], dtype=tf.float32)

        dataset = tf.data.Dataset.from_tensor_slices((embeddings, labels))
        dataset = dataset.shuffle(buffer_size=len(embeddings)).batch(1024).prefetch(tf.data.AUTOTUNE)
        combined_model.fit(dataset)

        batch_idx = batch_idx + 1
    else:
        break
"""
"""    
batch_idx = 0
while True:
    batch_filename = os.path.join(data_folder, f"{'test'}_batch{batch_idx}.pkl")
    if os.path.exists(batch_filename):
        with open(filename, 'rb') as file:
            data = pickle.load(file)
            embeddings, labels = zip(*data)
            embeddings = tf.convert_to_tensor(embeddings, dtype=tf.float32)
            labels = tf.convert_to_tensor(labels, dtype=tf.float32)
            dataset = tf.data.Dataset.from_tensor_slices((embeddings, labels))
            dataset = dataset.shuffle(buffer_size=len(data)).batch(1024).prefetch(tf.data.AUTOTUNE)
            combined_model.fit(dataset)
        batch_idx = batch_idx + 1
    else:
        break
"""
#Save weights
"""
encoder_input = combined_model.input
encoder_output = combined_model.layers[0].output
i = 1
for layer in combined_model.layers[1:finalLayer]:
    print(f"layer {i}")
    encoder_output = layer(encoder_output)
    i = i + 1

encoder_weights_model = keras.Model(inputs=encoder_input, outputs=encoder_output)
custom_model_path = '/Users/adityaasuratkal/Downloads/GitHub/SegmentAnythingClone/encoder_weights.h5'
encoder_weights_model.layers.save_weights(custom_model_path)
"""