import tensorflow as tf
from tensorflow import keras
from keras import layers as tfl
import numpy as np
from FullEncoder import encoder
from processImageNet import main
import time

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

"""
start_time = time.time()
#Prepare Imagenet dataset for training
image_data_folder = "/Users/adityaasuratkal/Downloads/ImageNet"
pickle_dump_folder = "/Users/adityaasuratkal/Downloads/ImageNet/pickles"
main(image_data_folder, pickle_dump_folder)

end_time = time.time()
elapsed_time = end_time - start_time

print("Elapsed time: {:.2f} seconds".format(elapsed_time))
"""
"""
batch_idx = 0
while True:
    batch_filename = os.path.join(data_folder, f"{'train'}_batch{batch_idx}.pkl")
    if os.path.exists(batch_filename):
        with open(filename, 'rb') as file:
            data = pickle.load(file)
            embeddings, labels = zip(*data)
            embeddings = tf.convert_to_tensor(embeddings, dtype=tf.float32)
            labels = tf.convert_to_tensor(labels, dtype=tf.float32)
            dataset = tf.data.Dataset.from_tensor_slices((embeddings, labels))
            dataset = dataset.shuffle(buffer_size=len(data)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            combined_model.fit(dataset)
        batch_idx = batch_idx + 1
    else:
        break
        
batch_idx = 0
while True:
    batch_filename = os.path.join(data_folder, f"{'val'}_batch{batch_idx}.pkl")
    if os.path.exists(batch_filename):
        with open(filename, 'rb') as file:
            data = pickle.load(file)
            embeddings, labels = zip(*data)
            embeddings = tf.convert_to_tensor(embeddings, dtype=tf.float32)
            labels = tf.convert_to_tensor(labels, dtype=tf.float32)
            dataset = tf.data.Dataset.from_tensor_slices((embeddings, labels))
            dataset = dataset.shuffle(buffer_size=len(data)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
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
            dataset = dataset.shuffle(buffer_size=len(data)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            combined_model.fit(dataset)
        batch_idx = batch_idx + 1
    else:
        break
"""
#Save weights
#custom_model_path = '/Users/adityaasuratkal/Downloads/GitHub/SegmentAnythingClone/encoder_weights.h5'
#combined_model.layers[0:finalLayer].save_weights(custom_model_path)