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
pickle_dump_folder = "/Users/adityaasuratkal/Downloads/ImageNet"
main(image_data_folder, pickle_dump_folder)

end_time = time.time()
elapsed_time = end_time - start_time

print("Elapsed time: {:.2f} seconds".format(elapsed_time))
"""
"""
#Modify this code to constantly fit the model instead of in processImageNet.py
def compile_data_from_batches(data_folder, dataset_name):
    compiled_data = []
    batch_idx = 0

    while True:
        batch_filename = os.path.join(data_folder, f"{dataset_name}_batch{batch_idx}.pkl")
        if os.path.exists(batch_filename):
            with open(batch_filename, 'rb') as file:
                batch_data = pickle.load(file)
            compiled_data.extend(batch_data)
            batch_idx += 1
        else:
            break

    compiled_filename = os.path.join(data_folder, f"{dataset_name}_compiled.pkl")
    with open(compiled_filename, 'wb') as file:
        pickle.dump(compiled_data, file)
"""
"""
#figure out how to convert pickle to workable data for model
train_dataset = 0
val_dataset = 0
test_dataset = 0

combined_dataset = train_dataset.concatenate(val_dataset).concatenate(test_dataset)

combined_model.fit(combined_dataset)
"""
#Save weights
#custom_model_path = '/Users/adityaasuratkal/Downloads/GitHub/SegmentAnythingClone/encoder_weights.h5'
#combined_model.layers[0:finalLayer].save_weights(custom_model_path)