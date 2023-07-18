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

#uncomment this loop to find out what layers to keep
#for i, layer in enumerate(combined_model.layers):
#    print("Layer {}: {}".format(i, layer.name))

start_time = time.time()
#prepare imagenet dataset for training
train_dataset, val_dataset, test_dataset = main()

end_time = time.time()
elapsed_time = end_time - start_time

print("Elapsed time: {:.2f} seconds".format(elapsed_time))

combined_dataset = train_dataset.concatenate(val_dataset).concatenate(test_dataset)

combined_model.fit(combined_dataset)

#Save weights
#custom_model_path = '/Users/adityaasuratkal/Downloads/GitHub/SegmentAnythingClone/encoder_weights.h5'
#combined_model.layers[0:71].save_weights(custom_model_path)