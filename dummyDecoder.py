import tensorflow as tf
from tensorflow import keras
from keras import layers as tfl
import numpy as np
from inputFormatting import formatImg
from inputFormatting import formatTensorFromPath
from FullEncoder import encoder

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
combined_model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Print the model summary
combined_model.summary()

#Preprocessing
#imgFilePath = "/Users/adityaasuratkal/Downloads/ML_Projects/UNet/Data/Indoor Semantic Segmentation/images/vedioDataCollection_July2019_Kent0001.png"
#tensor = formatTensorFromPath(imgFilePath)
#fullEncodings, MAEencodings = formatImg(tensor)

#Save weights
#custom_model_path = '/Users/adityaasuratkal/Downloads/GitHub/SegmentAnythingClone/encoder_weights.h5'
#combined_model.layers[0:116].save_weights(custom_model_path)