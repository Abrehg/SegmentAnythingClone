import tensorflow as tf
from tensorflow import keras as keras
from keras import layers as tfl
from formatText import formatText
from CLIPencoder import textEncoder

#textInput = 'Hello World'

#embeddings = formatText(textInput)

NNLayers = 4
unitsMid = 1000
unitsOut = 1

inputEmbeddings = keras.Input((None, 300))
textEnc = textEncoder()
x = textEnc(inputEmbeddings)

#x = tfl.GlobalAveragePooling1D()(x)
for i in range(NNLayers):
    x = tfl.Dense(unitsMid, activation='relu')(x)
output = tfl.Dense(unitsOut, activation='softmax')(x)

# Create the combined model
combined_model = keras.Model(inputs=inputEmbeddings, outputs=output)

combined_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
combined_model.summary()