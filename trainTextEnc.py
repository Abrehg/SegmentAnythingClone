#Train text encoder model for Named Entity Recognition in order to generate text encodings

import tensorflow as tf
import keras
from keras import layers as tfl
import numpy as np
from TextEncoder import textEncoder

#Model input text vectors (num examples, sequence length, 300)
textEncodings_input = keras.Input((20, 300), name="text_encodings")
print(f"Input shape: {tf.shape(textEncodings_input)}")
textEnc = textEncoder()
Q = textEnc(textEncodings_input)
print(f"Output shape: {tf.shape(Q)}")

#Create decoder for Named Entity Recognition
out = textEnc(Q)

#Define final model to be trained
combinedModel = keras.Model(inputs = textEncodings_input, outputs = out)

#Save weights of text encoder model
textEnc.save_weights('./txt_encoder_weights.h5')