import tensorflow as tf
from tensorflow import keras as keras
from keras import layers as tfl
from formatText import formatText

textInput = 'test'

embeddings = formatText(textInput)
out = tfl.Dense(1024, 'relu')(embeddings)

print(tf.shape(out))

#def textEncoder():
#    model = 0
#    return model