import tensorflow as tf
from tensorflow import keras as keras
from keras import layers as tfl
from formatText import formatText
from CLIPencoder import textEncoder

textInput = 'Hello World'

embeddings = formatText(textInput)
out = textEncoder()(embeddings)
print(tf.shape(out))