import tensorflow as tf
import optuna
from tensorflow import keras as keras
from keras import layers as tfl
from inputFormatting import formatImg
from Encoder import encoder
from formatText import formatText
from decodeFormat import decodeFormat
from Decoder import decoder

imgFilePath = "Test"
text = keras.Input(shape=(None))

#potential input for model
#if used, delete the first two lines in formatImg function (inputFormatting.py)
img = tf.io.read_file(imgFilePath)
tensor = tf.io.decode_image(img, channels=3, dtype=tf.dtypes.uint8)

#input encoding of image (Transformer encoder + MAE)
fullEncodings, MAEencodings, MAEpositions = formatImg(imgFilePath)
X = encoder()
#input encoding of text (CLIP)
Q = formatText(text)
#decode and return output mask (Transformer decoder)
X = decodeFormat(X, Q)
mask = decoder(X)
model = keras.Model(inputs = [imgFilePath, text], outputs = mask)
model.summary()