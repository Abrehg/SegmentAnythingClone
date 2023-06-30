import tensorflow as tf
import optuna
from tensorflow import keras as keras
from keras import layers as tfl
from inputFormatting import formatImg
from Encoder import encoder
from formatText import formatText
from decodeFormat import decodeFormat
from Decoder import decoder

image = keras.Input(shape=(None, None, 3))
text = keras.Input(shape=(None))

#input encoding of image (Transformer encoder + MAE)
X = formatImg(image)
X = encoder(X)
#input encoding of text (CLIP)
Q = formatText(text)
#decode and return output mask (Transformer decoder)
X = decodeFormat(X, Q)
mask = decoder(X)
model = keras.Model(inputs = [image, text], outputs = mask)
model.summary()