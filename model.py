import tensorflow as tf
import optuna
from tensorflow import keras as keras
from keras import layers as tfl
from formatImg import formatImg
from formatImg import formatTensorFromPath
from ImgEncoder import imgEncoder
from formatText import formatText
from TextEncoder import textEncoder
from Decoder import decoder
import numpy as np

imgFilePath = "/Users/adityaasuratkal/Downloads/Img_Data/Indoor Semantic Segmentation/images/vedioDataCollection_July2019_Kent0001.png"
#imgFilePath = "/Users/adityaasuratkal/Downloads/Img_Data/ADEChallengeData2016/images/training/ADE_train_00000001.jpg"
text = "Hello World"

fullEncodings, MAEencodings, measure = formatImg(formatTensorFromPath(imgFilePath))
imgEnc = imgEncoder()
X = imgEnc(fullEncodings)
sizes = tf.convert_to_tensor([measure])

Q = formatText(text)
textEnc = textEncoder()
Q = textEnc(Q)

out = decoder(Q, X, sizes)
print (tf.shape(out))

#input encoding of image (Transformer encoder + MAE)
measurements_input = keras.layers.Input((2))
imgEncodings_input = keras.Input((None, 1024), name="img_encodings")
imgEnc = imgEncoder()
X = imgEnc(imgEncodings_input)

#input encoding of text (CLIP)
textEncodings_input = keras.Input((None, 300), name="text_encodings")
textEnc = textEncoder()
Q = textEnc(textEncodings_input)

#decode and return output mask
masks = decoder(Q, X, measurements_input)

model = keras.Model(inputs = [imgEncodings_input, textEncodings_input, measurements_input], outputs = masks)
print(model.summary())


imgFilePath2 = "/Users/adityaasuratkal/Downloads/Img_Data/ADEChallengeData2016/images/training/ADE_train_00000001.jpg"
fullImgEncodings, MAEencodings, measure2 = formatImg(formatTensorFromPath(imgFilePath2))
print(measure2)
sizes = tf.convert_to_tensor([measure2])

textEncodings = formatText(text)

out = model.predict([fullImgEncodings, textEncodings, sizes])
print(tf.shape(out))