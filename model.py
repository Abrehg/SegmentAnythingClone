import tensorflow as tf
import optuna
from tensorflow import keras as keras
from keras import layers as tfl
from inputFormatting import formatImg
from FullEncoder import encoder
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
X = encoder(MAEencodings)
#input encoding of text (CLIP)
Q = formatText(text)
#decode and return output mask (Transformer decoder)
X = decodeFormat(X, Q)
mask = decoder(X)
model = keras.Model(inputs = [imgFilePath, text], outputs = mask)
model.summary()


#imgFilePath = "Test"
imgFilePath = "/Users/adityaasuratkal/Downloads/ML_Projects/UNet/Data/Indoor Semantic Segmentation/images/vedioDataCollection_July2019_Kent0001.png"
#imgFilePath = "/Users/adityaasuratkal/Downloads/ML_Projects/UNet/Data/ADEChallengeData2016/images/training/ADE_train_00000001.jpg"

fullEncodings, MAEencodings = formatImg(imgFilePath)
X = encoder(MAEencodings)