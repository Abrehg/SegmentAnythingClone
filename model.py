import tensorflow as tf
import optuna
from tensorflow import keras as keras
from keras import layers as tfl
from inputFormatting import formatImg
from inputFormatting import formatTensorFromPath
from FullEncoder import encoder
from formatText import formatText
from decodeFormat import decodeFormat
from Decoder import decoder

#encode text inputs using the CLIP architecture
#Then figure out how to decode into a full workable mask with lightweight decoder using MAE and CLIP 
#Finally train the decoder network to generate a workable mask using the Meta Segment Anything dataset

image = keras.Input((None, None, None, 3))
#input encoding of image (Transformer encoder + MAE)
fullEncodings, MAEencodings = formatImg(image)
X = encoder()(MAEencodings)

#input encoding of text (CLIP)
#Q = formatText(text)
#
#decode and return output mask (Transformer decoder)
#X = decodeFormat(X)
#mask = decoder(X)
#model = keras.Model(inputs = image, outputs = mask)
#print(model.summary())

#imgFilePath = "Test"
imgFilePath = "/Users/adityaasuratkal/Downloads/ML_Projects/UNet/Data/Indoor Semantic Segmentation/images/vedioDataCollection_July2019_Kent0001.png"
#imgFilePath = "/Users/adityaasuratkal/Downloads/ML_Projects/UNet/Data/ADEChallengeData2016/images/training/ADE_train_00000001.jpg"
text = "Hello World"

fullEncodings, MAEencodings = formatImg(formatTensorFromPath(imgFilePath))
X = encoder()(MAEencodings)

Q = formatText(text)
