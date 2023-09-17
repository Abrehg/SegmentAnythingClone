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

imgFilePath = "/Users/adityaasuratkal/Downloads/Img_Data/Indoor Semantic Segmentation/images/vedioDataCollection_July2019_Kent0001.png"
#imgFilePath = "/Users/adityaasuratkal/Downloads/Img_Data/ADEChallengeData2016/images/training/ADE_train_00000001.jpg"
text = "Hello World"

fullEncodings, MAEencodings, measure = formatImg(formatTensorFromPath(imgFilePath))
X = imgEncoder()(fullEncodings)
sizes = tf.convert_to_tensor([measure])

Q = formatText(text)
textEnc = textEncoder()
Q = textEnc(Q)

out = decoder(Q, X, sizes)
print (tf.shape(out))

#input encoding of image (Transformer encoder + MAE)
#fullEncodings, MAEencodings, height, width = formatImg(image)
measurements_input = keras.layers.Input((2))
imgEncodings = keras.layers.Input((None, 1024))
X = imgEncoder()(imgEncodings)

#input encoding of text (CLIP)
#Q = formatText(text)
textEncodings = keras.layers.Input((None, 300))
textEnc = textEncoder()
Q = textEnc(textEncodings)

#decode and return output mask (Transformer decoder)
masks = decoder(Q, X, measurements_input)

model = keras.Model(inputs = [imgEncodings, textEncodings, measurements_input], outputs = masks)
print(model.summary())