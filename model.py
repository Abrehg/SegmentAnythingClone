import tensorflow as tf
import optuna
from tensorflow import keras as keras
from formatImg import formatImg
from formatImg import formatTensorFromPath
from ImgEncoder import imgEncoder
from formatText import formatText
from TextEncoder import textEncoder
from Decoder import decoder

imgFilePath = "./vedioDataCollection_July2019_Kent0001.png"
text = "Hello World"

fullEncodings, MAEencodings, measure = formatImg(formatTensorFromPath(imgFilePath))
print(f"path to Img shape: {tf.shape(fullEncodings)}")
imgEnc = imgEncoder()
X = imgEnc(fullEncodings)
print(f"img encodings shape: {tf.shape(X)}")

Q = formatText(text)
textEnc = textEncoder()
Q = textEnc(Q)
print(f"text encodings shape: {tf.shape(Q)}")

out = decoder(Q, X)
print (f"output shape: {tf.shape(out)}")

"""
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


imgFilePath2 = "./ADE_train_00000001.jpg"
fullImgEncodings, MAEencodings, measure2 = formatImg(formatTensorFromPath(imgFilePath2))
print(measure2)
sizes = tf.convert_to_tensor([measure2])

textEncodings = formatText(text)

out = model.predict([fullImgEncodings, textEncodings, sizes])
print(tf.shape(out))
"""