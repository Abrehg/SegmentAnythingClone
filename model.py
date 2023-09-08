import tensorflow as tf
import optuna
from tensorflow import keras as keras
from keras import layers as tfl
from formatImg import formatImg
from formatImg import formatTensorFromPath
from ImgEncoder import imgEncoder
from formatText import formatText
from formatText import findWord
from TextEncoder import textEncoder
from Decoder import decoder

#imgFilePath = "Test"
imgFilePath = "/Users/adityaasuratkal/Downloads/Img_Data/Indoor Semantic Segmentation/images/vedioDataCollection_July2019_Kent0001.png"
#imgFilePath = "/Users/adityaasuratkal/Downloads/Img_Data/ADEChallengeData2016/images/training/ADE_train_00000001.jpg"
text = "Hello World"

image = formatTensorFromPath(imgFilePath)
#input encoding of image (Transformer encoder + MAE)
#fullEncodings, MAEencodings, height, width = formatImg(image)
heights = keras.layers.Input((1))
widths = keras.layers.Input((1))
imgEncodings = keras.layers.Input((None, 1024))
X = imgEncoder()(imgEncodings)

#input encoding of text (CLIP)
#Q = formatText(text)
textEncodings = keras.layers.Input((None, 300))
textEnc = textEncoder()
Q = textEnc(textEncodings)

#decode and return output mask (Transformer decoder)
masks = []
num_elements = tf.shape(Q)[0]

# Convert num_elements_int to a NumPy integer
num_elements_int = tf.reduce_prod(num_elements)

if num_elements_int == None:
    num_elements_int = 1

def process_input(i):
    mask = decoder(
        Q[i], 
        X[i], 
        heights[i], 
        widths[i]
    )
    return mask

for ctr in range(num_elements_int):
    masks.append(process_input(ctr))

outputMasks = tf.concat(masks, axis = 0)

model = keras.Model(inputs = [imgEncodings, textEncodings, heights, widths], outputs = outputMasks)
print(model.summary())

#imgFilePath = "Test"
imgFilePath = "/Users/adityaasuratkal/Downloads/Img_Data/Indoor Semantic Segmentation/images/vedioDataCollection_July2019_Kent0001.png"
#imgFilePath = "/Users/adityaasuratkal/Downloads/Img_Data/ADEChallengeData2016/images/training/ADE_train_00000001.jpg"
text = "Hello World"

fullEncodings, MAEencodings, height, width = formatImg(formatTensorFromPath(imgFilePath))
X = imgEncoder()(fullEncodings)

Q = formatText(text)
textEnc = textEncoder()
Q = textEnc(Q)

out = decoder(Q, X, height, width)