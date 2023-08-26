import tensorflow as tf
import optuna
from tensorflow import keras as keras
from keras import layers as tfl
from formatImg import formatTensorFromPath
from formatImg import formatImg
from ImgEncoder import imgEncoder
from formatText import formatText

#imgFilePath = "Test"
imgFilePath = "/Users/adityaasuratkal/Downloads/ML_Projects/UNet/Data/Indoor Semantic Segmentation/images/vedioDataCollection_July2019_Kent0001.png"
#imgFilePath = "/Users/adityaasuratkal/Downloads/ML_Projects/UNet/Data/ADEChallengeData2016/images/training/ADE_train_00000001.jpg"

fullEncodings, MAEencodings, shape = formatImg(formatTensorFromPath(imgFilePath))
enc = imgEncoder()
output = enc(MAEencodings)

textInput = 'Hello World this is a test'
embeddings = formatText(textInput)


#return final mask with correct image shape using FCNN, MHA, and maybe reverse CNNs

def decoder(text, image, shape):
    image = tfl.MultiHeadAttention(num_heads=18, key_dim=1024, dropout=0.3)(image,  image)
    text = tfl.Dense(1024, 'relu')(text)
    X = tfl.Add()([image, text])
    X = tfl.Normalization()(X)
    X = tfl.MultiHeadAttention(num_heads=20, key_dim=1024, dropout=0.3)(X, X)
    X = tfl.Dense(512)(X)
    return X


print(shape)
out = decoder(embeddings, fullEncodings, shape)
print(tf.shape(out))