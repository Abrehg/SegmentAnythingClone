import tensorflow as tf
from tensorflow import keras as keras
from keras import layers as tfl
from formatImg import formatTensorFromPath
from formatImg import formatImg
from ImgEncoder import imgEncoder
from formatText import formatText
from TextEncoder import textEncoder

#imgFilePath = "Test"
imgFilePath = "/Users/adityaasuratkal/Downloads/Img_Data/Indoor Semantic Segmentation/images/vedioDataCollection_July2019_Kent0001.png"
#imgFilePath = "/Users/adityaasuratkal/Downloads/Img_Data/ADEChallengeData2016/images/training/ADE_train_00000001.jpg"
fullEncodings, MAEencodings, height, width = formatImg(formatTensorFromPath(imgFilePath))
imgEnc = imgEncoder()
imgEncodings = imgEnc(fullEncodings)
heights = [height]
widths = [width]

textInput = 'Hello World this is a test'
embeddings = formatText(textInput)
textEnc = textEncoder()
textEncodings = textEnc(embeddings)

def decoder(text, image, height, width):
    image = tfl.MultiHeadAttention(num_heads=18, key_dim=1024, dropout=0.3)(image, image)
    text = tfl.Dense(1024, activation='relu')(text)
    
    combined = tf.matmul(image, text, transpose_b=True)
    combined_features = tf.matmul(combined, text)
    X = tfl.LayerNormalization()(combined_features)
    X = tfl.MultiHeadAttention(num_heads=20, key_dim=1024, dropout=0.3)(X, X)
    
    # Reshape each output tensor in the batch to its specified spatial dimensions
    X = tf.reshape(X, [-1, tf.cast(height // 16, tf.int32), tf.cast(width // 16, tf.int32), 1024])

    # Upsample the spatial dimensions using transposed convolutional layers
    X_upsampled = tfl.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', activation='relu')(X)
    X_upsampled = tfl.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', activation='relu')(X_upsampled)
    X_upsampled = tfl.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(X_upsampled)
    output = tfl.Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', activation='sigmoid')(X_upsampled)
    return output

out = decoder(textEncodings, imgEncodings, heights[0], widths[0])

"""
# Define your input layers
img = keras.Input(shape=(None, 1024))  # Assuming the None dimension is the batch size
text = keras.Input(shape=(None, 300))  # Assuming the None dimension is the batch size
imgShape = keras.Input(shape=(2))  # Assuming the None dimension is the batch size

# Build the decoder model
imgEncode = imgEnc(img)
textEncode = textEnc(text)
shapes = [[1000, 2000], [1000, 2000], [300, 400]]
output = decoder(textEncodings, imgEncodings, shape)

#fullModel = keras.Model(inputs=[img, text, imgShape], outputs=output)
#fullModel.summary()
"""