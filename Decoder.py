import tensorflow as tf
import optuna
from tensorflow import keras as keras
from keras import layers as tfl
from formatImg import formatTensorFromPath
from formatImg import formatImg
from ImgEncoder import imgEncoder
from formatText import formatText
from TextEncoder import textEncoder

#imgFilePath = "Test"
imgFilePath = "/Users/adityaasuratkal/Downloads/ML_Projects/UNet/Data/Indoor Semantic Segmentation/images/vedioDataCollection_July2019_Kent0001.png"
#imgFilePath = "/Users/adityaasuratkal/Downloads/ML_Projects/UNet/Data/ADEChallengeData2016/images/training/ADE_train_00000001.jpg"
fullEncodings, MAEencodings, shape = formatImg(formatTensorFromPath(imgFilePath))
print(tf.shape(fullEncodings))
imgEnc = imgEncoder()
imgEncodings = imgEnc(fullEncodings)

textInput = 'Hello World this is a test'
embeddings = formatText(textInput)
textEnc = textEncoder()
textEncodings = textEnc(embeddings)

def decoder(text, image, shape):
    image = tfl.MultiHeadAttention(num_heads=18, key_dim=1024, dropout=0.3)(image, image)
    text = tfl.Dense(1024, activation='relu')(text)
    
    # Accessing height and width values from the shape tensor
    height = shape[:, 0]
    width = shape[:, 1]
    
    combined = tf.matmul(image, text, transpose_b=True)
    combined_features = tf.matmul(combined, text)
    X = tfl.LayerNormalization()(combined_features)
    X = tfl.MultiHeadAttention(num_heads=20, key_dim=1024, dropout=0.3)(X, X)
    
    # Reshape the output tensor to match the desired spatial dimensions for the entire batch
    X_reshaped = tf.reshape(X, [-1, height[0] // 16, width[0] // 16, 1024])  # Using height[0] as an example
    
    # Upsample the spatial dimensions using transposed convolutional layers
    X_upsampled = tfl.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', activation='relu')(X_reshaped)
    X_upsampled = tfl.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', activation='relu')(X_upsampled)
    X_upsampled = tfl.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(X_upsampled)
    output = tfl.Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', activation='sigmoid')(X_upsampled)
    return output

# Define your input layers
img = keras.Input(shape=(None, 1024))  # Assuming the None dimension is the batch size
text = keras.Input(shape=(None, 300))  # Assuming the None dimension is the batch size
imgShape = keras.Input(shape=(4))  # Assuming the None dimension is the batch size

# Build the decoder model
imgEncode = imgEnc(img)
textEncode = textEnc(text)
output = decoder(textEncode, imgEncode, imgShape)

fullModel = keras.Model(inputs=[img, text, imgShape], outputs=output)
fullModel.summary()