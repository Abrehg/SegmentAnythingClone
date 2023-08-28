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
encodings = enc(fullEncodings)

textInput = 'Hello World this is a test'
embeddings = formatText(textInput)

def decoder(text, image, shape):
    image = tfl.MultiHeadAttention(num_heads=18, key_dim=1024, dropout=0.3)(image, image)
    text = tfl.Dense(1024, 'relu')(text)
    combined = tf.matmul(image, text, transpose_b=True)
    combined_features = tf.matmul(combined, text)
    X = tfl.LayerNormalization()(combined_features)
    print(tf.shape(X))
    X = tfl.MultiHeadAttention(num_heads=20, key_dim=1024, dropout=0.3)(X, X)
    
    # Reshape the output tensor to match the desired spatial dimensions
    X_reshaped = tf.reshape(X, [-1, shape[1] // 16, shape[2] // 16, 1024])
    print(tf.shape(X_reshaped))

    # Upsample the spatial dimensions using transposed convolutional layers
    X_upsampled = tfl.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', activation='relu')(X_reshaped)
    print(tf.shape(X_upsampled))
    X_upsampled = tfl.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', activation='relu')(X_upsampled)
    print(tf.shape(X_upsampled))
    X_upsampled = tfl.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(X_upsampled)
    print(tf.shape(X_upsampled))
    output = tfl.Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', activation='sigmoid')(X_upsampled)
    print(tf.shape(output))
    
    return output

print(shape)
out = decoder(embeddings, encodings, shape)
print(tf.shape(out))