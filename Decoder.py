import tensorflow as tf
from keras import layers as tfl
import keras

def decoder(text, image):

    image = tfl.MultiHeadAttention(num_heads=18, key_dim=1024, dropout=0.3)(image, image)
    text = tfl.Dense(1024, activation='relu')(text)
    
    combined = tf.matmul(image, text, transpose_b=True)
    combined_features = tf.matmul(combined, text)
    X = tfl.LayerNormalization()(combined_features)
    X = tfl.MultiHeadAttention(num_heads=20, key_dim=1024, dropout=0.3)(X, X)
    
    X_upsampled = tfl.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', activation='relu')(X)
    X_upsampled = tfl.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', activation='relu')(X_upsampled)
    X_upsampled = tfl.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(X_upsampled)
    output = tfl.Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', activation='sigmoid', name="mask")(X_upsampled)
    
    return output

decoderTextIn = keras.Input([None, 300])
decoderImgIn = keras.Input([None, None, 1024])

decoderOut = decoder(decoderTextIn, decoderImgIn)

model = keras.Model(inputs = [decoderTextIn, decoderImgIn], outputs = decoderOut)

print(model.summary())