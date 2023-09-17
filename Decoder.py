import tensorflow as tf
from tensorflow import keras as keras
from keras import layers as tfl
from keras import backend as K

def decoder(text, image, measurements):
    image = tfl.MultiHeadAttention(num_heads=18, key_dim=1024, dropout=0.3)(image, image)
    text = tfl.Dense(1024, activation='relu')(text)
    
    combined = tf.matmul(image, text, transpose_b=True)
    combined_features = tf.matmul(combined, text)
    X = tfl.LayerNormalization()(combined_features)
    X = tfl.MultiHeadAttention(num_heads=20, key_dim=1024, dropout=0.3)(X, X)

    # Define a custom function to reshape X based on measurements
    def reshape_x(i, X, measurements):
        height_i = tf.cast(measurements[i][0] // 16, tf.int32)
        width_i = tf.cast(measurements[i][1] // 16, tf.int32)
        X_i = tf.reshape(X[i], (-1, height_i, width_i, 1024))
        return i + 1, X_i, measurements

    i = tf.constant(0)
    _, reshaped_tensors, _ = tf.while_loop(lambda i, *_: i < tf.shape(measurements)[0],
                                            reshape_x,
                                            [i, X, measurements])

    # Stack the reshaped tensors along the first axis
    X = reshaped_tensors
    #X = tf.stack(reshaped_tensors)
    
    # Apply a final Conv2DTranspose layer if necessary
    X_upsampled = tfl.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', activation='relu')(X)
    X_upsampled = tfl.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', activation='relu')(X_upsampled)
    X_upsampled = tfl.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(X_upsampled)
    output = tfl.Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', activation='sigmoid')(X_upsampled)
    
    return output