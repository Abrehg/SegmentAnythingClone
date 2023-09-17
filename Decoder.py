import tensorflow as tf
from tensorflow import keras as keras
from keras import layers as tfl

def custom_reshape(x, height_i, width_i):
    return tf.reshape(x, (height_i, width_i, 1024))

def decoder(text, image, measurements):
    image = tfl.MultiHeadAttention(num_heads=18, key_dim=1024, dropout=0.3)(image, image)
    text = tfl.Dense(1024, activation='relu')(text)
    
    combined = tf.matmul(image, text, transpose_b=True)
    combined_features = tf.matmul(combined, text)
    X = tfl.LayerNormalization()(combined_features)
    X = tfl.MultiHeadAttention(num_heads=20, key_dim=1024, dropout=0.3)(X, X)

    # Create an empty list to store reshaped tensors
    reshaped_tensors = []

    # Custom for loop to reshape each output tensor slice
    for i in range(tf.shape(measurements)[0]):
        height_i = tf.cast(measurements[i][0] // 16, tf.int32)
        width_i = tf.cast(measurements[i][1] // 16, tf.int32)
        
        # Apply custom reshape to each slice of X
        X_i = custom_reshape(X[i], height_i, width_i)
        reshaped_tensors.append(X_i)

    # Stack the reshaped tensors along the first axis
    X = tf.stack(reshaped_tensors)
    
    # Apply a final Conv2DTranspose layer if necessary
    X_upsampled = tfl.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', activation='relu')(X)
    X_upsampled = tfl.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', activation='relu')(X_upsampled)
    X_upsampled = tfl.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(X_upsampled)
    output = tfl.Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', activation='sigmoid')(X_upsampled)
    
    return output