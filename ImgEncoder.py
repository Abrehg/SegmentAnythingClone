import tensorflow as tf
from tensorflow import keras as keras
from keras import layers as tfl

def encoder():
    encLayers = 3
    NNLayers = 5
    units = 1024

    def feedForwardNN(baseInput):
        X = baseInput
        for k in range(0, NNLayers):
            X = tfl.Dense(activation='relu', units=units)(X)
        return X

    def encoderLayer(inputs):
        X = tfl.MultiHeadAttention(num_heads=16, key_dim=1024, dropout=0.3)(inputs, inputs)
        X = tf.cast(X, dtype=tf.float32)
        inputs = tf.cast(inputs, dtype=tf.float32)
        X = tf.add(X, inputs)
        input2 = tfl.LayerNormalization()(X)
        X = feedForwardNN(input2)
        X = tf.cast(X, dtype=tf.float32)
        input2 = tf.cast(input2, dtype=tf.float32)
        X = tf.add(X, input2)
        output = tfl.LayerNormalization()(X)
        return output

    def encode(input):
        X = tf.cast(input, dtype=tf.float32)
        for i in range(0, encLayers):
            X = encoderLayer(X)
        return X

    baseInput = keras.Input(shape=(None, 1024))
    output = encode(baseInput)
    model = keras.Model(inputs=baseInput, outputs=output)
    
    return model