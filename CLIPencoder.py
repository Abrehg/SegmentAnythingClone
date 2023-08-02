import tensorflow as tf
from tensorflow import keras as keras
from keras import layers as tfl

def textEncoder():
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
        X = tfl.LayerNormalization()(X)
        input2 = tfl.LayerNormalization()(X)
        X = feedForwardNN(input2)
        X = tf.cast(X, dtype=tf.float32)
        input2 = tf.cast(input2, dtype=tf.float32)
        X = tf.add(X, input2)
        output = tfl.LayerNormalization()(X)
        return output

    def encode(input_tensor):
        X = tf.cast(input_tensor, dtype=tf.float32)
        for i in range(0, encLayers):
            X = encoderLayer(X)
        return X

    input_embeddings = keras.Input(shape=(None, 300))
    X = tfl.Dense(1024, 'relu')(input_embeddings)
    X = add_positional_encodings(X)
    output = encode(X)
    model = keras.Model(inputs=input_embeddings, outputs=output)
    
    return model

def positional_encoding(seq_len, d_model):

    position = tf.range(seq_len, dtype=tf.float32)
    i = tf.range(d_model, dtype=tf.float32)
    
    angles = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, dtype=tf.float32))
    angles = tf.reshape(angles, (1, -1))
    
    angles = tf.multiply(position[:, tf.newaxis], angles)
    
    angles = tf.concat([tf.sin(angles[:, 0::2]), tf.cos(angles[:, 1::2])], axis=-1)
    encodings = tf.expand_dims(angles, axis=0)

    return encodings

def add_positional_encodings(word_vectors):
    seq_length = tf.shape(word_vectors)[1]
    d_model = tf.shape(word_vectors)[2]
    positional_encodings = positional_encoding(seq_length, d_model)
    word_vectors_with_position = word_vectors + positional_encodings
    return word_vectors_with_position