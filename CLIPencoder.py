import tensorflow as tf
from tensorflow import keras as keras
from keras import layers as tfl
from formatText import formatText
import numpy as np

textInput = 'Hello World'

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
    position_encodings = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(d_model):
            if i % 2 == 0:
                position_encodings[pos, i] = np.sin(pos / (10000 ** (2 * i / d_model)))
            else:
                position_encodings[pos, i] = np.cos(pos / (10000 ** ((2 * i - 1) / d_model)))
    
    encodings = tf.convert_to_tensor(position_encodings, dtype=tf.float32)

    return encodings

def add_positional_encodings(word_vectors):
    seq_length = word_vectors.shape[1]
    d_model = word_vectors.shape[2]
    positional_encodings = positional_encoding(seq_length, d_model)
    print(tf.shape(word_vectors)) # (None, None, 1024)
    print(tf.shape(positional_encodings)) # (None, 2048)
    #positional_encodings = tf.tile(tf.expand_dims(positional_encodings, 0), [tf.shape(word_vectors)[0], 1, 1])
    word_vectors_with_position = word_vectors + positional_encodings
    return word_vectors_with_position

embeddings = formatText(textInput)
out = textEncoder()(embeddings)
print(tf.shape(out))