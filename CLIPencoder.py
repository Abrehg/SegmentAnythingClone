import tensorflow as tf
from tensorflow import keras as keras
from keras import layers as tfl
from formatText import formatText

textInput = 'test'

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

    input_embeddings = keras.Input(shape=(None, 1024))
    embeddingsWithPosition = add_positional_encodings(input_embeddings)
    X = tfl.Dense(1024, 'relu')(embeddingsWithPosition)
    output = encode(X)
    model = keras.Model(inputs=input_embeddings, outputs=output)
    
    return model

def get_positional_encoding(seq_length, d_model):
    i = tf.range(d_model, dtype=tf.float32) // 2
    angles = tf.pow(10000.0, -2 * i / tf.cast(d_model, dtype=tf.float32))
    positions = tf.cast(tf.range(seq_length), dtype=tf.float32)
    angles = tf.expand_dims(positions, -1) * tf.expand_dims(angles, 0)
    encoding = tf.concat([tf.sin(angles), tf.cos(angles)], axis=-1)
    return encoding

def add_positional_encodings(word_vectors):
    seq_length, d_model = tf.shape(word_vectors)[1], tf.shape(word_vectors)[2]
    positional_encodings = get_positional_encoding(seq_length, d_model)
    print(tf.shape(word_vectors))
    print(tf.shape(positional_encodings))
    positional_encodings = tf.tile(positional_encodings, [tf.shape(word_vectors)[0], 1, 1])
    word_vectors_with_position = word_vectors + positional_encodings
    return word_vectors_with_position

embeddings = formatText(textInput)
out = textEncoder()(embeddings)
print(tf.shape(out))