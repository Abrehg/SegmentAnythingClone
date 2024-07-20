#Define text encoder architecture

import tensorflow as tf
import keras
from keras import layers as tfl

#Text encoder architecture (Derived from basic Transformer architecture)
#Input: (Num entries, sequence length, 300)
#Output: encodings for decoder (Num entries, sequence length, 300)
def textEncoder():
    units = 300
    encLayers = 3
    NN_Layers = 5

    # Feedforward network
    def feedForwardNN(inputs):
        X = inputs
        for j in range(NN_Layers):
            X = tfl.Dense(units, activation='relu')(X)
        return X

    # Encoder layer function
    def encoderLayer(inputs):
        # Multi-head attention
        attention_output = tfl.MultiHeadAttention(num_heads=16, key_dim=units, dropout=0.3)(inputs, inputs)
        
        # Residual connection and normalization
        X = tfl.LayerNormalization()(inputs + attention_output)
        
        # Feedforward network
        feedforward_output = feedForwardNN(X)
        
        # Residual connection and normalization
        output = tfl.LayerNormalization()(X + feedforward_output)
        
        return output

    # Function to encode input_tensor through multiple encoder layers
    def encode(input_tensor):
        X = input_tensor
        for i in range(encLayers):
            X = encoderLayer(X)
        return X

    # Define input layer with ragged=True for ragged tensor support
    input_embeddings = keras.Input(shape=(None, 300), ragged=True)

    # Add positional encodings to input embeddings
    X = add_positional_encodings(input_embeddings)

    # Encode the input embeddings
    encoded_output = encode(X)

    # Define the Keras model
    model = keras.Model(inputs=input_embeddings, outputs=encoded_output)

    return model

#Generate Positional Encodings
def positional_encoding(seq_len, d_model):
    position = tf.range(seq_len, dtype=tf.float32)
    i = tf.range(d_model, dtype=tf.float32)
    
    angles = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, dtype=tf.float32))
    angles = tf.reshape(angles, (1, d_model))
    angles = tf.multiply(position[:, tf.newaxis], angles)
    angles = tf.concat([tf.math.sin(angles[:, 0::2]), tf.math.cos(angles[:, 1::2])], axis=-1)
    encodings = tf.expand_dims(angles, axis=0)

    return encodings

def add_positional_encodings(word_vectors):
    # Convert to dense tensor if necessary
    if isinstance(word_vectors, tf.RaggedTensor):
        word_vectors = word_vectors.to_tensor()

    # Get sequence length and model dimension
    seq_len = tf.shape(word_vectors)[-2]
    d_model = tf.shape(word_vectors)[-1]

    # Generate positional encodings
    positional_encodings = positional_encoding(seq_len, d_model)

    # Add positional encodings to word_vectors
    word_vectors_with_position = word_vectors + positional_encodings

    return word_vectors_with_position