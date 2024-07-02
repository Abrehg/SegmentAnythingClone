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

    def feedForwardNN(inputs):
        X = inputs
        for j in range(NN_Layers):
            X = tfl.Dense(units, activation='relu')(X)
        return X

    def encoderLayer(inputs):
        # Convert ragged tensor to dense tensor
        dense_inputs = inputs.to_tensor()

        X = tfl.MultiHeadAttention(num_heads=16, key_dim=units, dropout=0.3)(dense_inputs, dense_inputs)
        X = tf.RaggedTensor.from_tensor(X, inputs.nested_row_lengths())
        
        X = tf.add(X, inputs)
        X = tfl.LayerNormalization()(X)
        X = feedForwardNN(X)
        X = tf.add(X, inputs)
        output = tfl.LayerNormalization()(X)
        return output

    def encode(input_tensor):
        X = input_tensor
        for i in range(encLayers):
            X = encoderLayer(X)
        return X

    input_embeddings = keras.Input(shape=(None, 300), ragged=True)

    X = add_positional_encodings(input_embeddings)
    encoded_output = encode(X)

    model = keras.Model(inputs=input_embeddings, outputs=encoded_output)

    return model

#Generate Positional Encodings
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
    # Get sequence lengths for each row
    row_lengths = word_vectors.row_lengths(axis=1)
    max_seq_length = tf.reduce_max(row_lengths)
    d_model = tf.shape(word_vectors.values)[1]
    
    # Create positional encodings for the maximum sequence length
    positional_encodings = positional_encoding(max_seq_length, d_model)
    
    # Expand positional encodings to match batch size
    positional_encodings = tf.tile(positional_encodings, [tf.shape(row_lengths)[0], 1, 1])
    
    # Create a mask for truncating the positional encodings to the actual sequence lengths
    mask = tf.sequence_mask(row_lengths, max_seq_length)
    
    # Convert mask to a 3D tensor
    mask = tf.cast(mask, dtype=tf.float32)
    mask = tf.expand_dims(mask, axis=-1)
    
    # Apply the mask to the positional encodings
    truncated_encodings = positional_encodings * mask
    
    # Convert the truncated encodings to a ragged tensor
    truncated_encodings = tf.RaggedTensor.from_tensor(truncated_encodings, lengths=row_lengths)
    
    # Add positional encodings to the word vectors
    word_vectors_with_position = tf.ragged.map_flat_values(tf.add, word_vectors, truncated_encodings)
    
    return word_vectors_with_position