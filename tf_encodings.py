#Courtesy of chatgpt
import tensorflow as tf
import numpy as np

#Calculates the angles based on the vertical and horizontal position
def get_angles(positions, i, d_model):
    angles = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return positions * angles

#Generates the positional encodings based on angles made in get_angles
def positional_encoding_2D(width, length, d_model):
    position_dims = d_model // 2
    width_enc = get_angles(np.arange(width)[:, np.newaxis], np.arange(position_dims)[np.newaxis, :], position_dims)
    length_enc = get_angles(np.arange(length)[:, np.newaxis], np.arange(position_dims)[np.newaxis, :], position_dims)

    # Apply sine and cosine to positional encodings
    pos_encoding = np.concatenate([np.sin(width_enc), np.cos(width_enc), np.sin(length_enc), np.cos(length_enc)], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float32)

#Adds the encoding to the original image embeddings
def add_positional_encoding(embeddings):
    num_patches, d_model = embeddings.shape[1], embeddings.shape[2]
    position_encoding = positional_encoding_2D(num_patches, d_model)
    return embeddings + position_encoding