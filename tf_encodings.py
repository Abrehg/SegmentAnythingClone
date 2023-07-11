#Courtesy of chatgpt
import tensorflow as tf
import numpy as np
import math

def positional_encoding_2D(i, j, d_model):
    encoding = []
    for k in range(d_model):
        if k % 2 == 0:
            encoding.append(math.sin(i / (10000 ** (2 * k / d_model))) + math.sin(j / (10000 ** (2 * k / d_model))))
        else:
            encoding.append(math.cos(i / (10000 ** ((2 * k + 1) / d_model))) + math.cos(j / (10000 ** ((2 * k + 1) / d_model))))
    return encoding

#Adds the encoding to the original image embeddings
def add_positional_encoding(embeddings):
    width, length, d_model= embeddings.shape[0], embeddings.shape[1], embeddings.shape[2]
    finalEmbedding = np.ndarray((width, length, d_model))
    for i in range(0, width):
        for j in range(0, length):
            position_encoding = positional_encoding_2D(i, j, d_model)
            finalEmbedding[i][j] = position_encoding + embeddings[i][j]
    return embeddings + position_encoding
