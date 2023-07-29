import tensorflow as tf
from gensim.models import KeyedVectors
import string
import numpy as np

def formatText(text):
    text = remove_punctuation(text)
    slicedInput = text.split()
    glove_file = '/Users/adityaasuratkal/Downloads/glove.840B.300d.txt'
    word_vectors = KeyedVectors.load_word2vec_format(glove_file, binary=False, no_header=True)
    embeddings = [word_vectors[word] for word in slicedInput if word in word_vectors]
    word_vectors_with_position = add_positional_encodings(embeddings)
    finalEmbedding = tf.convert_to_tensor(word_vectors_with_position, dtype=tf.float32)

    return finalEmbedding

def get_positional_encoding(position, d_model):
    i = np.arange(d_model) // 2
    angles = position / 10000 ** (2 * i / d_model)
    encoding = np.zeros(d_model)
    encoding[0::2] = np.sin(angles)
    encoding[1::2] = np.cos(angles)
    return encoding

def add_positional_encodings(word_vectors):
    word_vectors_np = np.array(word_vectors)
    seq_length, d_model = word_vectors_np.shape
    positional_encodings = np.array([get_positional_encoding(i, d_model) for i in range(seq_length)])
    word_vectors_with_position = word_vectors_np + positional_encodings
    return word_vectors_with_position

def remove_punctuation(input_string):
    # Create a translation table to map punctuation characters to None
    translator = str.maketrans('', '', string.punctuation)
    
    # Remove punctuation from the input string
    cleaned_string = input_string.translate(translator)
    
    return cleaned_string