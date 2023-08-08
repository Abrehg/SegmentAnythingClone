import tensorflow as tf
from gensim.models import KeyedVectors
import string
import numpy as np

glove_file = '/Users/adityaasuratkal/Downloads/glove.840B.300d.txt'
word_vectors = KeyedVectors.load_word2vec_format(glove_file, binary=False, no_header=True)

def formatText(text):
    text = remove_punctuation(text)
    slicedInput = text.split()
    embeddings = [word_vectors[word] for word in slicedInput if word in word_vectors]
    embeddings = np.array(embeddings)
    embeddings = np.expand_dims(embeddings, axis=0)
    return embeddings

def findWord(input_vector):
    most_similar_words = []
    for sentence_vectors in input_vector:
        predicted_sentence = ""
        for word_vector in sentence_vectors:
            most_similar_word = word_vectors.similar_by_vector(word_vector, topn=1)[0][0]
            predicted_sentence = predicted_sentence + most_similar_word + " "
        predicted_sentence = predicted_sentence[:-1]
        most_similar_words.append(predicted_sentence)
    return most_similar_words

def remove_punctuation(input_string):
    # Create a translation table to map punctuation characters to None
    translator = str.maketrans('', '', string.punctuation)
    
    # Remove punctuation from the input string
    cleaned_string = input_string.translate(translator)
    
    return cleaned_string