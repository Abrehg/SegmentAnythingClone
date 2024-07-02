#Convert string to vectors using GloVe (Global Vectors)

from gensim.models import KeyedVectors
import string
import numpy as np

#Load in GloVe file
glove_file = './glove.840B.300d.txt'
word_vectors = KeyedVectors.load_word2vec_format(glove_file, binary=False, no_header=True)

# Compute the mean vector (UNK)
mean_vector = np.mean(word_vectors.vectors, axis=0)

#Remove punctuation for easier input
def remove_punctuation(input_string):
    # Create a translation table to map punctuation characters to None
    translator = str.maketrans('', '', string.punctuation)
    
    # Remove punctuation from the input string
    cleaned_string = input_string.translate(translator)
    
    return cleaned_string

#General function for converting text to vector
def formatText(text):
    text = remove_punctuation(text)
    slicedInput = text.split()
    embeddings = []
    for word in slicedInput:
        if word in word_vectors:
            embeddings.append(word_vectors[word])
        else:
            embeddings.append(mean_vector)
    embeddings = np.array(embeddings)
    return embeddings

def formatTextOneWord(word):
    if word in word_vectors:
        embeddings = word_vectors[word]
    else:
        embeddings = mean_vector
    #embeddings = np.array(embeddings)
    #embeddings = np.expand_dims(embeddings, axis=0)
    return embeddings

#Find a word when given a vector
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