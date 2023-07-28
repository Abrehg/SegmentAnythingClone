import tensorflow as tf
from gensim.models import KeyedVectors
import string

textInput = "Hello, World"

def formatText(text):
    text = remove_punctuation(text)
    slicedInput = text.split()
    glove_file = '/Users/adityaasuratkal/Downloads/glove.840B.300d.txt'
    word_vectors = KeyedVectors.load_word2vec_format(glove_file, binary=False, no_header=True)
    embeddings = [word_vectors[word] for word in slicedInput if word in word_vectors]

    return embeddings

def remove_punctuation(input_string):
    # Create a translation table to map punctuation characters to None
    translator = str.maketrans('', '', string.punctuation)
    
    # Remove punctuation from the input string
    cleaned_string = input_string.translate(translator)
    
    return cleaned_string

out = formatText(textInput)
print(tf.shape(out))