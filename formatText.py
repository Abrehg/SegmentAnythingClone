import tensorflow as tf
from gensim.models import KeyedVectors

textInput = "Hello World"

def formatText(text):
    slicedInput = text.split(" ")
    glove_file = '/Users/adityaasuratkal/Downloads/glove.840B.300d.txt'
    word_vectors = KeyedVectors.load_word2vec_format(glove_file, binary=False, no_header=True)
    embeddings = [word_vectors[word] for word in slicedInput if word in word_vectors]

    return embeddings

out = formatText(textInput)
print(tf.shape(out))