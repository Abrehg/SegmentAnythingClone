#Train text encoder model for Named Entity Recognition in order to generate text encodings

import tensorflow as tf
import keras
from keras import layers as tfl
from TextEncoder import textEncoder
from formatText import formatTextOneWord
import tensorflow_datasets as tfds
import numpy as np
from formatText import mean_vector

#Define your model
textEncodings_input = keras.Input((None, 300), name="text_encodings")
textEnc = textEncoder()
Q = textEnc(textEncodings_input)

#Create decoder for Named Entity Recognition
X = tfl.Dense(units=150, activation="relu")(Q)
X = tfl.Dense(units=75, activation="relu")(X)
X = tfl.Dense(units=25, activation="relu")(X)
X = tfl.Dense(units=10, activation="relu")(X)
out = tfl.Dense(units=1, activation="relu")(X)

#Define final model to be trained
combinedModel = keras.Model(inputs=textEncodings_input, outputs=out)

#Compile model
combinedModel.compile(
    optimizer=keras.optimizers.legacy.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#Load the dataset
train = tfds.load('conll2003', split='train', shuffle_files=True)
assert isinstance(train, tf.data.Dataset)
test = tfds.load('conll2003', split='test', shuffle_files=True)
assert isinstance(test, tf.data.Dataset)
dev = tfds.load('conll2003', split='dev', shuffle_files=True)
assert isinstance(dev, tf.data.Dataset)

# Function to decode byte strings to regular strings
def decode_tokens(tokens):
    return [token.decode('utf-8') for token in tokens]

# Split unknown word into known words (if known words are separated by punctuation)
def splitWord(word):
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 
               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
               "'", '-', '+', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    punctuation = ['.', '/', '*', '+', '=', ',', ':', ';', ')', '&', "`"]
    words = []
    temp = ""
    i = 0
    if (word == "25.49,(3-yr") or (word == "m;+21.5yr") or (word == "Videoton(*") or (word == "6-7(3-7") or (word == "+$0.50") or (word == "0#NKEL.RUO"):
        return [word]
    while i < len(word):
        if word[i] in letters:
            temp = temp + word[i]
        else:
            if (i > 0 and (word[i-1] in numbers or word[i-1] in punctuation)) and (word[i] in punctuation):
                temp = temp + word[i]
            elif word[i-1] == '.' and word[i] == '.':
                temp = temp + word[i]
            elif word[i] in punctuation:
                temp = temp + word[i]
            elif (i > 0 and word[i-1] in letters) and (word[i] == '.'):
                temp = temp + word[i]
            elif (i < len(word)-1 and (word[i+1] in numbers or word[i+1] in punctuation)) and (word[i] in punctuation):
                temp = temp + word[i]
            else:
                if temp != "":
                    words.append(temp)
                words.append(word[i])
                temp = ""
        i = i + 1
    if temp != "":
        words.append(temp)
    return words

# Preprocess example to convert tokens to GloVe embeddings (Test and fix)
def preprocess_example(tokens, pos_tags):
    tokens = decode_tokens(tokens.numpy())
    embeddings = []
    i = 0
    for token in tokens:
        print(token)
        print(pos_tags[i])
        print(i)
        print(len(pos_tags))
        embeddings.append(formatTextOneWord(token))
        if np.array_equal(embeddings[-1], mean_vector):
            print("Re proccessing")
            i = i - 1
            embeddings.pop()
            split = splitWord(token)
            for word in split:
                if i == len(pos_tags):
                    break
                print(word)
                print(pos_tags[i])
                print(i)
                print(len(pos_tags))
                embeddings.append(formatTextOneWord(word))
                i = i + 1
        i = i + 1
    print(tf.shape(embeddings))
    print(tf.shape(pos_tags))
    embeddings = tf.convert_to_tensor(embeddings, dtype = tf.float32)
    pos_tags = tf.convert_to_tensor(pos_tags, dtype = tf.int64)
    return embeddings, pos_tags

testPosTags = [12, 22, 22, 38, 15, 22, 28, 38, 15, 16, 21, 35, 24, 35, 37, 16, 21, 15, 24, 41, 15, 16, 21, 21, 20, 37, 40, 35, 21, 7]
testTokens = ["The", "European", "Commission", "said", "on", "Thursday", "it", "disagreed", "with", "German", "advice", "to", "consumers", "to", "shun", "British", "lamb", "until", "scientists", "determine", "whether", "mad", "cow", "disease", "can", "be", "transmitted", "to", "sheep", "."]
testPosTags = tf.convert_to_tensor(testPosTags, tf.int64)
testTokens = tf.convert_to_tensor(testTokens, tf.string)

# Wrapper for TensorFlow's tf.py_function
def tf_preprocess_example(example):
    tokens = example['tokens']
    pos_tags = example['pos']
    token_encodings, pos_tags = preprocess_example(tokens, pos_tags)
    if(token_encodings.shape[0] == 0):
        token_encodings, pos_tags = preprocess_example(testTokens, testPosTags)
    token_encodings.set_shape([None, 300])
    pos_tags.set_shape([None])
    return token_encodings, pos_tags

#Apply preprocessing to datasets
trainingDataX = []
trainingDataY = []

for entry in train:
    print("Train Entry started")
    X, Y = tf_preprocess_example(entry)
    trainingDataX.append(X.numpy())
    trainingDataY.append(Y.numpy())

for entry in dev:
    print("Dev Entry started")
    X, Y = tf_preprocess_example(entry)
    trainingDataX.append(X.numpy())
    trainingDataY.append(Y.numpy())

for entry in test:
    print("Test Entry started")
    X, Y = tf_preprocess_example(entry)
    trainingDataX.append(X.numpy())
    trainingDataY.append(Y.numpy())

print("Training data complete")

# Convert to numpy array for training step
print("Starting processing training data")
trainingDataX = tf.ragged.constant(trainingDataX)
print("Training Data X compiled")
trainingDataY = tf.ragged.constant(trainingDataY)
print("Training Data Y compiled")
print("Ragged tensors built")

#Train the model
EPOCHS = 10
BATCH_SIZE = 64
combinedModel.fit(x = trainingDataX, y = trainingDataY, epochs = EPOCHS, batch_size = BATCH_SIZE)
print("Model fitted")

#Save weights of text encoder model
textEnc.save_weights('./txt_encoder_weights.h5')