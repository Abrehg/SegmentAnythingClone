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
textEncodings_input = keras.Input((None, 300), ragged = True, name="text_encodings")
textEnc = textEncoder()
Q = textEnc(textEncodings_input)

#Create decoder for Named Entity Recognition
X = tfl.Dense(units=150, activation="relu")(Q)
X = tfl.Dense(units=75, activation="relu")(X)
X = tfl.Dense(units=25, activation="relu")(X)
X = tfl.Dense(units=10, activation="relu")(X)
X = tfl.Dense(units=1, activation="relu")(X)
out = tf.squeeze(X, axis=-1)

#Define final model to be trained
combinedModel = keras.Model(inputs=textEncodings_input, outputs=out)

print("Text encoder: ")
textEnc.summary()

print("Combined model:")
combinedModel.summary()

# Define custom loss to handle ragged tensors
def sparse_categorical_crossentropy_per_sample(y_true, y_pred):
    if isinstance(y_true, tf.RaggedTensor):
        y_true = y_true.flat_values
    if isinstance(y_pred, tf.RaggedTensor):
        y_pred = y_pred.flat_values
    
    y_true = tf.reshape(y_true, [-1])
    y_pred_shape = tf.shape(y_pred)
    y_pred = tf.reshape(y_pred, [y_pred_shape[0], -1]) # type: ignore
    
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    return tf.reduce_mean(loss)

class RaggedAccuracy(keras.metrics.Metric):
    def __init__(self, name='accuracy', **kwargs):
        super(RaggedAccuracy, self).__init__(name=name, **kwargs)
        self.correct = self.add_weight(name='correct', initializer='zeros', dtype=tf.float32)
        self.total = self.add_weight(name='total', initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Flatten RaggedTensors if necessary
        if isinstance(y_true, tf.RaggedTensor):
            y_true = y_true.flat_values
        if isinstance(y_pred, tf.RaggedTensor):
            y_pred = y_pred.flat_values
        
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.argmax(tf.reshape(y_pred, [tf.shape(y_pred)[0], -1]), axis=-1)
        
        # Calculate matches
        matches = tf.cast(tf.equal(y_true, y_pred), tf.float32)
        self.correct.assign_add(tf.reduce_sum(matches))
        
        # Calculate total number of elements
        total_elements = tf.size(y_true, out_type=tf.int32)
        self.total.assign_add(tf.cast(total_elements, tf.float32))

    def result(self):
        return self.correct / self.total

    def reset_states(self):
        self.correct.assign(0)
        self.total.assign(0)

# Compile model
combinedModel.compile(
    optimizer=keras.optimizers.legacy.Adam(learning_rate=0.001),
    loss=sparse_categorical_crossentropy_per_sample,
    metrics=[RaggedAccuracy()]
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
    decoded_tokens = []
    for token in tokens:
        if isinstance(token, bytes):
            decoded_tokens.append(token.decode('utf-8'))
        else:
            # Handle cases where token is not a byte string
            decoded_tokens.append(str(token))
    return decoded_tokens

# Split unknown word into known words (if known words are separated by punctuation)
def splitWord(word):
    if not isinstance(word, str):
        word = str(word)
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
    pos_tags_adjusted = []

    for token, pos_tag in zip(tokens, pos_tags):
        emb = formatTextOneWord(token)
        if np.array_equal(emb, mean_vector):
            split_tokens = splitWord(token)
            for split_token in split_tokens:
                split_emb = formatTextOneWord(split_token)
                if isinstance(split_emb, np.ndarray):
                    split_emb = split_emb.tolist()
                embeddings.append(split_emb)
                pos_tags_adjusted.append(pos_tag)
        else:
            if isinstance(emb, np.ndarray):
                emb = emb.tolist()
            embeddings.append(emb)
            pos_tags_adjusted.append(int(pos_tag))

    return embeddings, pos_tags_adjusted

# Test preprocess_example directly
testPosTags = [12, 22, 22, 38, 15, 22, 28, 38, 15, 16, 21, 35, 24, 35, 37, 16, 21, 15, 24, 41, 15, 16, 21, 21, 20, 37, 40, 35, 21, 7]
testTokens = ["The", "European", "Commission", "said", "on", "Thursday", "it", "disagreed", "with", "German", "advice", "to", "consumers", "to", "shun", "British", "lamb", "until", "scientists", "determine", "whether", "mad", "cow", "disease", "can", "be", "transmitted", "to", "sheep", "."]
testPosTags = tf.convert_to_tensor(testPosTags, tf.int64)
testTokens = tf.convert_to_tensor(testTokens, tf.string)

def data_generator(data):
    for entry in data:
        if 'tokens' in entry and 'pos' in entry:
            tokens = entry['tokens']
            pos_tags = entry['pos']
            print(tokens)
            print(pos_tags)

            pos_tags = tf.convert_to_tensor(pos_tags, tf.int64)
            tokens = tf.convert_to_tensor(tokens, tf.string)

            embeddings, pos_tags_adjusted = preprocess_example(tokens, pos_tags)

            print(tf.shape(embeddings))
            print(type(embeddings))
            print(tf.shape(pos_tags_adjusted))
            print(type(pos_tags_adjusted))
            print("First shape and type check")

            embeddings = tf.ragged.constant(embeddings, dtype=tf.float32)
            pos_tags_adjusted = tf.ragged.constant(pos_tags_adjusted, dtype=tf.int64)

            print(tf.shape(embeddings))
            print(type(embeddings))
            print(tf.shape(pos_tags_adjusted))
            print(type(pos_tags_adjusted))
            print("Second shape and type check")

            yield embeddings, pos_tags_adjusted

# Starting dataset compilation
print("Starting dataset compilation")
BATCH_SIZE = 64

# Concatenate the datasets
combined = train.concatenate(dev).concatenate(test)
shuffled = combined.shuffle(buffer_size=10000)

# Convert the shuffled dataset to a list of entries
entries = list(shuffled.as_numpy_iterator())
print(f"Number of entries: {len(entries)}")

# # Test the generator function
# gen = data_generator(entries)
# for i, sample in enumerate(gen):
#     if i >= 1:  # Just take one sample for debugging
#         break
#     print(f"Sample {i}: {sample}")

## problem here
# test with smaller ragged dataset first
dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(entries),
    output_signature=(
        tf.RaggedTensorSpec(shape=[None, 300], dtype=tf.float32, ragged_rank=1),
        tf.RaggedTensorSpec(shape=[None], dtype=tf.int64, ragged_rank=0)
    )
)

print("Sample entries from dataset:")
for batch in dataset.take(1):
    inputs_batch, outputs_batch = batch
    print("Inputs batch:", inputs_batch)
    print("Outputs batch:", outputs_batch)

print("Training data complete")

# #Train the model
# EPOCHS = 10
# combinedModel.fit(dataset, epochs = EPOCHS, batch_size=BATCH_SIZE)
# print("Model fitted")

# #Save weights of text encoder model
# textEnc.save_weights('./txt_encoder_weights.h5')