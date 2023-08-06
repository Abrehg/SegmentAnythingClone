import tensorflow as tf
from tensorflow import keras as keras
from keras import layers as tfl
from formatText import formatText
from CLIPencoder import textEncoder
import tensorflow_datasets as tfds
import pandas as pd
import gc
import numpy as np

#textInput = 'Hello World'

#embeddings = formatText(textInput)

NNLayers = 7
unitsMid = 1000
unitsOut = 1

inputEmbeddings = keras.Input((None, 300))
textEnc = textEncoder()
x = textEnc(inputEmbeddings)

for i in range(NNLayers):
    x = tfl.Dense(unitsMid, activation='relu')(x)
output = tfl.Dense(unitsOut, activation='sigmoid')(x)

# Create the combined model
combined_model = keras.Model(inputs=inputEmbeddings, outputs=output)

combined_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
combined_model.summary()

#sentiment analysis training
dataset_name = 'multi_news'
(train_data, val_data, test_data), info = tfds.load(dataset_name, split=['train', 'validation', 'test'], shuffle_files=True, with_info=True)
print(info)

def process_dataset(data, key):
    vectors = [formatText(entry[key].numpy().decode("utf-8")) for entry in data]
    return vectors

x_train = process_dataset(train_data, 'document')
y_train = process_dataset(train_data, 'summary')

x_val = process_dataset(val_data, 'document')
y_val = process_dataset(val_data, 'summary')

x_test = process_dataset(test_data, 'document')
y_test = process_dataset(test_data, 'summary')

"""
batch_size = 512
num_epochs = 10

combined_model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs)

combined_model.fit(x=x_val, y=y_val, batch_size=batch_size, epochs=num_epochs)

combined_model.test_on_batch(x=x_test, y=y_test)

textEnc.save_weights('./text_encoder_weights.h5')
"""
"""
with tf.GradientTape() as tape:
    predictions = model(inputs)
    loss = loss_fn(targets, predictions)
"""

def transformer_decoder_layer(inputs, enc_output):
    # Define your decoder layer using Functional API
    # Example:
    attention = tfl.Attention()([inputs, enc_output])
    add = tfl.Add()([attention, inputs])
    normalized = tfl.LayerNormalization()(add)
    ffn = tfl.Dense(512, activation='relu')(normalized)
    outputs = tfl.Dense(300)(ffn)  # Modify the output dimension to match GloVe vectors
    return outputs

def generate_sequence(encoder_output, start_token):
    # Initialize sequence with start token
    sequence = [start_token]
    decoder_input = tf.expand_dims([start_token], 0)

    max_length = 20000

    # Iterate over each time step
    for _ in range(max_length):
        # Generate next token's distribution using decoder layer
        decoder_output = transformer_decoder_layer(decoder_input, encoder_output)

        # Select the token with highest probability (greedy sampling)
        next_token = tf.argmax(decoder_output, axis=-1)
        
        # Append the token to the sequence
        sequence.append(next_token.numpy()[0, 0])

        # Update the decoder input for the next time step
        decoder_input = tf.concat([decoder_input, next_token], axis=-1)

    return sequence

# Example usage
start_token = tf.random.normal((1, 300))  # Start-of-sequence token

# Sample encoder output (replace with your actual GloVe vectors)
encoder_output = textEnc(inputEmbeddings)  # Batch size of 1, sequence length of max_length, GloVe dimension of 300

# Generate a sequence using autoregressive decoding
generated_sequence = generate_sequence(encoder_output, start_token)

print("Generated Sequence:", generated_sequence)

