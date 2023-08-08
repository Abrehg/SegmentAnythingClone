import tensorflow as tf
from tensorflow import keras as keras
from keras import layers as tfl
from formatText import formatText
from formatText import findWord
from CLIPencoder import textEncoder
import tensorflow_datasets as tfds

"""
textInput = 'Hello World this is a test'

embeddings = formatText(textInput)
word = findWord(embeddings)
print(word)
"""

NNLayers = 7
unitsMid = 1000
unitsOut = 1

eos_token = tf.zeros((1, 300))

def positional_encoding(inputs):
    seq_len, d_model = tf.shape(inputs)[1], inputs.shape[-1]
    position = tf.range(start=0, limit=seq_len, delta=1, dtype=tf.float32)
    position = tf.expand_dims(position, axis=0)
    div_term = tf.pow(10000.0, 2 * tf.range(d_model // 2, dtype=tf.float32) / d_model)
    div_term = tf.expand_dims(div_term, axis=0)
    angles = position / div_term

    angle_rads = tf.concat([tf.sin(angles), tf.cos(angles)], axis=-1)
    angle_rads = tf.expand_dims(angle_rads, axis=0)
    return angle_rads

def transformer_decoder_layer(inputs, enc_output):
    pos_encodings = positional_encoding(inputs)
    vec = inputs + pos_encodings
    X = tfl.MultiHeadAttention(num_heads=16, key_dim=300, dropout=0.3)(vec, vec)
    add1 = tfl.Add()([X, vec])
    norm1 = tfl.LayerNormalization()(add1)
    enc = tfl.MultiHeadAttention(num_heads=16, key_dim=300, dropout=0.3)(enc_output, enc_output, norm1)
    add2 = tfl.Add()([enc, norm1])
    norm2 = tfl.LayerNormalization()(add2)
    ffn1 = tfl.Dense(512, activation='relu')(norm2)
    ffn2 = tfl.Dense(300, activation='relu')(ffn1)
    add3 = tfl.Add()([ffn2, norm2])
    norm3 = tfl.LayerNormalization()(add3)
    outputs = tfl.Dense(300, 'linear')(norm3)
    return outputs

def generate_sequence(encoder_output, start_token):
    sequence = [start_token]
    max_length = 2

    for i in range(max_length):
        sequence_tensor = tf.expand_dims(tf.concat(sequence, axis=0), axis=0)
        decoder_output = transformer_decoder_layer(sequence_tensor, encoder_output)
        next_token = tfl.GlobalAveragePooling1D()(decoder_output)

        # Append the token to the sequence
        sequence.append(next_token)

    sequence = sequence[1:]
    sequence = tf.stack(sequence, axis=1)
      # Remove the start_token

    return sequence

# encoder output
inputEmbeddings = keras.Input((None, 300))
textEnc = textEncoder()
encoder_output = textEnc(inputEmbeddings)
X = tfl.GlobalAveragePooling1D()(encoder_output)
start_token = tfl.Dense(300, 'relu')(X)

# Generate a sequence using autoregressive decoding
generated_sequence = generate_sequence(encoder_output, start_token)

print("Generated Sequence:", generated_sequence)

# Create the combined model
combined_model = keras.Model(inputs=inputEmbeddings, outputs=generated_sequence)

combined_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
combined_model.summary()

"""
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