import tensorflow as tf
from tensorflow import keras as keras
from keras import layers as tfl
from formatText import formatText
from formatText import findWord
from TextEncoder import textEncoder
import tensorflow_datasets as tfds
from keras import backend as K

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

#positional encoding function
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

#Single decoder layer (typical layer used for chat GPT, BERT, and more)
def transformer_decoder_layer(inputs, enc_output):
    pos_encodings = positional_encoding(inputs)
    vec = inputs + pos_encodings
    X = tfl.MultiHeadAttention(num_heads=16, key_dim=300, dropout=0.3)(vec, vec)
    print(tf.shape(X))
    print(tf.shape(vec))
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

def beam_search_generate_sequence(encoder_output, start_token):
    beam_size = 5
    max_length = 2000
    sequences = [start_token]

    for _ in range(max_length):
        all_candidates = []
        for seq in sequences:
            if seq[-1] is eos_token:  # Check if the last token is EOS token
                all_candidates.append(seq)
                continue
            sequence_tensor = tf.expand_dims(seq, axis=0)
            decoder_output = transformer_decoder_layer(sequence_tensor, encoder_output)
            candidate_token = tfl.GlobalAveragePooling1D()(decoder_output)
            candidate_seq = tf.concat([seq, candidate_token], axis=0)
            all_candidates.append((candidate_seq, tf.reduce_sum(candidate_token)))
        
        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        sequences = [ordered[i][0] for i in range(min(beam_size, len(ordered)))]

    return sequences[0][:-1]

def cosine_similarity_loss(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    cosine_sim = K.sum(y_true * y_pred, axis=-1)
    similarity = 1 - cosine_sim
    return similarity

def custom_loss(y_true, y_pred):
    similarity_loss = cosine_similarity_loss(y_true, y_pred)
    
    max_length = 2000
    sequence_length = K.shape(y_pred)[1]
    length_penalty = K.maximum(0.0, sequence_length / max_length - 1.0)
    
    total_loss = similarity_loss + 0.25 * length_penalty  # Adjust the weight of the length penalty
    
    return total_loss

# input tensor
inputEmbeddings = keras.Input((None, 300))

#generate encoder output (works)
textEnc = textEncoder()
encoder_output = textEnc(inputEmbeddings)

# condense encoder output into a start token
X = tfl.GlobalAveragePooling1D()(encoder_output)
start_token = tfl.Dense(300, 'relu')(X)

# Generate a sequence using transformer decoder
generated_sequence = beam_search_generate_sequence(encoder_output, start_token)

print("Generated Sequence:", generated_sequence)

# Create the combined model
combined_model = keras.Model(inputs=inputEmbeddings, outputs=generated_sequence)

combined_model.compile(optimizer='adam', loss=custom_loss)

# Print the model summary
combined_model.summary()

#text summarization training
dataset_name = 'multi_news'
(train_data, val_data, test_data), info = tfds.load(dataset_name, split=['train', 'validation', 'test'], shuffle_files=True, with_info=True)
print(info)

def process_dataset(data, key):
    vectors = [formatText(entry[key].numpy().decode("utf-8")) for entry in data]
    return vectors

print("processing started")

x_train = process_dataset(train_data, 'document')
y_train = process_dataset(train_data, 'summary')
print("train processed")

x_val = process_dataset(val_data, 'document')
y_val = process_dataset(val_data, 'summary')
print("val processed")

x_test = process_dataset(test_data, 'document')
y_test = process_dataset(test_data, 'summary')
print("test processed")

batch_size = 512
num_epochs = 10

#train dataset
combined_model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs)
print("train fitted")

combined_model.fit(x=x_val, y=y_val, batch_size=batch_size, epochs=num_epochs)
print("val fitted")

combined_model.test_on_batch(x=x_test, y=y_test)
print("test set implemented")

#save desired weights
textEnc.save_weights('./text_encoder_weights.h5')
print("process finished")