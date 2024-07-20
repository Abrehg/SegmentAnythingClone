import tensorflow as tf
from keras import layers as tfl
import keras

class CustomDecoderLayer(tfl.Layer):
    def __init__(self):
        super(CustomDecoderLayer, self).__init__()
        self.attention1 = tfl.MultiHeadAttention(num_heads=18, key_dim=1024, dropout=0.3)
        self.dense = tfl.Dense(1024, activation='relu')
        self.layer_norm = tfl.LayerNormalization()
        self.attention2 = tfl.MultiHeadAttention(num_heads=20, key_dim=1024, dropout=0.3)
        self.conv1 = tfl.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', activation='relu')
        self.conv2 = tfl.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', activation='relu')
        self.conv3 = tfl.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu')
        self.output_conv = tfl.Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', activation='sigmoid')

    def call(self, inputs):
        text, image = inputs

        if isinstance(text, tf.RaggedTensor):
            text = text.to_tensor()
        if isinstance(image, tf.RaggedTensor):
            image = image.to_tensor()

        image = self.attention1(image, image)
        text = self.dense(text)

        text = tf.expand_dims(text, axis=0)
        combined = tf.matmul(image, text, transpose_b=True)
        combined_features = tf.matmul(combined, text)

        X = self.layer_norm(combined_features)
        X = self.attention2(X, X)

        X = tf.expand_dims(X, axis=0)

        X_upsampled = self.conv1(X)
        X_upsampled = self.conv2(X_upsampled)
        X_upsampled = self.conv3(X_upsampled)
        output = self.output_conv(X_upsampled)

        output = tf.squeeze(output, axis=0)

        return output

# def process_single_instance(text, image):
#     image = tfl.MultiHeadAttention(num_heads=18, key_dim=1024, dropout=0.3)(image, image)
#     text = tfl.Dense(1024, activation='relu')(text)
    
#     combined = tf.matmul(image, text, transpose_b=True)
#     combined_features = tf.matmul(combined, text)
#     X = tfl.LayerNormalization()(combined_features)
#     X = tfl.MultiHeadAttention(num_heads=20, key_dim=1024, dropout=0.3)(X, X)
    
#     X_upsampled = tfl.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', activation='relu')(X)
#     X_upsampled = tfl.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', activation='relu')(X_upsampled)
#     X_upsampled = tfl.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(X_upsampled)
#     output = tfl.Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', activation='sigmoid')(X_upsampled)
    
#     return output

# def decoder(inputs):
#     text, image = inputs
#     text = tf.expand_dims(text, axis=0)
#     image = tf.expand_dims(image, axis=0)
#     return process_single_instance(text, image)

# decoderTextIn = keras.Input([None, 300])
# decoderImgIn = keras.Input([None, None, 1024])

# decoderOut = decoder(decoderTextIn, decoderImgIn)

# model = keras.Model(inputs = [decoderTextIn, decoderImgIn], outputs = decoderOut)

# print(model.summary())