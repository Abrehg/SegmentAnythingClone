import tensorflow as tf

def EncGen(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(4, (2, 2), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(16, (2, 2), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(2048, activation='relu'))
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    return model