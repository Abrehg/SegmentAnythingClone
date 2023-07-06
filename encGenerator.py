import tensorflow as tf
import matplotlib.pyplot as plt
import optuna
from tensorflow import keras as keras
from keras import layers as tfl
import numpy as np
import tensorflow_datasets as tfds

Inputs = keras.Input(shape=(16,16,3))
X = tfl.Conv2D(4, (2,2))(Inputs)
X = tfl.Flatten()(X)
X = tfl.Dense(512, 'relu')(X)
Y = tfl.Dense(1024, 'relu')(X)
EncGenerator = keras.Model(inputs = Inputs, outputs = Y)

In = keras.Input(shape=(32, 32, 3))
X = tfl.Conv2D(3, (2,2))(In)
X = EncGenerator.predict(X)
Out = tfl.Dense(100,activation='softmax')(X)
FullModel = keras.Model(inputs=In, outputs=Out)
print(FullModel.summary())

FullModel.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics = ['accuracy'])

(ds_train, ds_test) = tfds.load('cifar100', split=['train','test'], shuffle_files=True)

FullModel.fit(ds_train, epochs=20, validation_data=ds_test,)