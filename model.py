import tensorflow as tf
import optuna
from tensorflow import keras as keras
from formatImg import formatImg
from formatImg import formatTensorFromPath
from ImgEncoder import imgEncoder
from formatText import formatText
from TextEncoder import textEncoder
from Decoder import decoder



#input encoding of image (Transformer encoder + MAE)
imgEncodings_input = keras.Input((None, None, 1024), name="img_encodings")
imgEnc = imgEncoder()
X = imgEnc(imgEncodings_input)

#input encoding of text (CLIP)
textEncodings_input = keras.Input((None, 300), name="text_encodings")
textEnc = textEncoder()
Q = textEnc(textEncodings_input)

#decode and return output mask
masks = decoder(Q, X)

model = keras.Model(inputs = [imgEncodings_input, textEncodings_input], outputs = masks)

imgFilePath2 = "/Users/adityaasuratkal/Downloads/Img_Data/ADE20K/ADE_frame_00000007/X/ADE_frame_00000007.jpg"
text = "Hello World"
img = formatTensorFromPath(imgFilePath2)
print(tf.shape(img))
fullImgEncodings, MAEencodings = formatImg(img)

textEncodings = formatText(text)

out = model.predict([fullImgEncodings, textEncodings])
print(tf.shape(out))

mask = tf.io.read_file("/Users/adityaasuratkal/Downloads/Img_Data/ADE20K/ADE_frame_00000007/Y/instance_018_ADE_frame_00000007.png")
correct = tf.io.decode_image(mask, channels=1, dtype=tf.dtypes.uint8)

def custom_loss(y_true, y_pred):
    epsilon = 0.
    epsilonTensor = tf.cast(epsilon, y_true.dtype)
    conditionYTrue = tf.math.greater(y_true, epsilonTensor)
    conditionYHat = tf.math.greater(y_pred, epsilonTensor)

    Intersection = tf.logical_and(conditionYTrue, conditionYHat)
    Union = tf.logical_or(conditionYTrue, conditionYHat)

    intersecCount = tf.reduce_sum(tf.cast(Intersection, tf.int32))
    unionCount = tf.reduce_sum(tf.cast(Union, tf.int32))

    IoU = intersecCount/unionCount

    loss = 1-IoU

    return loss

print(custom_loss(correct, out[0]))

model.compile(optimizer= 'adam', loss=custom_loss)

print("model compiled")