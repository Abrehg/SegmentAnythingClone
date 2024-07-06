#Defines complete model instance

import tensorflow as tf
import keras as keras
from formatImg import formatImg
from formatImg import formatImageTensorFromPath
from formatImg import formatMaskTensorFromPath
from ImgEncoder import imgEncoder
from formatText import formatText
from TextEncoder import textEncoder
from Decoder import decoder
from traverseADE20K import ADE20K

# Initialize text encoder for compiling text input
textEnc = textEncoder()
textEnc.load_weights("./txt_encoder_weights.h5")

#input encoding of image (Transformer encoder + MAE)
imgEncodings_input = keras.Input((None, None, 1024), name="img_encodings", ragged = True)
imgEnc = imgEncoder()
X = imgEnc(imgEncodings_input)

#input encoding of text (CLIP)
Q = keras.Input((None, 300), name="text_encodings", ragged = True)

#decode and return output mask
masks = decoder(Q, X)

model = keras.Model(inputs = [imgEncodings_input, Q], outputs = masks)

#Custom loss function for output mask (Intersection over Union)
def custom_loss(y_true, y_pred):
    epsilon = 0.3
    epsilonTensorTrue = tf.cast(epsilon, y_true.dtype)
    epsilonTensorPred = tf.cast(epsilon, y_pred.dtype)

    print(f"y_true shape: {tf.shape(y_true)}")
    print(f"y_pred shape: {tf.shape(y_pred)}")

    batchSizeTrue, heightTrue, widthTrue, finalDimTrue = y_true.shape
    batchSizePred, heightPred, widthPred, finalDimPred = y_pred.shape

    print(f"Y True shape: {batchSizeTrue}, {heightTrue}, {widthTrue}, {finalDimTrue}")
    print(f"Y Pred shape: {batchSizePred}, {heightPred}, {widthPred}, {finalDimPred}")

    y_pred = tf.image.crop_to_bounding_box(y_pred, (heightPred-heightTrue)//2, (widthPred-widthTrue)//2, heightTrue, widthTrue)
    conditionYTrue = tf.math.greater(y_true, epsilonTensorTrue)
    conditionYHat = tf.math.greater(y_pred, epsilonTensorPred)

    Intersection = tf.logical_and(conditionYTrue, conditionYHat)
    Union = tf.logical_or(conditionYTrue, conditionYHat)

    intersecCount = tf.reduce_sum(tf.cast(Intersection, tf.int32))
    unionCount = tf.reduce_sum(tf.cast(Union, tf.int32))

    IoU = intersecCount/unionCount

    loss = 1-IoU

    return loss

#Compile model using ADAM and custom loss defined above 
model.compile(optimizer= 'adam', loss=custom_loss, metrics=['accuracy'])

print("model compiled")

#Find and compile ADE20K data
batch_size = 64
EPOCHS = 10
ADE20K = ADE20K.batch(batch_size)
model.fit(ADE20K, epochs=EPOCHS)

# Create function that predicts a single input (image url, string)

def predictFromInput(imgUrl, textInput):
    img = formatImageTensorFromPath(imgUrl)
    print(f"original image shape: {tf.shape(img)}")
    fullImgEncodings, MAEencodings = formatImg(img)

    textEncodings = formatText(textInput)
    textEncodings = textEnc(textEncodings)

    out = model.predict([fullImgEncodings, textEncodings])
    return out

# Real world example
# imgFilePath2 = "/Users/adityaasuratkal/Downloads/Img_Data/ADE20K/ADE_frame_00000007/X/ADE_frame_00000007.jpg"
# text = "Hello World"

# out = predictFromInput(imgFilePath2, text)

# mask = tf.io.read_file("/Users/adityaasuratkal/Downloads/Img_Data/ADE20K/ADE_frame_00000007/Y/instance_018_ADE_frame_00000007.png")
# correct = tf.io.decode_image(mask, channels=1, dtype=tf.dtypes.uint8)

# print(f"model output shape: {tf.shape(out[0])}")
# print(f"mask shape: {tf.shape(correct)}")

# print(custom_loss(correct, out[0]))