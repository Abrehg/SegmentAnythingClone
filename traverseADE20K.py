# Traverse entirety of ADE20K to compile dataset

# Import packages
import os
import json
import tensorflow as tf
from formatImg import formatImageTensorFromPath
from formatImg import formatMaskTensorFromPath
from formatImg import formatImg
from TextEncoder import textEncoder
from formatText import formatText

textEnc = textEncoder()
textEnc.load_weights("./txt_encoder_weights.h5")

# Function
def findADE20KData(path):
    # Check if the path is a directory
    if not os.path.isdir(path):
        print(f"{path} is not a valid directory.")
        return

    # Get the list of all files and directories in the current path
    try:
        items = os.listdir(path)
    except PermissionError:
        print(f"Permission denied: {path}")
        return

    # Loop through each item
    for item in items:
        item_path = os.path.join(path, item)

        # If the item is a directory, recursively call find_json_file
        if os.path.isdir(item_path):
            findADE20KData(item_path)
        # If the item is a .json file, print the path and stop further traversal
        elif item.endswith('.json'):
            print(f"Found JSON file: {item_path}")
            try:
                with open(item_path, 'r') as json_file:
                    data = json.load(json_file)
                    print("JSON file content:")
                    for i in range(len(data["annotation"]["object"])):
                        name = data["annotation"]["object"][i]["name"]
                        temp = ""
                        index = 0
                        while index < len(name):
                            if name[index] != ",":
                                temp = temp + name[index]
                            else:
                                index = index + 1
                                imgPath = path + "/" + data["annotation"]["filename"]
                                imgTensor = formatImageTensorFromPath(imgPath)
                                img, MAEenc = formatImg(imgTensor)
                                img = tf.squeeze(img, axis=0)
                                print(f"Img shape: {tf.shape(img)}")
                                img = tf.RaggedTensor.from_tensor(img)
                                maskPath = path + "/" + data["annotation"]["object"][i]["instance_mask"][data["annotation"]["object"][i]["instance_mask"].find("/")+1:]
                                mask = formatMaskTensorFromPath(maskPath)
                                print(f"Mask shape: {tf.shape(mask)}")
                                mask = tf.RaggedTensor.from_tensor(mask)
                                textFormatting = formatText(temp)
                                textFormatting = tf.expand_dims(textFormatting, axis = 0)
                                txt = textEnc(textFormatting)
                                txt = tf.squeeze(txt, axis = 0)
                                print(f"Txt shape: {tf.shape(txt)}")
                                txt = tf.RaggedTensor.from_tensor(txt)
                                temp = ""
                                yield {'input_4': img, 'input_5': txt}, mask
                            index = index + 1
                        if len(temp) != 0:
                            imgPath = path + "/" + data["annotation"]["filename"]
                            imgTensor = formatImageTensorFromPath(imgPath)
                            img, MAEenc = formatImg(imgTensor)
                            img = tf.squeeze(img, axis = 0)
                            print(f"Img shape: {tf.shape(img)}")
                            img = tf.RaggedTensor.from_tensor(img)
                            maskPath = path + "/" + data["annotation"]["object"][i]["instance_mask"][data["annotation"]["object"][i]["instance_mask"].find("/")+1:]
                            mask = formatMaskTensorFromPath(maskPath)
                            print(f"Mask shape: {tf.shape(mask)}")
                            mask = tf.RaggedTensor.from_tensor(mask)
                            textFormatting = formatText(temp)
                            textFormatting = tf.expand_dims(textFormatting, axis = 0)
                            txt = textEnc(textFormatting)
                            txt = tf.squeeze(txt, axis = 0)
                            print(f"Txt shape: {tf.shape(txt)}")
                            txt = tf.RaggedTensor.from_tensor(txt)
                            temp = ""
                            yield {'input_6': img, 'input_7': txt}, mask
            except Exception as e:
                print(f"Error reading JSON file {item_path}: {e}")
            return
    
    return

# Test function
def ADE20K_generator():
    home_directory = os.path.expanduser("/Users/adityaasuratkal/Downloads/Img_Data/ADE20K")
    yield from findADE20KData(home_directory)

# Define output signatures for the generator
output_signature = (
    {
        'input_6': tf.RaggedTensorSpec(shape=(None, None, 1024), dtype=tf.float32),
        'input_7': tf.RaggedTensorSpec(shape=(None, 300), dtype=tf.float32)
    },
    tf.RaggedTensorSpec(shape=(None, None, 1), dtype=tf.int32)
)

# Create the dataset from the generator
ADE20K = tf.data.Dataset.from_generator(lambda: ADE20K_generator(), output_signature=output_signature)