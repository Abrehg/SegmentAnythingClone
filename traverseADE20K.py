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
                        #print(json.dumps(data["annotation"]["object"][i], indent=4))
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
                                maskPath = path + "/" + data["annotation"]["object"][i]["instance_mask"][data["annotation"]["object"][i]["instance_mask"].find("/")+1:]
                                mask = formatMaskTensorFromPath(maskPath)
                                print(f"Mask shape: {tf.shape(mask)}")
                                textFormatting = formatText(temp)
                                textFormatting = tf.expand_dims(textFormatting, axis = 0)
                                txt = textEnc(textFormatting)
                                txt = tf.squeeze(txt, axis = 0)
                                print(f"Txt shape: {tf.shape(txt)}")
                                temp = ""
                                #yield {'img_encodings': img, 'text_encodings': txt}, mask
                            index = index + 1
                        if len(temp) != 0:
                            imgPath = path + "/" + data["annotation"]["filename"]
                            imgTensor = formatImageTensorFromPath(imgPath)
                            img, MAEenc = formatImg(imgTensor)
                            img = tf.squeeze(img, axis = 0)
                            print(f"Img shape: {tf.shape(img)}")
                            maskPath = path + "/" + data["annotation"]["object"][i]["instance_mask"][data["annotation"]["object"][i]["instance_mask"].find("/")+1:]
                            mask = formatMaskTensorFromPath(maskPath)
                            print(f"Mask shape: {tf.shape(mask)}")
                            textFormatting = formatText(temp)
                            textFormatting = tf.expand_dims(textFormatting, axis = 0)
                            txt = textEnc(textFormatting)
                            txt = tf.squeeze(txt, axis = 0)
                            print(f"Txt shape: {tf.shape(txt)}")
                            temp = ""
                            #yield {'img_encodings': img, 'text_encodings': txt}, mask
            except Exception as e:
                print(f"Error reading JSON file {item_path}: {e}")
            return
    
    return

# findADE20KData(os.path.expanduser("/Users/adityaasuratkal/Downloads/Img_Data/ADE20K"))

# Test function
def ADE20K_generator():
    home_directory = os.path.expanduser("/Users/adityaasuratkal/Downloads/Img_Data/ADE20K")
    yield from findADE20KData(home_directory)

# Define output signatures for the generator
output_signature = (
    {
        'img_encodings': tf.TensorSpec(shape=(None, None, 1024), dtype=tf.float32),
        'text_encodings': tf.TensorSpec(shape=(None, 300), dtype=tf.float32)
    },
    tf.TensorSpec(shape=(None, None, 1), dtype=tf.int32)
)

# Create the dataset from the generator
ADE20K = tf.data.Dataset.from_generator(lambda: ADE20K_generator(), output_signature=output_signature)

# # Old Version
# def findADE20KData(path, img, txt, mask):
#     # Check if the path is a directory
#     if not os.path.isdir(path):
#         print(f"{path} is not a valid directory.")
#         return img, txt, mask

#     # Get the list of all files and directories in the current path
#     try:
#         items = os.listdir(path)
#     except PermissionError:
#         print(f"Permission denied: {path}")
#         return img, txt, mask

#     # Loop through each item
#     for item in items:
#         item_path = os.path.join(path, item)

#         # If the item is a directory, recursively call find_json_file
#         if os.path.isdir(item_path) and item_path != "/Users/adityaasuratkal/Downloads/Img_Data/ADE20K/ADE20K_2021_17_01":
#             img, txt, mask = findADE20KData(item_path, img, txt, mask)
#         # If the item is a .json file, print the path and stop further traversal
#         elif item.endswith('.json'):
#             print(f"Found JSON file: {item_path}")
#             try:
#                 with open(item_path, 'r') as json_file:
#                     data = json.load(json_file)
#                     print("JSON file content:")
#                     for i in range(len(data["annotation"]["object"])):
#                         #print(json.dumps(data["annotation"]["object"][i], indent=4))
#                         name = data["annotation"]["object"][i]["name"]
#                         temp = ""
#                         index = 0
#                         while index < len(name):
#                             if name[index] != ",":
#                                 temp = temp + name[index]
#                             else:
#                                 index = index + 1
#                                 imgPath = path + "/" + data["annotation"]["filename"]
#                                 print(imgPath)
#                                 print("Starting formatting image from data path")
#                                 imgTensor = formatImageTensorFromPath(imgPath)
#                                 print("Converting image into embedding")
#                                 imgEnc, MAEEnc = formatImg(imgTensor)
#                                 img.append(imgEnc)
#                                 maskPath = path + "/" + data["annotation"]["object"][i]["instance_mask"][data["annotation"]["object"][i]["instance_mask"].find("/")+1:]
#                                 print(maskPath)
#                                 print("Starting formatting mask from data path")
#                                 mask.append(formatMaskTensorFromPath(maskPath))
#                                 print(temp)
#                                 print("Starting formatting text to GloVe encodings")
#                                 textFormatting = formatText(temp)
#                                 print("Converting GloVe encodings into proper embeddings")
#                                 textFormatting = tf.expand_dims(textFormatting, axis = 0)
#                                 print(f"Text to GloVe shape: {tf.shape(textFormatting)}")
#                                 print(f"GloVe to Encoding shape: {tf.shape(textEnc(textFormatting))}")
#                                 txt.append(textEnc(textFormatting))
#                                 temp = ""
#                             index = index + 1
#                         if len(temp) != 0:
#                             imgPath = path + "/" + data["annotation"]["filename"]
#                             print(imgPath)
#                             print("Starting formatting image from data path")
#                             imgTensor = formatImageTensorFromPath(imgPath)
#                             print("Converting image into embedding")
#                             imgEnc, MAEEnc = formatImg(imgTensor)
#                             img.append(imgEnc)
#                             maskPath = path + "/" + data["annotation"]["object"][i]["instance_mask"][data["annotation"]["object"][i]["instance_mask"].find("/")+1:]
#                             print(maskPath)
#                             print("Starting formatting mask from data path")
#                             mask.append(formatMaskTensorFromPath(maskPath))
#                             print(temp)
#                             print("Starting formatting text to GloVe encodings")
#                             textFormatting = formatText(temp)
#                             print("Converting GloVe encodings into proper embeddings")
#                             textFormatting = tf.expand_dims(textFormatting, axis = 0)
#                             print(f"Text to GloVe shape: {tf.shape(textFormatting)}")
#                             print(f"GloVe to Encoding shape: {tf.shape(textEnc(textFormatting))}")
#                             txt.append(textEnc(textFormatting))
#                             temp = ""
#             except Exception as e:
#                 print(f"Error reading JSON file {item_path}: {e}")
#             return img, txt, mask
    
#     return img, txt, mask

# home_directory = os.path.expanduser("/Users/adityaasuratkal/Downloads/Img_Data/ADE20K")
# imgEncs = []
# txtEncs = []
# maskEncs = []
# imgEncs, txtEncs, maskEncs = findADE20KData(home_directory, imgEncs, txtEncs, maskEncs)
