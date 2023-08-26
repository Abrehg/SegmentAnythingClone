import tensorflow as tf
import os
import h5py
import numpy as np
import gc
from formatImg import formatImg
from formatImg import formatTensorFromPath
import multiprocessing

#image_folder -> path in string form to imagenet 2010 dataset
#dump_folder -> path in string form to target for pickle files
def main(image_folder, dump_folder):
    # Load and process train set, save to disk in batches
    load_and_process_images(image_folder, 'train', dump_folder, batch_size=1024)

    # Load and process validation set, save to disk in batches
    load_and_process_images(image_folder, 'val', dump_folder, batch_size=1024)

    # Load and process test set, save to disk in batches
    #load_and_process_images(image_folder, 'test', dump_folder, batch_size=1024)
    
    #highly RAM intensive, only uncomment if your machine can handle it
"""
    # Compile train data from batches into a single pickle file
    compile_data_from_batches(dump_folder, 'train')

    # Compile validation data from batches into a single pickle file
    compile_data_from_batches(dump_folder, 'val')

    # Compile test data from batches into a single pickle file
    compile_data_from_batches(dump_folder, 'test')
"""

def load_and_process_images(dir_path, data_folder, pickleFolder, batch_size):
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    image_files = []
    batch_idx = 0
    image_num = 0
    image_folder = os.path.join(dir_path, data_folder)
    for folder_name in os.listdir(image_folder):
        class_folder = os.path.join(image_folder, folder_name)
        if os.path.isdir(class_folder):
            for file_name in os.listdir(class_folder):
                image_num = image_num + 1
                image_path = os.path.join(class_folder, file_name)
                image_files.append(image_path)
                if image_num % batch_size == 0:
                    batch_data = pool.map(process_image, image_files)
                    embeddings, labels = zip(*batch_data)
                    save_data_to_disk(embeddings, labels, data_folder, batch_idx, pickleFolder)
                    image_files = []
                    batch_idx = batch_idx + 1
                    batch_data = None
                    embeddings = None
                    labels = None
                    gc.collect()
    
    if len(image_files) != 0:
        batch_data = pool.map(process_image, image_files)
        embeddings, labels = zip(*batch_data)
        save_data_to_disk(embeddings, labels, data_folder, batch_idx, pickleFolder)
        batch_data = None
        embeddings = None
        labels = None
        gc.collect()

def process_image(image_path):
    path = image_path.split("/")
    for i in range(0, len(path)):
        if path[i] == "ImageNet":
            imgFolder = image_path.split("/")[(i+1)]
            mainFolder = image_path.split("/")[:(i+1)]
            break
    mainFolderPath = ""
    for j in range(0, len(mainFolder)):
        mainFolderPath = mainFolderPath + mainFolder[j]
    data_folder = os.path.join(mainFolderPath, 'data')
    if imgFolder == 'train':
        finalEmbeddings, other, other = formatImg(formatTensorFromPath(image_path))
        raw_label = os.path.basename(image_path).split('.')[0][1:]
        i = raw_label.find("_")
        class_label = int(raw_label[(i+1):])
        one_hot_label = tf.one_hot(class_label, depth=1000)
    elif imgFolder == 'val':
        labelPath = os.path.join(data_folder, 'ILSVRC2010_validation_ground_truth.txt')
        finalEmbeddings, other, other = formatImg(formatTensorFromPath(image_path))
        imgIndex = int(os.path.basename(path).split('.')[0].split("_")[-1])
        label_list = read_file_as_list(labelPath)
        class_label = label_list[imgIndex]
        one_hot_label = tf.one_hot(class_label, depth=1000)
    elif imgFolder == 'test':
        labelPath = os.path.join(data_folder, 'ILSVRC2010_testing_ground_truth.txt')
        finalEmbeddings, other, other = formatImg(formatTensorFromPath(image_path))
        imgIndex = int(os.path.basename(path).split('.')[0].split("_")[-1])
        label_list = read_file_as_list(labelPath)
        class_label = label_list[imgIndex]
        one_hot_label = tf.one_hot(class_label, depth=1000)
    return finalEmbeddings, one_hot_label

def save_data_to_disk(embeddings, labels, dataset, batchNum, folder):
    filename = f"{dataset}_batch{batchNum}.h5"
    hdf5_file_path = os.path.join(folder, filename)

    with h5py.File(hdf5_file_path, "w") as file:
        # Save each embedding and label separately for each image
        for idx, (emb, lab) in enumerate(zip(embeddings, labels)):
            emb_name = f"embedding_{idx}"
            lab_name = f"label_{idx}"

            file.create_dataset(emb_name, data=emb.numpy())
            file.create_dataset(lab_name, data=lab.numpy())

    print(f"{filename} has been saved")

def compile_data_from_batches(data_folder, dataset_name):
    compiled_data = []
    batch_idx = 0

    while True:
        batch_filename = os.path.join(data_folder, f"{dataset_name}_batch{batch_idx}.h5")
        if os.path.exists(batch_filename):
            with h5py.File(batch_filename, "r") as file:
                batch_data = list(zip(file["embeddings"], file["labels"]))
            compiled_data.extend(batch_data)
            #possibly remove
            os.remove(batch_filename)
            batch_idx += 1
        else:
            break

    compiled_filename = os.path.join(data_folder, f"{dataset_name}_compiled.h5")
    with h5py.File(compiled_filename, "w") as file:
        embeddings = file.create_dataset("embeddings", data=[item[0] for item in compiled_data])
        labels = file.create_dataset("labels", data=[item[1] for item in compiled_data])

def read_file_as_list(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            int_list = [int(line.strip()) for line in lines]
        return int_list
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []