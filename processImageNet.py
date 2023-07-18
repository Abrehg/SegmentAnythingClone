import tensorflow as tf
import scipy.io as sio
import os
from inputFormatting import formatImg
from inputFormatting import formatTensorFromPath
import multiprocessing

def process_image(image_path):
    finalEmbeddings, other = formatImg(formatTensorFromPath(image_path))
    class_label = int(os.path.basename(image_path).split('.')[0][1:])
    one_hot_label = tf.one_hot(class_label, depth=1000)
    return finalEmbeddings, one_hot_label

def load_and_process_images(image_files):
    results = []
    for image_path in image_files:
        result = process_image(image_path)
        results.append(result)

    final_embeddings, one_hot_labels = zip(*results)
    dataset = tf.data.Dataset.from_tensor_slices((final_embeddings, one_hot_labels))
    return dataset

def main():
    image_folder = "/Users/adityaasuratkal/Downloads/ImageNet"
    data_folder = os.path.join(image_folder, 'data')
    train_folder = os.path.join(image_folder, 'train')
    val_folder = os.path.join(image_folder, 'val')
    test_folder = os.path.join(image_folder, 'test')

    # Load the class labels
    meta_data = sio.loadmat(os.path.join(data_folder, 'meta.mat'))
    class_labels = meta_data['synsets']

    image_files = []
    class_labels = []
    for folder_name in os.listdir(train_folder):
        class_folder = os.path.join(train_folder, folder_name)
        if os.path.isdir(class_folder):
            class_label = int(folder_name[1:])
            for file_name in os.listdir(class_folder):
                image_path = os.path.join(class_folder, file_name)
                image_files.append(image_path)
                class_labels.append(class_label)

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        train_results = pool.map(process_image, image_files)

    final_embeddings, one_hot_labels = zip(*train_results)
    train_dataset = tf.data.Dataset.from_tensor_slices((final_embeddings, one_hot_labels))

    image_files = []
    class_labels = []
    for folder_name in os.listdir(val_folder):
        class_folder = os.path.join(val_folder, folder_name)
        if os.path.isdir(class_folder):
            class_label = int(folder_name[1:])
            for file_name in os.listdir(class_folder):
                image_path = os.path.join(class_folder, file_name)
                image_files.append(image_path)
                class_labels.append(class_label)

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        val_results = pool.map(process_image, image_files)

    final_embeddings, one_hot_labels = zip(*val_results)
    val_dataset = tf.data.Dataset.from_tensor_slices((final_embeddings, one_hot_labels))

    image_files = []
    class_labels = []
    for folder_name in os.listdir(test_folder):
        class_folder = os.path.join(test_folder, folder_name)
        if os.path.isdir(class_folder):
            class_label = int(folder_name[1:])
            for file_name in os.listdir(class_folder):
                image_path = os.path.join(class_folder, file_name)
                image_files.append(image_path)
                class_labels.append(class_label)

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        test_results = pool.map(process_image, image_files)

    final_embeddings, one_hot_labels = zip(*test_results)
    test_dataset = tf.data.Dataset.from_tensor_slices((final_embeddings, one_hot_labels))

    batch_size = 128

    # Configure dataset prefetching and batching
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE).batch(batch_size)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE).batch(batch_size)
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE).batch(batch_size)

    return train_dataset, val_dataset, test_dataset

# Call the `main` function to execute the code
train_dataset, val_dataset, test_dataset = main()
