import os
import pickle
import numpy as np
import cv2 as cv


def image_loader(path, size):
    # String path to image
    # Tuple size of output image
    image = cv.imread(path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.resize(image, size, cv.INTER_CUBIC)

    return image


def dataset_preprocessing(dataset_path, labels_file_path, size,
                          image_paths_pickle):
    # String path to dataset
    # String path to labels file
    # Tuple size of image
    # String name of pickle file where image paths are stored
    with open(labels_file_path, 'r') as f:
        classes = f.read().split('\n')[:-1]

    images = []
    labels = []
    image_paths = []

    for image_name in os.listdir(dataset_path):
        try:
            image_path = os.path.join(dataset_path, image_name)
            images.append(image_loader(image_path, size))
            image_paths.append(image_path)
            for idx in range(len(classes)):
                if classes[idx] in image_name:
                    labels.append(idx)
        except:
            pass

    with open("pickle_saves/" + image_paths_pickle + ".pickle", 'wb') as f:
        pickle.dump(image_paths, f)

    assert len(images) == len(labels)
    return np.array(images), np.array(labels)