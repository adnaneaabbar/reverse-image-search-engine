import os
import pickle
import numpy as np
import cv2 as cv
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import config as cfg

from utils.utils import *
from utils.dataset import image_loader
from model import ImageSearchModel


def simple_inference(
        model,
        session,
        train_set_vectors,
        uploaded_image_path,  # string, path to the uploaded image
        image_size,
        distance='hamming'):

    image = image_loader(uploaded_image_path, image_size)
    channels = cv.split(image)
    features = []

    for chan in channels:
        hist = cv.calcHist([chan], [0], None, [256], [0, 256])
        features.append(hist)

    color_features = np.vstack(features).T

    feed_dict = {model.inputs: [image], model.dropout_rate: 0.0}

    dense_2_features, dense_4_features = session.run(
        [model.dense_2_features, model.dense_4_features], feed_dict=feed_dict)

    closest_ids = None
    if distance == 'hamming':
        dense_2_features = np.where(dense_2_features < 0.5, 0, 1)
        dense_4_features = np.where(dense_4_features < 0.5, 0, 1)

        uploaded_image_vector = np.hstack((dense_2_features, dense_4_features))

        closest_ids = hamming_distance(train_set_vectors,
                                       uploaded_image_vector)

    elif distance == 'cosine':
        uploaded_image_vector = np.hstack((dense_2_features, dense_4_features))

        closest_ids = cosine_distance(train_set_vectors, uploaded_image_vector)

    return closest_ids


def simple_inference_with_color_filters(model,
                                        session,
                                        train_set_vectors,
                                        uploaded_image_path,
                                        color_vectors,
                                        image_size,
                                        distance='hamming'):

    image = image_loader(uploaded_image_path, image_size)

    ####################################################
    ## Calculating color histogram of the query image ##
    channels = cv.split(image)
    features = []
    for chan in channels:
        hist = cv.calcHist([chan], [0], None, [256], [0, 256])
        features.append(hist)

    color_features = np.vstack(features).T
    ####################################################

    feed_dict = {model.inputs: [image], model.dropout_rate: 0.0}

    dense_2_features, dense_4_features = session.run(
        [model.dense_2_features, model.dense_4_features], feed_dict=feed_dict)

    closest_ids = None
    if distance == 'hamming':
        dense_2_features = np.where(dense_2_features < 0.5, 0, 1)
        dense_4_features = np.where(dense_4_features < 0.5, 0, 1)

        uploaded_image_vector = np.hstack((dense_2_features, dense_4_features))

        closest_ids = hamming_distance(train_set_vectors,
                                       uploaded_image_vector)

        # Comparing color features between query image and closest images selected by the model
        closest_ids = compare_color(
            np.array(color_vectors)[closest_ids], color_features, closest_ids)

    elif distance == 'cosine':
        uploaded_image_vector = np.hstack((dense_2_features, dense_4_features))

        closest_ids = cosine_distance(train_set_vectors, uploaded_image_vector)

        # Comparing color features between query image and closest images selected by the model
        closest_ids = compare_color(
            np.array(color_vectors)[closest_ids], color_features, closest_ids)

    return closest_ids