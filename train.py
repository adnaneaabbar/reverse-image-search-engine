<<<<<<< HEAD
import os
import pickle
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import config as cfg
from model import ImageSearchNetwork


def train(model,
          epochs,
          drop_rate,
          batch_size,
          data,
          save_dir,
          saver_delta=0.15):

    x_train, y_train, x_test, y_test = data

    # start session
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    # define saver
    saver = tf.train.Saver()

    best_test_accuracy = 0.0
    # start training loop
    for epoch in range(epochs):

        train_accuracy = []
        train_loss = []

        for j in range(len(x_train) // batch_size):
            start_id = j * batch_size
            end_id = start_id + batch_size

            x_batch = x_train[start_id:end_id]
            y_batch = y_train[start_id:end_id]

            feed_dict = {
                model.inputs: x_batch,
                model.targets: y_batch,
                model.dropout_rate: drop_rate
            }

            _, t_loss, preds_tr = session.run(
                [model.optimizer, model.loss, model.predictions],
                feed_dict=feed_dict)

            train_accuracy.append(sparse_accuracy(y_batch, preds_tr))
            train_loss.append(t_loss)

        print("Epoch: {}/{}".format(epoch, epochs),
              " | Training accuracy: {}".format(np.mean(train_accuracy)),
              " | Training loss: {}".format(np.mean(train_loss)))

        test_accuracy = []

        for j in range(len(x_test) // batch_size):
            start_id = j * batch_size
            end_id = start_id + batch_size

            x_batch = x_test[start_id:end_id]
            y_batch = y_test[start_id:end_id]

            feed_dict = {model.inputs: x_batch, model.dropout_rate: 0.0}

            preds_test = session.run(model.predictions, feed_dict=feed_dict)
            test_accuracy.append(sparse_accuracy(y_batch, preds_test))

        print("Test accuracy: {}".format(np.mean(test_accuracy)))

        # saving the model
        if np.mean(train_accuracy) > np.mean(
                test_accuracy):  # to prevent underfitting
            if np.abs(np.mean(train_accuracy) - np.mean(test_accuracy)
                      ) <= saver_delta:  # to prevent overfitting
                if np.mean(test_accuracy) >= best_test_accuracy:
                    best_test_accuracy = np.mean(test_accuracy)
                    saver.save(
                        session,
                        "{}/model_epoch_{}.ckpt".format(save_dir, epoch))

    session.close()


def create_training_set_vectors(
        model,
        x_train,
        y_train,
        batch_size,
        checkpoint_path,  # string path to the model checkpoint
        image_size,
        distance='hamming'):
    # Define session
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    # Restore session
    saver = tf.train.Saver()
    saver.restore(session, checkpoint_path)

    dense_2_features = []
    dense_4_features = []

    #iterate through training set
    for j in range(len(x_train) // batch_size):
        start_id = j * batch_size
        end_id = start_id + batch_size

        x_batch = x_train[start_id:end_id]

        feed_dict = {model.inputs: x_batch, model.dropout_rate: 0.0}

        dense_2, dense_4 = session.run(
            [model.dense_2_features, model.dense_4_features],
            feed_dict=feed_dict)

        dense_2_features.append(dense_2)
        dense_4_features.append(dense_4)

    dense_2_features = np.vstack(dense_2_features)
    dense_4_features = np.vstack(dense_4_features)
    # hamming distance - vectors processing
    if distance == 'hamming':
        dense_2_features = np.where(dense_2_features < 0.5, 0,
                                    1)  # binarize vectors
        dense_4_features = np.where(dense_4_features < 0.5, 0, 1)

        training_vectors = np.hstack((dense_2_features, dense_4_features))
        with open('pickle_saves/hamming_train_vectors.pickle', 'wb') as f:
            pickle.dump(training_vectors, f)

    # cosine distance - vectors processing
    elif distance == 'cosine':
        training_vectors = np.hstack((dense_2_features, dense_4_features))
        training_vectors = np.hstack(
            (training_vectors, color_features[:len(training_vectors)]))
        with open('pickle_saves/cosine_train_vectors.pickle', 'wb') as f:
            pickle.dump(training_vectors, f)


def create_training_set_vectors_with_colors(model,
                                            x_train,
                                            y_train,
                                            batch_size,
                                            checkpoint_path,
                                            image_size,
                                            distance='hamming'):
    # Define session
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    # Restore session
    saver = tf.train.Saver()
    saver.restore(session, checkpoint_path)

    dense_2_features = []
    dense_4_features = []

    ##########################################################################
    ### Calculate color feature vectors for each image in the training set ###
    color_features = []
    for img in x_train:
        channels = cv.split(img)
        features = []
        for chan in channels:
            hist = cv.calcHist([chan], [0], None, [256], [0, 256])
            features.append(hist)

        color_features.append(np.vstack(features).squeeze())
    ##########################################################################

    #iterate through training set
    for j in range(len(x_train) // batch_size):
        start_id = j * batch_size
        end_id = start_id + batch_size

        x_batch = x_train[start_id:end_id]

        feed_dict = {model.inputs: x_batch, model.dropout_rate: 0.0}

        dense_2, dense_4 = session.run(
            [model.dense_2_features, model.dense_4_features],
            feed_dict=feed_dict)

        dense_2_features.append(dense_2)
        dense_4_features.append(dense_4)

    dense_2_features = np.vstack(dense_2_features)
    dense_4_features = np.vstack(dense_4_features)
    #hamming distance - vectors processing
    if distance == 'hamming':
        dense_2_features = np.where(dense_2_features < 0.5, 0,
                                    1)  #binarize vectors
        dense_4_features = np.where(dense_4_features < 0.5, 0, 1)

        training_vectors = np.hstack((dense_2_features, dense_4_features))
        with open('pickle_saves/hamming_train_vectors.pickle', 'wb') as f:
            pickle.dump(training_vectors, f)

    #cosine distance - vectors processing
    elif distance == 'cosine':
        training_vectors = np.hstack((dense_2_features, dense_4_features))
        training_vectors = np.hstack(
            (training_vectors, color_features[:len(training_vectors)]))
        with open('pickle_saves/cosine_train_vectors.pickle', 'wb') as f:
            pickle.dump(training_vectors, f)

    ### Save training set color feature vectors to a separate pickle file ###
    with open('pickle_saves/color_vectors.pickle', 'wb') as f:
        pickle.dump(color_features[:len(training_vectors)], f)
=======
import os
import pickle
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import config as cfg
from model import ImageSearchNetwork


def train(model,
          epochs,
          drop_rate,
          batch_size,
          data,
          save_dir,
          saver_delta=0.15):

    x_train, y_train, x_test, y_test = data

    # start session
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    # define saver
    saver = tf.train.Saver()

    best_test_accuracy = 0.0
    # start training loop
    for epoch in range(epochs):

        train_accuracy = []
        train_loss = []

        for j in range(len(x_train) // batch_size):
            start_id = j * batch_size
            end_id = start_id + batch_size

            x_batch = x_train[start_id:end_id]
            y_batch = y_train[start_id:end_id]

            feed_dict = {
                model.inputs: x_batch,
                model.targets: y_batch,
                model.dropout_rate: drop_rate
            }

            _, t_loss, preds_tr = session.run(
                [model.optimizer, model.loss, model.predictions],
                feed_dict=feed_dict)

            train_accuracy.append(sparse_accuracy(y_batch, preds_tr))
            train_loss.append(t_loss)

        print("Epoch: {}/{}".format(epoch, epochs),
              " | Training accuracy: {}".format(np.mean(train_accuracy)),
              " | Training loss: {}".format(np.mean(train_loss)))

        test_accuracy = []

        for j in range(len(x_test) // batch_size):
            start_id = j * batch_size
            end_id = start_id + batch_size

            x_batch = x_test[start_id:end_id]
            y_batch = y_test[start_id:end_id]

            feed_dict = {model.inputs: x_batch, model.dropout_rate: 0.0}

            preds_test = session.run(model.predictions, feed_dict=feed_dict)
            test_accuracy.append(sparse_accuracy(y_batch, preds_test))

        print("Test accuracy: {}".format(np.mean(test_accuracy)))

        # saving the model
        if np.mean(train_accuracy) > np.mean(
                test_accuracy):  # to prevent underfitting
            if np.abs(np.mean(train_accuracy) - np.mean(test_accuracy)
                      ) <= saver_delta:  # to prevent overfitting
                if np.mean(test_accuracy) >= best_test_accuracy:
                    best_test_accuracy = np.mean(test_accuracy)
                    saver.save(
                        session,
                        "{}/model_epoch_{}.ckpt".format(save_dir, epoch))

    session.close()


def create_training_set_vectors(
        model,
        x_train,
        y_train,
        batch_size,
        checkpoint_path,  # string path to the model checkpoint
        image_size,
        distance='hamming'):
    # Define session
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    # Restore session
    saver = tf.train.Saver()
    saver.restore(session, checkpoint_path)

    dense_2_features = []
    dense_4_features = []

    #iterate through training set
    for j in range(len(x_train) // batch_size):
        start_id = j * batch_size
        end_id = start_id + batch_size

        x_batch = x_train[start_id:end_id]

        feed_dict = {model.inputs: x_batch, model.dropout_rate: 0.0}

        dense_2, dense_4 = session.run(
            [model.dense_2_features, model.dense_4_features],
            feed_dict=feed_dict)

        dense_2_features.append(dense_2)
        dense_4_features.append(dense_4)

    dense_2_features = np.vstack(dense_2_features)
    dense_4_features = np.vstack(dense_4_features)
    # hamming distance - vectors processing
    if distance == 'hamming':
        dense_2_features = np.where(dense_2_features < 0.5, 0,
                                    1)  # binarize vectors
        dense_4_features = np.where(dense_4_features < 0.5, 0, 1)

        training_vectors = np.hstack((dense_2_features, dense_4_features))
        with open('pickle_saves/hamming_train_vectors.pickle', 'wb') as f:
            pickle.dump(training_vectors, f)

    # cosine distance - vectors processing
    elif distance == 'cosine':
        training_vectors = np.hstack((dense_2_features, dense_4_features))
        training_vectors = np.hstack(
            (training_vectors, color_features[:len(training_vectors)]))
        with open('pickle_saves/cosine_train_vectors.pickle', 'wb') as f:
            pickle.dump(training_vectors, f)


def create_training_set_vectors_with_colors(model,
                                            x_train,
                                            y_train,
                                            batch_size,
                                            checkpoint_path,
                                            image_size,
                                            distance='hamming'):
    # Define session
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    # Restore session
    saver = tf.train.Saver()
    saver.restore(session, checkpoint_path)

    dense_2_features = []
    dense_4_features = []

    ##########################################################################
    ### Calculate color feature vectors for each image in the training set ###
    color_features = []
    for img in x_train:
        channels = cv.split(img)
        features = []
        for chan in channels:
            hist = cv.calcHist([chan], [0], None, [256], [0, 256])
            features.append(hist)

        color_features.append(np.vstack(features).squeeze())
    ##########################################################################

    #iterate through training set
    for j in range(len(x_train) // batch_size):
        start_id = j * batch_size
        end_id = start_id + batch_size

        x_batch = x_train[start_id:end_id]

        feed_dict = {model.inputs: x_batch, model.dropout_rate: 0.0}

        dense_2, dense_4 = session.run(
            [model.dense_2_features, model.dense_4_features],
            feed_dict=feed_dict)

        dense_2_features.append(dense_2)
        dense_4_features.append(dense_4)

    dense_2_features = np.vstack(dense_2_features)
    dense_4_features = np.vstack(dense_4_features)
    #hamming distance - vectors processing
    if distance == 'hamming':
        dense_2_features = np.where(dense_2_features < 0.5, 0,
                                    1)  #binarize vectors
        dense_4_features = np.where(dense_4_features < 0.5, 0, 1)

        training_vectors = np.hstack((dense_2_features, dense_4_features))
        with open('pickle_saves/hamming_train_vectors.pickle', 'wb') as f:
            pickle.dump(training_vectors, f)

    #cosine distance - vectors processing
    elif distance == 'cosine':
        training_vectors = np.hstack((dense_2_features, dense_4_features))
        training_vectors = np.hstack(
            (training_vectors, color_features[:len(training_vectors)]))
        with open('pickle_saves/cosine_train_vectors.pickle', 'wb') as f:
            pickle.dump(training_vectors, f)

    ### Save training set color feature vectors to a separate pickle file ###
    with open('pickle_saves/color_vectors.pickle', 'wb') as f:
        pickle.dump(color_features[:len(training_vectors)], f)
>>>>>>> 1a52691e478b935107d7c1df5d7314afca17c587
