import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from utils.model import *


# Refer to README.md to see the complete summary of the model when following along this cell of code
class ImageSearchModel(object):
    def __init__(
        self,
        learning_rate,
        size,  # tuple of (height, width) of an image
        number_of_classes=10):  # integer number of classes in a dataset

        tf.reset_default_graph(
        )  # escape possibility of having nested models graphs

        # model inputs
        self.inputs, self.targets, self.dropout_rate = model_inputs(size)
        # every image is 0 to 255, so we add batch norm layer
        normalized_images = tf.layers.batch_normalization(self.inputs)

        # conv_1 block
        conv_block_1, self.conv_1_features = conv_block(
            inputs=normalized_images,
            number_of_filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='SAME',
            activation=tf.nn.relu,
            max_pool=True,
            batch_norm=True)
        # conv_2 block
        conv_block_2, self.conv_2_features = conv_block(inputs=conv_block_1,
                                                        number_of_filters=128,
                                                        kernel_size=(3, 3),
                                                        strides=(1, 1),
                                                        padding='SAME',
                                                        activation=tf.nn.relu,
                                                        max_pool=True,
                                                        batch_norm=True)
        # conv_3 block
        conv_block_3, self.conv_3_features = conv_block(inputs=conv_block_2,
                                                        number_of_filters=256,
                                                        kernel_size=(5, 5),
                                                        strides=(1, 1),
                                                        padding='SAME',
                                                        activation=tf.nn.relu,
                                                        max_pool=True,
                                                        batch_norm=True)
        # conv_4 block
        conv_block_4, self.conv_4_features = conv_block(inputs=conv_block_3,
                                                        number_of_filters=512,
                                                        kernel_size=(5, 5),
                                                        strides=(1, 1),
                                                        padding='SAME',
                                                        activation=tf.nn.relu,
                                                        max_pool=True,
                                                        batch_norm=True)
        # flattening the last conv lock to one single vector flat_layer
        flat_layer = tf.layers.flatten(conv_block_4)

        # dense_1 block
        dense_block_1, dense_1_features = dense_block(
            inputs=flat_layer,
            units=128,
            activation=tf.nn.relu,
            dropout_rate=self.dropout_rate,
            batch_norm=True)
        # dense_2 block
        dense_block_2, self.dense_2_features = dense_block(
            inputs=dense_block_1,
            units=256,
            activation=tf.nn.relu,
            dropout_rate=self.dropout_rate,
            batch_norm=True)
        # dense_3 block
        dense_block_3, self.dense_3_features = dense_block(
            inputs=dense_block_2,
            units=512,
            activation=tf.nn.relu,
            dropout_rate=self.dropout_rate,
            batch_norm=True)
        # dense_4 block
        dense_block_4, self.dense_4_features = dense_block(
            inputs=dense_block_3,
            units=1024,
            activation=tf.nn.relu,
            dropout_rate=self.dropout_rate,
            batch_norm=True)
        # output layer
        logits = tf.layers.dense(inputs=dense_block_4,
                                 units=number_of_classes,
                                 activation=None)

        self.predictions = tf.nn.softmax(
            logits)  # turn logits into a probs vector perdictions

        self.loss, self.optimizer = opt_loss(logits=logits,
                                             targets=self.targets,
                                             learning_rate=learning_rate)