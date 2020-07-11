import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def model_inputs(size):
    # tuple of (height, width) of an image
    # shape = [batch_size, size[0], size[1], 3]  we set batch_size to None so it accepts any number
    # defining CNN inputs as placeholders
    inputs = tf.placeholder(dtype=tf.float32,
                            shape=[None, size[0], size[1], 3],
                            name='images')
    targets = tf.placeholder(dtype=tf.int32, shape=[
        None,
    ], name='targets')  # array of true labels
    dropout_prob = tf.placeholder(dtype=tf.float32, name='dropout_probs')

    return inputs, targets, dropout_prob


def conv_block(
    inputs,  # data from a previous layer
    number_of_filters,  # integer, number of conv filters
    kernel_size,  # tuple, size of conv layer kernel
    strides=(1, 1),
    padding='SAME',  # string, type of padding technique: SAME or VALID
    activation=tf.nn.relu,  # tf.object, activation function used on the layer
    max_pool=True,  # boolean, if true the conv block will use max_pool
    batch_norm=True
):  # boolean, if true the conv block will use batch normalization

    conv_features = layer = tf.layers.conv2d(inputs=inputs,
                                             filters=number_of_filters,
                                             kernel_size=kernel_size,
                                             strides=strides,
                                             padding=padding,
                                             activation=activation)

    if max_pool:
        layer = tf.layers.max_pooling2d(layer,
                                        pool_size=(2, 2),
                                        strides=(2, 2),
                                        padding='SAME')

    if batch_norm:
        layer = tf.layers.batch_normalization(layer)

    return layer, conv_features


def dense_block(
    inputs,  # data from a previous layer
    units,  # integer, number of neurons/units for a dense layer
    activation=tf.nn.relu,  # tf.object, activation function used on the layer
    dropout_rate=None,  # dropout rate used in this dense block
    batch_norm=True
):  # boolean, if true the conv block will use batch normalization

    dense_features = layer = tf.layers.dense(inputs,
                                             units=units,
                                             activation=activation)

    if dropout_rate is not None:
        layer = tf.layers.dropout(layer, rate=dropout_rate)

    if batch_norm:
        layer = tf.layers.batch_normalization(layer)

    return layer, dense_features


def opt_loss(
        logits,  # pre-activated model outputs
        targets,  # true labels for each input sample
        learning_rate):
    # sparse means that we don't have to convert our targets to one hot encoding version
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,
                                                       logits=logits))
    # Adam optimizer performs best on CNNs
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(loss)

    return loss, optimizer