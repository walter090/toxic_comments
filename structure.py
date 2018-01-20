import numpy as np
import tensorflow as tf


def conv(x, ksize, stride, out_channels, name='conv', padding='VALID'):
    """Convolution layer

    Args:
        x: Input tensor
        ksize: Filter size
        stride: Stride size
        out_channels: Number of filters
        name: String name of op
        padding: String padding method, defaults 'VALID'

    Returns:
        Convoluted tensor
    """
    with tf.variable_scope(name):
        weights = tf.get_variable(name='conv_w',
                                  shape=[ksize[0], ksize[1],
                                         x.get_shape().as_list()[3], out_channels],
                                  initializer=tf.random_normal_initializer())
        bias = tf.get_variable(name='conv_b',
                               shape=[out_channels],
                               initializer=tf.zeros_initializer())
        stride = [1, *stride, 1]

        convoluted = tf.nn.convolution(x, filter=weights,
                                       strides=stride, padding=padding)
        convoluted = tf.nn.bias_add(convoluted, bias)

        return convoluted


def lrelu(x, alpha=0.1):
    """Leaky ReLU activation.

    Args:
        x(Tensor): Input from the previous layer.
        alpha(float): Parameter for if x < 0.

    Returns:
        Output tensor
    """
    # linear = 0.5 * x + 0.5 * tf.abs(x)
    # leaky = 0.5 * alpha * x - 0.5 * alpha * tf.abs(x)
    # output = leaky + linear

    linear = tf.add(
        tf.multiply(0.5, x),
        tf.multiply(0.5, tf.abs(x))
    )
    half = tf.multiply(0.5, alpha)
    leaky = tf.subtract(
        tf.multiply(half, x),
        tf.multiply(half, tf.abs(x))
    )
    output = tf.add(linear, leaky)

    return output


def flatten(x):
    """Flatten a tensor for the fully connected layer.
    Each image in a batch is flattened.

    Args:
        x(Tensor): 4-D tensor of shape [batch, height, width, channels] to be flattened
            to the shape of [batch, height * width * channels]

    Returns:
        Flattened tensor.
    """
    return tf.reshape(x, shape=[-1, np.prod(x.get_shape().as_list()[1:])])


def fully_conn(x,
               num_output,
               name='fc',
               activation='lrelu',
               keep_prob=1.):
    """Fully connected layer, this is is last parts of convnet.
    Fully connect layer requires each image in the batch be flattened.

    Args:
        x: Input from the previous layer.
        num_output: Output size of the fully connected layer.
        name: Name for the fully connected layer variable scope.
        activation: Set to True to add a leaky relu after fully connected
            layer. Set this argument to False if this is the final layer.
        keep_prob: Keep probability for dropout layers, if keep probability is 1
            there is no dropout. Defaults 1.s

    Returns:
        Output tensor.
    """
    with tf.variable_scope(name):
        weights = tf.get_variable(name='fc_w',
                                  shape=[x.get_shape().as_list()[-1], num_output],
                                  initializer=tf.random_normal_initializer(stddev=0.02))
        biases = tf.get_variable(name='fc_b',
                                 shape=[num_output],
                                 initializer=tf.zeros_initializer())

        output = tf.nn.bias_add(tf.matmul(x, weights), biases)
        output = tf.nn.dropout(output, keep_prob=keep_prob)

        if activation == 'sigmoid':
            output = tf.sigmoid(output)
        elif activation == 'lrelu':
            output = lrelu(output)
        else:
            pass

        return output


def embedding(ids, vocab_size, embedding_size, name='embedding'):
    """ Function for embedding lookup

    Args:
        ids: Tensor, index id.
        vocab_size: Int, vocabulary size.
        embedding_size: Int, size of word vector.
        name: String, operation name.

    Returns:
        embedded_word_exp: Tensor, word embedding
    """
    with tf.variable_scope(name):
        weights = tf.get_variable(name='embedding_w',
                                  shape=[vocab_size, embedding_size],
                                  initializer=tf.random_uniform_initializer(-1, 1))
        embedded_word_exp = tf.expand_dims(tf.nn.embedding_lookup(weights, ids), -1)

        return embedded_word_exp


def file_read_op(file_names, batch_size, num_epochs):
    """Read csv files in batch

    Args:
        file_names: List of file names.
        batch_size: Int, batch size.
        num_epochs: Int, number of epochs.

    Returns:
        comment_text_batch: A tensor for a batch of comment text.
        toxicity_batch: A tensor of a batch of one hot encoded toxicity.
    """
    reader = tf.TextLineReader(skip_header_lines=1)
    queue = tf.train.string_input_producer(file_names,
                                           num_epochs=num_epochs,
                                           shuffle=True)

    _, value = reader.read(queue)
    record_defaults = [[''], [''], [0], [0], [0], [0], [0], [0]]
    cols = tf.decode_csv(value, record_defaults=record_defaults)
    comment_text = cols[1]  # Skip id column
    toxicity = tf.stack(cols[2:])

    min_after_dequeue = 10000
    capacity = min_after_dequeue + 4 * batch_size
    comment_text_batch, toxicity_batch = tf.train.shuffle_batch(
        [comment_text, toxicity], batch_size=batch_size,
        capacity=capacity, min_after_dequeue=min_after_dequeue)
    return comment_text_batch, toxicity_batch
