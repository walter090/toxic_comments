from functools import wraps

import tensorflow as tf

import structure


def property_wrap(attr):
    """Checks if the function has already been called"""

    def de_facto_wrap(func):
        @property
        @wraps(func)
        def decorator(self):
            if getattr(self, attr) is None:
                setattr(self, attr, func(self))
            return getattr(self, attr)

        return decorator

    return de_facto_wrap


class ToxicityCNN:
    def __init__(self, csvs=None, batch_size=None,
                 num_epochs=None, vocab_size=None, embedding_size=None,
                 num_labels=None):
        """
        Args:
            csvs: list, a list of strings that are names of csv files to be used
                as dataset.
            batch_size: int, size of each batch.
            num_epochs: int, number of epochs.
            vocab_size: int, vocabulary size for the word embeddings.
            embedding_size: int, size of each word vector.
        """
        self.comment_length = None
        self.comment_batch, self.toxicity_batch = None, None
        self.embedded = None
        self.embeddings = None
        self.embedding_size = embedding_size
        self.num_labels = num_labels
        self.vocab_size = vocab_size

        self._loss = None
        self._optimize = None
        self._prediction = None

        if csvs and batch_size and num_epochs and num_labels:
            self.file_read_op(csvs, batch_size, num_labels, num_epochs)

        if vocab_size and embedding_size:
            self.create_embedding(vocab_size, embedding_size)

    def file_read_op(self, file_names, batch_size,
                     num_labels, num_epochs):
        """Read csv files in batch

        Args:
            file_names: list, list of file names.
            batch_size: int, batch size.
            num_labels: int, number of labels.
            num_epochs: int, number of epochs.

        Returns:
            None
        """
        self.num_labels = num_labels

        reader = tf.TextLineReader(skip_header_lines=1)
        queue = tf.train.string_input_producer(file_names,
                                               num_epochs=num_epochs,
                                               shuffle=True)

        _, value = reader.read(queue)
        record_defaults = [[''], [''], [0], [0], [0], [0], [0], [0], *([[-3]] * 60)]
        cols = tf.decode_csv(value, record_defaults=record_defaults)
        comment_text = tf.stack(cols[-60:])  # Skip id column
        toxicity = tf.stack(cols[2:8])

        min_after_dequeue = 10000
        capacity = min_after_dequeue + 4 * batch_size
        self.comment_batch, self.toxicity_batch = tf.train.shuffle_batch(
            [comment_text, toxicity], batch_size=batch_size,
            capacity=capacity, min_after_dequeue=min_after_dequeue)

    def create_embedding(self, vocab_size, embedding_size,
                         name='embedding'):
        """ Create embedding

        Args:
            vocab_size: Int, vocabulary size.
            embedding_size: Int, size of word vector.
            name: String, operation name.

        Returns:
            None
        """
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        with tf.variable_scope(name):
            self.embeddings = tf.get_variable(name='embedding_w',
                                              shape=[vocab_size, embedding_size],
                                              initializer=tf.random_uniform_initializer(-1, 1))
            self.embedded = tf.nn.embedding_lookup(self.embeddings, self.comment_batch)

    def network(self, x_input=None, num_output=None,
                layer_config=None, fully_conn_config=None, pool='max',
                name='network', padding='VALID', batchnorm=True,
                reuse_variables=False):
        """This is where the neural net is implemented. Each of the config is a list,
        each element for one layer. Inception is available by adding more dimensions
        to the config lists. The prediction property calls this function with all its
        default arguments.

        Args:
            x_input: Tensor, input tensor to the network.
            num_output: int, size of final output from the output layer.
            layer_config: list, a list that contains configuration for each layer.
            fully_conn_config: list, a list that contains configuration for each fully connected
                layer.
            pool: string, pooling method.
            name: string, name for the network.
            padding: string, specify padding method used by convolution. Choose from SAME and
                VALID, defaults VALID.
            batchnorm: boolean, set True to use batch normalization.
            reuse_variables: boolean, Set to True to reuse weights and biases.

        Returns:
            output: Tensor, output tensor from the network.
        """

        def pool_size(ksize):
            return self.comment_length - ksize + 1

        with tf.variable_scope(name, reuse=reuse_variables):
            if not (x_input and num_output):
                x_input = self.embedded
                num_output = self.num_labels

            layer_config = [
                # Convolution layer configuration
                # ksize, stride, out_channels, pool_ksize, pool_stride
                [
                    [2, 1, 256, pool_size(2), 1],
                ],
                [
                    [3, 1, 256, pool_size(3), 1],
                ],
                [
                    [4, 1, 256, pool_size(4), 1],
                ],
                [
                    [5, 1, 256, pool_size(5), 1],
                ],
            ] if not layer_config else layer_config

            fully_conn_config = [
                [1024, 'lrelu', 0.5],
                [512, 'lrelu', 0.5],
            ] if not fully_conn_config else fully_conn_config

            outputs = []
            pool_output = x_input

            for config_i, config in enumerate(layer_config):
                for layer_i, layer in enumerate(config):
                    pool_output = structure.conv_pool(
                        pool_output, ksize=[self.embedding_size, layer[0]],
                        stride=[1, layer[1]], out_channels=layer[2],
                        pool_ksize=[self.embedding_size, layer[3]],
                        pool_stride=[1, layer[4]], alpha=0.1,
                        padding=padding, batchnorm=batchnorm,
                        method=pool, name='conv_{}_{}'.format(config_i, layer_i))

                outputs.append(pool_output)

        pool_concat = tf.concat(outputs, axis=3)
        output_flat = structure.flatten(pool_concat)

        output_fully_conn = output_flat

        for layer_i, layer in enumerate(fully_conn_config):
            output_fully_conn = structure.fully_conn(output_fully_conn, num_output=layer[0],
                                                     activation=layer[1], keep_prob=layer[2],
                                                     name='fc_{}'.format(layer_i))

        output_logits = structure.fully_conn(output_fully_conn, num_output=num_output,
                                             name='fc_out', activation=None)
        output = tf.nn.sigmoid(output_logits)
        prediction = tf.argmax(output, axis=1)

        return output_logits, output, prediction

    @property_wrap('_prediction')
    def prediction(self):
        self._prediction = self.network()
        return self._prediction

    @property_wrap('_loss')
    def loss(self):
        logits, output, pred = self.prediction
        losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.toxicity_batch)
        loss = tf.reduce_mean(losses)

        tf.summary.scalar('loss', loss)

        self._loss = loss
        return self._loss

    @property_wrap('_optimize')
    def optimize(self):
        global_step = tf.Variable(0, trainable=False, name='global_step')

        optimizer = tf.train.AdamOptimizer(1e-4)
        grads = optimizer.compute_gradients(self.loss)

        for grad_i, grad in enumerate(grads):
            tf.summary.histogram('grad_{}'.format(grad[1].name), grad)

        self._optimize = optimizer.apply_gradients(grads_and_vars=grads, global_step=global_step)
        return self._optimize
