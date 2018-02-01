import tensorflow as tf

from .model import Model
from structure import property_wrap


class WordEmbedding(Model):
    def __init__(self, csvs=None, batch_size=None,
                 num_epochs=None, vocab_size=None, embedding_size=None, ):
        super(WordEmbedding, self).__init__(vocab_size=vocab_size,
                                            embedding_size=embedding_size)
        self.target_batch, self.context_batch = None, None
        self.embedded_context = None

        if csvs and batch_size and num_epochs:
            self.file_read_op(file_names=csvs, batch_size=batch_size,
                              num_epochs=num_epochs)

    def file_read_op(self, file_names, batch_size,
                     num_epochs, num_labels=None, comment_length=None):
        """Read csv files and create input batches.

        Args:
            file_names: list, a list of string file names.
            batch_size: int, size of each batch.
            num_epochs: int, number of epochs to create.
            num_labels: int, optional, number of label in the training data.
            comment_length: int, optional, number of words in a text.

        Returns:

        """
        reader = tf.TextLineReader(skip_header_lines=1)
        queue = tf.train.string_input_producer(file_names,
                                               num_epochs=num_epochs,
                                               shuffle=True)

        _, value = reader.read(queue)
        record_defaults = [[0], [0]]
        cols = tf.decode_csv(value, record_defaults=record_defaults)
        target = cols[0]
        context = cols[1]

        min_after_dequeue = 10000
        capacity = min_after_dequeue + 4 * batch_size
        self.target_batch, self.context_batch = tf.train.shuffle_batch(
            [target, context], batch_size=batch_size,
            capacity=capacity, min_after_dequeue=min_after_dequeue)

    def network(self, input_x, name='nce'):
        pass

    @property_wrap('_prediction')
    def prediction(self):
        return self._prediction

    @property_wrap('_loss')
    def loss(self):
        with tf.get_variable('nce'):
            weights = tf.get_variable(name='nce_weights',
                                      shape=[self.vocab_size, self.embedding_size],
                                      initializer=tf.truncated_normal_initializer)
            biases = tf.get_variable(name='nce_biases',
                                     shape=[self.vocab_size],
                                     initializer=tf.zeros_initializer)
            self._loss = tf.nn.nce_loss(weights=weights, biases=biases,
                                        labels=self.context_batch, inputs=self.target_batch,
                                        num_classes=self.vocab_size, num_sampled=2000)
        return self._loss

    @property_wrap('_optimize')
    def optimize(self):
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        optimizer = tf.train.AdamOptimizer(1e-4)
        grads = optimizer.compute_gradients(self.loss)

        self._optimize = grads, optimizer.apply_gradients(grads_and_vars=grads,
                                                          global_step=self.global_step)
        return self._optimize

    @property_wrap('_embeddings')
    def embeddings(self):
        with tf.variable_scope('embedding'):
            self._embeddings = tf.get_variable(name='embedding_w',
                                               shape=[self.vocab_size, self.embedding_size],
                                               initializer=tf.random_uniform_initializer(-1, 1))
            self.embedded = tf.nn.embedding_lookup(self._embeddings, self.target_batch)
            self.embedded_context = tf.nn.embedding_lookup(self._embeddings, self.context_batch)
            return self._embeddings
