import tensorflow as tf

from structure import property_wrap


class Model:
    def __init__(self, csvs=None, batch_size=None,
                 num_epochs=None, vocab_size=None, embedding_size=None,
                 num_labels=None, comment_length=None, testing=False,
                 vec=None, keep_prob=0.5):
        """
        Args:
            vocab_size: int, vocabulary size for the word embeddings.
            embedding_size: int, size of each word vector.
        """
        self.num_labels = num_labels
        self.comment_length = comment_length
        self.comment_batch, self.toxicity_batch, self.id_batch = None, None, None
        self.testing = testing
        self.vec = vec
        self.keep_prob = keep_prob

        self.embedded = None
        self.embedding_size = embedding_size
        self.global_step = None
        self.vocab_size = vocab_size
        self.batch_size = batch_size

        self._embeddings = None
        self._loss = None
        self._metric = None
        self._optimize = None
        self._prediction = None

        if csvs and batch_size and num_epochs \
                and num_labels and comment_length:
            self._file_read_op(csvs, batch_size, num_epochs,
                               num_labels, comment_length)

        if self.vec is not None:
            self.vocab_size = vec.shape[0]
            self.embedding_size = len(vec[0])

    def provide_vector(self, vec):
        """ Provide pre trained word embeddings

        Args:
            vec: numpy array, word vectors.

        Returns:
            None
        """
        self.vec = vec
        self.vocab_size = vec.shape[0]
        self.embedding_size = len(vec[0])

    def create_embedding(self, vocab_size, embedding_size):
        """ Create embedding

            Args:
                vocab_size: Int, vocabulary size.
                embedding_size: Int, size of word vector.

            Returns:
                None
        """
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

    def _file_read_op(self, file_names, batch_size,
                      num_epochs, num_labels=None, comment_length=None,
                      record_defaults=None, *args, **kwargs):
        """Read csv files in batch

        Args:
            file_names: list, list of file names.
            batch_size: int, batch size.
            num_epochs: int, number of epochs.
            num_labels: int, number of labels.
            comment_length: int, length of each comment.

        Returns:
            None
        """
        self.num_labels = num_labels
        self.comment_length = comment_length
        self.batch_size = batch_size

        reader = tf.TextLineReader(skip_header_lines=1)
        queue = tf.train.string_input_producer(file_names,
                                               num_epochs=num_epochs,
                                               shuffle=True)

        _, value = reader.read(queue)

        record_defaults = record_defaults if record_defaults\
            else [[''], [''], [0], [0], [0], [0], [0], [0], *([[-3]] * 60)]
        cols = tf.decode_csv(value, record_defaults=record_defaults)
        comment_id = cols[0]
        comment_text = tf.stack(cols[-60:])
        toxicity = tf.stack(cols[2:8])

        min_after_dequeue = 10000
        capacity = min_after_dequeue + 4 * batch_size
        self.comment_batch, self.toxicity_batch, self.id_batch = tf.train.shuffle_batch(
            [comment_text, toxicity, comment_id], batch_size=batch_size,
            capacity=capacity, min_after_dequeue=min_after_dequeue
        )
        self.toxicity_batch = tf.cast(self.toxicity_batch, dtype=tf.float32)

    def _network(self, *args, **kwargs):
        pass

    def predict(self, data, *args, **kwargs):
        pass

    @property_wrap('_prediction')
    def prediction(self):
        self._prediction = self._network()
        return self._prediction

    @property_wrap('_loss')
    def loss(self):
        logits, output, _ = self.prediction
        losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.toxicity_batch)
        loss = tf.reduce_mean(losses)

        self._loss = loss
        return self._loss

    @property_wrap('_metric')
    def metric(self):
        _, auc = tf.metrics.auc(labels=self.toxicity_batch,
                                predictions=self.prediction[1])
        self._metric = auc
        return self._metric

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
            embedding_initializer = tf.constant_initializer(self.vec) if self.vec is not None \
                else tf.random_uniform_initializer(-1, 1)

            self._embeddings = tf.get_variable(name='embedding_w',
                                               shape=[self.vocab_size, self.embedding_size],
                                               initializer=embedding_initializer,
                                               trainable=self.vec is None)
            embedded = tf.nn.embedding_lookup(self._embeddings, self.comment_batch)

            mask = tf.concat([tf.ones([1, self.embedding_size]),
                              tf.zeros([1, self.embedding_size]),
                              tf.ones([self.vocab_size - 1, self.embedding_size])], axis=0)
            embedded_masked = tf.nn.embedding_lookup(mask, self.comment_batch)
            self.embedded = tf.multiply(embedded, embedded_masked)

            return self._embeddings, self.embedded
