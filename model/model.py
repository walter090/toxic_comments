import tensorflow as tf

from structure import property_wrap


class Model:
    def __init__(self, csvs=None, batch_size=None,
                 num_epochs=None, vocab_size=None, embedding_size=None,
                 num_labels=None, comment_length=None, testing=False,
                 vec=None):
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
            self.file_read_op(csvs, batch_size, num_epochs,
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

    def file_read_op(self, file_names, batch_size,
                     num_epochs, num_labels, comment_length,
                     *args, **kwargs):
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

        record_defaults = [[''], [''], [0], [0], [0], [0], [0], [0], *([[-3]] * 60)]
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

    def network(self, *args, **kwargs):
        pass

    @property_wrap('_prediction')
    def prediction(self):
        self._prediction = self.network()
        return self._prediction

    @property_wrap('_loss')
    def loss(self):
        return self._loss

    @property_wrap('_metric')
    def metric(self):
        return self._metric

    @property_wrap('_optimize')
    def optimize(self):
        return self._optimize

    @property_wrap('_embeddings')
    def embeddings(self):
        return self._embeddings
