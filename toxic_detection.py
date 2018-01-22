import structure
import tensorflow as tf
from functools import wraps


def property_wrap(attr):
    """Checks if the function has already be called"""

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
    def __init__(self, csvs=None, batch_size=None, num_epochs=None,
                 vocab_size=None, embedding_size=None):
        """
        Args:
            csvs: list, a list of strings that are names of csv files to be used
                as dataset.
            batch_size: int, size of each batch.
            num_epochs: int, number of epochs.
            vocab_size: int, vocabulary size for the word embeddings.
            embedding_size: int, size of each word vector.
        """
        self.comment_batch, self.toxicity_batch = None, None
        self.embeddings = None

        self._prediction = None
        self._optimizer = None
        self._error = None

        if csvs and batch_size and num_epochs:
            self._file_read_op(csvs, batch_size, num_epochs)

        if vocab_size and embedding_size:
            self._create_embedding(vocab_size, embedding_size)

    def _file_read_op(self, file_names, batch_size, num_epochs):
        """Read csv files in batch

        Args:
            file_names: List of file names.
            batch_size: Int, batch size.
            num_epochs: Int, number of epochs.

        Returns:
            None
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
        self.comment_batch, self.toxicity_batch = tf.train.shuffle_batch(
            [comment_text, toxicity], batch_size=batch_size,
            capacity=capacity, min_after_dequeue=min_after_dequeue)

    def _create_embedding(self, vocab_size, embedding_size, name='embedding'):
        """ Create embedding

        Args:
            vocab_size: Int, vocabulary size.
            embedding_size: Int, size of word vector.
            name: String, operation name.

        Returns:
            None
        """
        with tf.variable_scope(name):
            self.embeddings = tf.get_variable(name='embedding_w',
                                              shape=[vocab_size, embedding_size],
                                              initializer=tf.random_uniform_initializer(-1, 1))

    def network(self, layer_config, fully_conn_config, name='network'):
        # TODO implement network
        """This is where the neural net is implemented. Each of the config is a list,
        each element for one layer. Inception is available by adding more dimensions
        to the config lists.
        """
        raise NotImplementedError

    @property_wrap('_prediction')
    def predict(self, x_input, model):
        # TODO implement predict
        raise NotImplementedError

    @property_wrap('_optimizer')
    def optimize(self):
        # TODO implement optimize
        raise NotImplementedError

    @property_wrap('_error')
    def geterror(self):
        # TODO implement geterror
        raise NotImplementedError
