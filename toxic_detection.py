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
    def __init__(self):
        self.comment_batch = None
        self.toxicity_batch = None
        self.embeddings = None

        self._prediction = None
        self._optimizer = None
        self._error = None

    def file_read_op(self, file_names, batch_size, num_epochs):
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

    def create_embedding(self, vocab_size, embedding_size, name='embedding'):
        """ Create embedding

        Args:
            vocab_size: Int, vocabulary size.
            embedding_size: Int, size of word vector.
            name: String, operation name.

        Returns:
            embedded_word_exp: Tensor, word embedding
        """
        with tf.variable_scope(name):
            self.embeddings = tf.get_variable(name='embedding_w',
                                              shape=[vocab_size, embedding_size],
                                              initializer=tf.random_uniform_initializer(-1, 1))

    @property_wrap('_prediction')
    def predict(self):
        raise NotImplementedError

    @property_wrap('_optimizer')
    def optimize(self):
        raise NotImplementedError

    @property_wrap('_error')
    def geterror(self):
        raise NotImplementedError
