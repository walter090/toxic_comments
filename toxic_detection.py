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
    def __init__(self, vocab_size, embedding_size, ksize, out_channels, csv):
        self.comment_batch, self.toxicity_batch = structure. \
            file_read_op(file_names=[csv], batch_size=4000, num_epochs=100)
        self._prediction = None
        self._optimizer = None
        self._error = None
        raise NotImplementedError

    @property_wrap('_prediction')
    def predict(self):
        raise NotImplementedError

    @property_wrap('_optimizer')
    def optimize(self):
        raise NotImplementedError

    @property_wrap('_error')
    def geterror(self):
        raise NotImplementedError
