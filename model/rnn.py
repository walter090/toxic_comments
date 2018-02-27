import tensorflow as tf

import structure
from .model import Model
from structure import property_wrap


class ToxicityLSTM(Model):
    def __init__(self, vocab_size=None, embedding_size=None):
        super().__init__(vocab_size=vocab_size,
                         embedding_size=embedding_size)

    def network(self, x_input, state_size):
        lstm = tf.nn.rnn_cell.LSTMCell(num_units=state_size)
        raise NotImplementedError

    @property_wrap('_prediction')
    def prediction(self):
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
