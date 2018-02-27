import tensorflow as tf

from structure import property_wrap
from .model import Model


class ToxicityLSTM(Model):
    def __init__(self, csvs=None, batch_size=None,
                 num_epochs=None, vocab_size=None, embedding_size=None,
                 num_labels=None, comment_length=None, testing=False,
                 vec=None):
        super().__init__(
            csvs=csvs, batch_size=batch_size, num_epochs=num_epochs,
            vocab_size=vocab_size, embedding_size=embedding_size, num_labels=num_labels,
            comment_length=comment_length, testing=testing, vec=vec
        )

    def network(self, x_input, len_sequence,
                batch_size, num_classes, state_size,
                peepholes=False, name='network', num_proj=None,
                num_layers=2, keep_prob=0.5):
        with tf.variable_scope(name):
            weights = tf.get_variable(shape=[state_size, num_classes],
                                      initializer=tf.random_normal_initializer(),
                                      name='weights')
            bias = tf.get_variable(shape=[num_classes],
                                   initializer=tf.zeros_initializer(),
                                   name='bias')

            cell = tf.nn.rnn_cell.LSTMCell(num_units=state_size, use_peepholes=peepholes,
                                           num_proj=num_proj)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, input_keep_prob=keep_prob,
                                                 output_keep_prob=keep_prob)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)

            init_state = cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            sequence_length = tf.cast([len_sequence] * self.batch_size, dtype=tf.int16)

            outputs, state = tf.nn.dynamic_rnn(cell=cell, inputs=x_input,
                                               sequence_length=sequence_length, initial_state=init_state)
            outputs = tf.reshape(outputs, [-1, state_size])

            logits = tf.matmul(outputs, weights)
            logits = tf.nn.bias_add(logits, bias=bias)

            output = tf.nn.sigmoid(logits)

            threshold = tf.constant([0.5] * num_classes)
            condition = tf.less(threshold, output)
            prediction = tf.where(condition, tf.ones_like(output), tf.zeros_like(output))

            return logits, condition, prediction

    @property_wrap('_prediction')
    def prediction(self):
        self._prediction = self.network(x_input=self.embeddings[1], len_sequence=self.comment_length,
                                        batch_size=self.batch_size, num_classes=self.num_labels,
                                        state_size=256)
        return self._prediction