import tensorflow as tf

from structure import property_wrap
from .model import Model


class ToxicityLSTM(Model):
    def __init__(self, csvs=None, batch_size=None,
                 num_epochs=None, vocab_size=None, embedding_size=None,
                 num_labels=None, comment_length=None, testing=False,
                 vec=None, peepholes=False, bi=True,
                 keep_prob=0.5):
        super().__init__(
            csvs=csvs, batch_size=batch_size, num_epochs=num_epochs,
            vocab_size=vocab_size, embedding_size=embedding_size, num_labels=num_labels,
            comment_length=comment_length, testing=testing, vec=vec,
            keep_prob=keep_prob
        )
        self.peepholes = peepholes
        self.bi = bi

    @staticmethod
    def _create_cell(num_layers, state_size, keep_prob, peepholes):
        cell = tf.nn.rnn_cell.LSTMCell(num_units=state_size, use_peepholes=peepholes)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, input_keep_prob=keep_prob,
                                             output_keep_prob=keep_prob)
        if num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

        return cell

    @staticmethod
    def _create_birnn(cell_fw, cell_bw, init_state_fw,
                      init_state_bw, sequence_length, inputs):
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw, cell_bw=cell_bw,
            initial_state_fw=init_state_fw, initial_state_bw=init_state_bw,
            sequence_length=sequence_length, inputs=inputs
        )
        outputs = tf.concat(outputs, 2)
        return outputs

    def _network(self, x_input, len_sequence,
                 batch_size, num_classes, state_size,
                 peepholes=False, name='network', num_layers=2,
                 keep_prob=0.5, bi=True):
        with tf.variable_scope(name):
            if self.testing:
                keep_prob = 1.

            sequence_length = tf.cast([len_sequence] * batch_size, dtype=tf.int32)

            weights = tf.get_variable(shape=[state_size, num_classes],
                                      initializer=tf.random_normal_initializer(),
                                      name='weights')
            bias = tf.get_variable(shape=[num_classes],
                                   initializer=tf.zeros_initializer(),
                                   name='bias')

            if bi:
                cell_fw = self._create_cell(num_layers=num_layers, state_size=state_size,
                                            keep_prob=keep_prob, peepholes=peepholes)
                cell_bw = self._create_cell(num_layers=num_layers, state_size=state_size,
                                            keep_prob=keep_prob, peepholes=peepholes)
                init_state_fw = cell_fw.zero_state(batch_size=batch_size, dtype=tf.float32)
                init_state_bw = cell_fw.zero_state(batch_size=batch_size, dtype=tf.float32)
                outputs = self._create_birnn(cell_fw=cell_fw, cell_bw=cell_bw,
                                             init_state_fw=init_state_fw, init_state_bw=init_state_bw,
                                             sequence_length=sequence_length, inputs=x_input)
            else:
                cell = self._create_cell(num_layers=num_layers, state_size=state_size,
                                         keep_prob=keep_prob, peepholes=peepholes)

                init_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)

                outputs, _ = tf.nn.dynamic_rnn(cell=cell, inputs=x_input,
                                               sequence_length=sequence_length, initial_state=init_state)

            outputs = tf.reshape(outputs, [-1, state_size])

            last = tf.range(0, batch_size) * len_sequence + (sequence_length - 1)
            outputs = tf.gather(outputs, last)

            logits = tf.matmul(outputs, weights)
            logits = tf.nn.bias_add(logits, bias=bias)

            output = tf.nn.sigmoid(logits)

            threshold = tf.constant([0.5] * num_classes)
            condition = tf.less(threshold, output)
            prediction = tf.where(condition, tf.ones_like(output), tf.zeros_like(output))

            return logits, output, prediction

    @property_wrap('_prediction')
    def prediction(self):
        self._prediction = self._network(x_input=self.embeddings[1], len_sequence=self.comment_length,
                                         batch_size=self.batch_size, num_classes=self.num_labels,
                                         state_size=self.embedding_size, keep_prob=self.keep_prob,
                                         peepholes=self.peepholes, bi=self.bi)
        return self._prediction
