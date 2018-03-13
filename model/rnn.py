import tensorflow as tf

import structure
from structure import property_wrap
from .model import Model


class ToxicityLSTM(Model):
    def __init__(self, csvs=None, batch_size=None,
                 num_epochs=None, vocab_size=None, embedding_size=None,
                 num_labels=None, comment_length=None, testing=False,
                 vec=None, peepholes=False, bi=True,
                 keep_prob=0.5, num_layers=None, attention=False):
        super().__init__(
            csvs=csvs, batch_size=batch_size, num_epochs=num_epochs,
            vocab_size=vocab_size, embedding_size=embedding_size, num_labels=num_labels,
            comment_length=comment_length, testing=testing, vec=vec,
            keep_prob=keep_prob
        )
        self.peepholes = peepholes
        self.bi = bi
        self.num_layers = num_layers
        self.attention = attention

    @staticmethod
    def _create_cell(num_layers, state_size, keep_prob, peepholes):
        """Function for creating a lstm cell.

        Args:
            num_layers: int, number of stacked lstm layers.
            state_size: int, size of state.
            keep_prob: float, keep probability for dropout.
            peepholes: bool, set True to use peephole connections.

        Returns:
            Tensor, lstm cell.
        """
        cell = tf.nn.rnn_cell.LSTMCell(num_units=state_size, use_peepholes=peepholes)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, state_keep_prob=keep_prob,
                                             output_keep_prob=keep_prob, input_keep_prob=keep_prob)
        if num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

        return cell

    @staticmethod
    def _create_birnn(cell_fw, cell_bw, init_state_fw,
                      init_state_bw, sequence_length, inputs):
        """Create a bidirectional RNN

        Args:
            cell_fw: Forward cell.
            cell_bw: Backward cell.
            init_state_fw: Forward initial state.
            init_state_bw: Backward initial state.
            sequence_length: Tensor, sequence length.
            inputs: Tensor, input to network.

        Returns:
            outputs: tuple, forward and backward tensors.
        """
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw, cell_bw=cell_bw,
            initial_state_fw=init_state_fw, initial_state_bw=init_state_bw,
            sequence_length=sequence_length, inputs=inputs
        )
        return outputs

    def _network(self, x_input, len_sequence,
                 batch_size, num_classes, state_size,
                 peepholes=False, name='network', num_layers=2,
                 keep_prob=0.5, bi=True):
        """Build network

        Args:
            x_input: Tensor, input to network.
            len_sequence: int, length of each sequence.
            batch_size: int, size of each mini-batch.
            num_classes: int, number of labels.
            state_size: int, size of hidden state.
            peepholes: bool, set True to use peephole connections.
            name: str, name for the variable scope.
            num_layers: int, number of layer for stacked lstm.
            keep_prob: float, keep probability for dropout.
            bi: bool, set True to use BiRNN.

        Returns:
            logits: logit output from network.
            output: activated logits.
            prediction: prediction from network.
        """
        with tf.variable_scope(name):
            if self.testing:
                keep_prob = 1.

            sequence_length = tf.cast([len_sequence] * batch_size, dtype=tf.int32)

            if bi:
                weights = tf.get_variable(shape=[state_size * 2, num_classes],
                                          initializer=tf.random_normal_initializer(),
                                          name='weights')
                bias = tf.get_variable(shape=[num_classes],
                                       initializer=tf.zeros_initializer(),
                                       name='bias')

                cell_fw = self._create_cell(num_layers=num_layers, state_size=state_size,
                                            keep_prob=keep_prob, peepholes=peepholes)
                cell_bw = self._create_cell(num_layers=num_layers, state_size=state_size,
                                            keep_prob=keep_prob, peepholes=peepholes)
                init_state_fw = cell_fw.zero_state(batch_size=batch_size, dtype=tf.float32)
                init_state_bw = cell_fw.zero_state(batch_size=batch_size, dtype=tf.float32)
                output_fw, output_bw = self._create_birnn(cell_fw=cell_fw, cell_bw=cell_bw,
                                                          init_state_fw=init_state_fw, init_state_bw=init_state_bw,
                                                          sequence_length=sequence_length, inputs=x_input)

                outputs = tf.stack([output_fw, output_bw], axis=-1)
            else:
                weights = tf.get_variable(shape=[state_size, num_classes],
                                          initializer=tf.random_normal_initializer(),
                                          name='weights')
                bias = tf.get_variable(shape=[num_classes],
                                       initializer=tf.zeros_initializer(),
                                       name='bias')

                cell = self._create_cell(num_layers=num_layers, state_size=state_size,
                                         keep_prob=keep_prob, peepholes=peepholes)

                init_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)

                outputs, _ = tf.nn.dynamic_rnn(cell=cell, inputs=x_input,
                                               sequence_length=sequence_length, initial_state=init_state)

                outputs = tf.reshape(outputs, [-1, state_size])

            last_indices = tf.range(0, batch_size) * len_sequence + (sequence_length - 1)
            last_hidden = tf.gather(outputs, last_indices)

            if self.attention:
                outputs_in_batch = tf.reshape(outputs, shape=[batch_size, -1, state_size * 2 if bi else state_size])
                attention_weights = structure.weigh_attention(outputs_in_batch)
                last_hidden = structure.get_context_vector(source_hidden=outputs_in_batch,
                                                           attention_weights=attention_weights)

            logits = tf.matmul(last_hidden, weights)
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
                                         peepholes=self.peepholes, bi=self.bi, num_layers=self.num_layers)
        return self._prediction
