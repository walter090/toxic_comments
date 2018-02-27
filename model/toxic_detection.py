import tensorflow as tf

import structure
from .model import Model
from structure import property_wrap


class ToxicityCNN(Model):
    def __init__(self, csvs=None, batch_size=None,
                 num_epochs=None, vocab_size=None, embedding_size=None,
                 num_labels=None, comment_length=None, testing=False,
                 vec=None, layer_config=None, fully_conn_config=None):
        """
        Args:
            csvs: list, a list of strings that are names of csv files to be used
                as dataset.
            batch_size: int, size of each batch.
            num_epochs: int, number of epochs.
            vocab_size: int, vocabulary size for the word embeddings.
            embedding_size: int, size of each word vector.
            num_labels: int, number of labels.
            comment_length: int, length of the each comment.
            vec: list, optional, a numpy array of pre trained word embeddings.
                embedding_size and vocab_size are ignored if this argument is
                provided.
        """
        super(ToxicityCNN, self).__init__(
            csvs=csvs, batch_size=batch_size, num_epochs=num_epochs,
            vocab_size=vocab_size, embedding_size=embedding_size, num_labels=num_labels,
            comment_length=comment_length, testing=testing, vec=vec
        )
        self.layer_config = layer_config
        self.fully_conn_config = fully_conn_config

    def _network(self, x_input, num_output,
                 layer_config=None, fully_conn_config=None, pool='max',
                 name='network', padding='VALID', batchnorm=False,
                 reuse_variables=False):
        """This is where the neural net is implemented. Each of the config is a list,
        each element for one layer. Inception is available by adding more dimensions
        to the config lists. The prediction property calls this function with all its
        default arguments.

        Args:
            x_input: Tensor, input tensor to the network.
            num_output: int, size of final output from the output layer.
            layer_config: list, a list that contains configuration for each layer.
            fully_conn_config: list, a list that contains configuration for each fully connected
                layer.
            pool: string, pooling method.
            name: string, name for the network.
            padding: string, specify padding method used by convolution. Choose from SAME and
                VALID, defaults VALID.
            batchnorm: boolean, set True to use batch normalization.
            reuse_variables: boolean, Set to True to reuse weights and biases.

        Returns:
            output_logits: Tensor, output tensor from the network.
            output: Tensor, output after sigmoid.
            prediction: Tensor, output prediction.
        """

        def pool_size(ksize):
            return self.comment_length - ksize + 1

        with tf.variable_scope(name, reuse=reuse_variables):
            x_input = tf.expand_dims(x_input, -1)

            layer_config = [
                # Convolution layer configuration
                # ksize, stride, out_channels, pool_ksize, pool_stride
                [
                    [2, 1, 256, pool_size(2), 1],
                ],
                [
                    [3, 1, 256, pool_size(3), 1],
                ],
                [
                    [4, 1, 256, pool_size(4), 1],
                ],
                [
                    [5, 1, 256, pool_size(5), 1],
                ],
            ] if not self.layer_config else self.layer_config

            fully_conn_config = [
                [1024, 'lrelu', 0.75],
                [512, 'lrelu', 0.75],
            ] if not self.fully_conn_config else self.fully_conn_config

            outputs = []

            for config_i, config in enumerate(layer_config):
                pool_output = x_input
                for layer_i, layer in enumerate(config):
                    pool_output = structure.conv_pool(
                        pool_output, ksize=[layer[0], self.embedding_size],
                        stride=[1, layer[1]], out_channels=layer[2],
                        pool_ksize=[layer[3], 1],
                        pool_stride=[1, layer[4]], alpha=0.1,
                        padding=padding, batchnorm=batchnorm,
                        method=pool, name='conv_{}_{}'.format(config_i, layer_i))

                outputs.append(pool_output)

        pool_concat = tf.concat(outputs, axis=3)
        output_flat = structure.flatten(pool_concat)

        output_fully_conn = output_flat

        for layer_i, layer in enumerate(fully_conn_config):
            output_fully_conn = structure.fully_conn(output_fully_conn,
                                                     num_output=layer[0],
                                                     activation=layer[1],
                                                     keep_prob=layer[2] if not self.testing else 1.,
                                                     name='fc_{}'.format(layer_i))

        output_logits = structure.fully_conn(output_fully_conn, num_output=num_output,
                                             name='fc_out', activation=None)
        output = tf.nn.sigmoid(output_logits)

        threshold = tf.constant([0.5] * 6)
        condition = tf.less(threshold, output)
        prediction = tf.where(condition, tf.ones_like(output), tf.zeros_like(output))

        return output_logits, output, prediction

    @property_wrap('_prediction')
    def prediction(self):
        self._prediction = self._network(x_input=self.embeddings[1], num_output=self.num_labels,
                                         layer_config=self.layer_config, fully_conn_config=self.fully_conn_config)
        return self._prediction
