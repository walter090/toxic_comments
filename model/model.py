from structure import property_wrap


class Model:
    def __init__(self, vocab_size=None, embedding_size=None):
        """
        Args:
            vocab_size: int, vocabulary size for the word embeddings.
            embedding_size: int, size of each word vector.
        """
        self.embedded = None
        self.embedding_size = embedding_size
        self.global_step = None
        self.vocab_size = vocab_size

        self._embeddings = None
        self._loss = None
        self._metric = None
        self._optimize = None
        self._prediction = None

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

    def file_read_op(self, *args, **kwargs):
        pass

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
