import numpy as np
from keras.utils import Sequence
from text_utils import encode_sequence

class SequenceGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, text, seq_length, mapping, vocab_size,
                 to_fit=True, batch_size=32, shuffle=False):
        self.text = text
        self.seq_length = seq_length
        self.mapping = mapping
        self.vocab_size = vocab_size
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor((len(self.text) - self.seq_length + 1) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X = self._generate_X(indexes)

        if self.to_fit:
            y = self._generate_y(indexes)
            return X, y
        else:
            return X

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.text) - self.seq_length + 1, dtype=np.int32)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _generate_X(self, list_index):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        X = np.empty((self.batch_size, self.seq_length, self.vocab_size),  dtype=np.float64)

        # Generate data
        for i, ID in enumerate(list_index):
            char_seq = self.text[ID:ID+self.seq_length]
            # print(self.seq_length)
            encoded_seq = encode_sequence(self.mapping, char_seq, self.seq_length)[0]
            # print(encoded_seq)
            X[i] = encoded_seq

        return X

    def _generate_y(self, list_index):
        """Generates data containing batch_size masks
        :param list_IDs_temp: list of label ids to load
        :return: batch if masks
        """
        y = np.empty((self.batch_size, self.vocab_size), dtype=np.float64)

        # Generate data
        for i, ID in enumerate(list_index):
            next_char = self.text[ID+self.seq_length]
            encoded_char = encode_sequence(self.mapping, [next_char], self.seq_length)[0][0]
            y[i] = encoded_char

        return y
