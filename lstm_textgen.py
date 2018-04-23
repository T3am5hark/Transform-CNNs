'''
Functionality related to LSTM text modeling and generation.
'''
from keras.utils import np_utils
import numpy as np

class TextToTrainingData(object):
    '''
    class TextToTrainingData(object)

    Description:

        Provides methods for transforming raw text into training data sets for LSTMs.
        Convenience functions for converting text into training data and using LSTM
        for recursive text generation after training the model.
    '''

    def __init__(self, raw_text):
        self.raw_text = raw_text
        self.chars = sorted(list(set(raw_text)))
        self.char_to_int = dict((c, i) for i,c in enumerate(self.chars))
        self.int_to_char = dict((i, c) for i,c in enumerate(self.chars))

        self.n_chars = len(self.raw_text)
        self.n_vocab = len(self.chars)

    def get_training_data(self, seq_length):
        # prepare the dataset of input to output pairs encoded as integers
        X = []
        y = []
        max_index = self.n_chars-seq_length
        for i in range(0, max_index, 1):
            seq_in = self.raw_text[i:i + seq_length]
            seq_out = self.raw_text[i + seq_length]
            X.append([self.char_to_int[char] for char in seq_in])
            y.append(self.char_to_int[seq_out])

        n_patterns = len(X)
        # reshape X to be [samples, time steps, features]
        X = np.reshape(X, (n_patterns, seq_length, 1))
        # normalize
        X = X / float(self.n_vocab)
        # one hot encode the output variable
        y = np_utils.to_categorical(y)

        return X,y

    def str_to_int_sequence(self, a_string):
        return [self.char_to_int[char] for char in a_string]

    def prompt_input(str, seq_length):
        return str.rjust(seq_length)

    def generate_text(self, model, generated_seq_length, prompt):

        n_outputs = model.output_shape[1]
        output_text = prompt
        network_input = self.str_to_int_sequence(prompt.lower())

        for i in range(generated_seq_length):
            x = np.reshape(network_input, (1, len(network_input), 1))
            x = x / float(n_outputs)
            prediction = model.predict(x, verbose=0, batch_size=x.shape[0])
            char_index = np.argmax(prediction)
            result = self.int_to_char[char_index]
            output_text += result
            #seq_in = [self.int_to_char[value] for value in network_input]
            network_input.append(index)
            network_input = network_input[1:]

        return output_text





