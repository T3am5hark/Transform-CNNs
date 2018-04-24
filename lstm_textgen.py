'''
Functionality related to LSTM text modeling and generation.
'''
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.text import text_to_word_sequence
import numpy as np


def pad_list(a_list, list_length, pad_with):
    new_list = a_list
    length = len(a_list)
    if length > list_length:
        new_list = a_list[-list_length:]
    elif length < list_length:
        new_list = [pad_with]*list_length
        new_list[-length:] = a_list
    return new_list

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

    def prompt_input(self, str, seq_length):
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
            network_input.append(char_index)
            network_input = network_input[1:]

        return output_text

    def loop_gen_from_prompt(self, model, generated_seq_length):
        inp = 'nothing'
        print('[Leave prompt blank and hit enter to exit]')
        while len(inp) > 0:
            inp = input('?')
            if len(inp) > input_shape[0]:
                inp = inp[-input_shape[0]:]
            prompt = text_to_train.prompt_input(inp, input_shape[0])
            text = generate_text(model=model,
                                 generated_seq_length=generated_seq_length,
                                 prompt=prompt)
            print(text)


class TextToWordBasedTrainingData(object):

    def __init__(self, raw_text,
                 num_words=None,
                 lower=True,
                 split=' ',
                 retained_symbols=None):

        self.raw_text = raw_text

        self.retained_symbols = retained_symbols
        self.filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'
        self.lower = lower
        self.split=' '
        if self.retained_symbols is not None:
            for char in self.retained_symbols:
                self.raw_text = self.raw_text.replace(char, ' {0} '.format(char))
                self.filters=self.filters.replace(char, '')

        self.tokenizer = Tokenizer(num_words=num_words,
                                   filters=self.filters,
                                   lower=lower,
                                   split=self.split)

        self.tokenizer.fit_on_texts([self.raw_text])
        self.n_vocab = self.tokenizer.num_words
        self.index_to_word = dict((index, word) for word,index in self.tokenizer.word_index.items() )

    def to_word_sequence(self, text=None):
        if text is None:
            text = self.raw_text
        sequence = self.tokenizer.texts_to_sequences([text])[0]
        return sequence

    def get_training_data(self, seq_length):

        sequence = self.to_word_sequence()
        # prepare the dataset of input to output pairs encoded as integers
        X = []
        y = []
        max_index = len(sequence)-seq_length
        for i in range(0, max_index):
            seq_in = sequence[i:i + seq_length]
            seq_out = sequence[i + seq_length]
            X.append(seq_in)
            y.append(seq_out)

        n_patterns = len(X)
        # reshape X to be [samples, time steps, features]
        X = np.reshape(X, (n_patterns, seq_length, 1))
        # normalize
        X = X / float(self.tokenizer.num_words)
        # one hot encode the output variable
        y = np_utils.to_categorical(y)

        return X,y

    def generate_text(model, generated_seq_length, prompt):

        n_outputs = model.output_shape[1]
        n_inputs = model.input_shape[1]
        output_text = prompt
        network_input = self.to_word_sequence(text=prompt.lower())
        network_input = pad_list(network_input, list_length=n_inputs, pad_with=0)

        for i in range(generated_seq_length):
            x = np.reshape(network_input, (1, n_inputs, 1))
            x = x / float(n_outputs)
            prediction = model.predict(x, verbose=0, batch_size=x.shape[0])
            word_index = np.argmax(prediction)
            result = self.index_to_word[word_index]
            output_text += (' ' + result)
            network_input.append(word_index)
            network_input = network_input[1:]

        return output_text

    def loop_gen_from_prompt(self, model, generated_seq_length):
        inp = 'nothing'
        print('[Leave prompt blank and hit enter to exit]')
        while len(inp) > 0:
            prompt = input('?')
            text = self.generate_text(model=model,
                                      generated_seq_length=generated_seq_length,
                                      prompt=prompt)
            print(text)
