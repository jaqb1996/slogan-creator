from keras.models import Sequential
from keras.layers import Dense, Activation, SimpleRNN, LSTM, Dropout
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import random
import pandas as pd


def sample(preds, temperature=1.0):
    """
        sampling index from a probability array

        :param preds: model.predict
        :param temperature: defines the freedom the function has when creating text.
        :return: sample an index from the output(probability array)
        """
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)



def generate_slogan(model, all_slogans_as_text, maxlen, chars, char_indices, indices_char, length, diversity, end_after_pipe_character = True):
    """
        Function invoked at end of each epoch. Prints generated text.

        :param model: model
        :param all_slogans_as_text: text
        :param maxlen: length of subsequences
        :param chars: list of all unique chars
        :param char_indices: mapping from character to integer
        :param indices_char: mapping from integer to character
        :param length: maximum slogan length
        :param diversity: defines the freedom the function has when creating text.
        :param end_after_pipe_character: should generated slogan end if certain char appears
        :return: generated text
        """
    num_pipes = all_slogans_as_text.count('|')
    end_index = random.randint(3, num_pipes)

    def find(str, ch):
        for i, ltr in enumerate(str):
            if ltr == ch:
                yield i

    end_index = list(find(all_slogans_as_text, '|'))[end_index]

    sentence = all_slogans_as_text[end_index - maxlen + 1:end_index + 1]
    generated = ''
    for i in range(length):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.
            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            if next_char == '|' and end_after_pipe_character:
                return generated

            generated += next_char
            sentence = sentence[1:] + next_char
    return generated


def get_data(filepath):
    """
        loading in the data and create an mapping from character to integer and integer to character
        :param filepath: filepath
        :return: char_indices, indices_char, text, chars, maxlen
        """
    df = pd.read_csv(filepath, sep="\n", header=0)
    slogan_lengths = []
    for item in df['Slogans']:
        slogan_lengths.append(len(item))

    text = ""
    for item in df['Slogans']:
        text += "".join(item)
        text += "|"

    chars = sorted(set(text))

    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    maxlen = round(np.mean(slogan_lengths))
    return char_indices, indices_char, text, chars, maxlen


def get_x_y(text, maxlen, chars):
    """
        Splitting data up into subsequences with the given length, then transforming it into boolean array.
        :param text: all slogans as text
        :param maxlen: length of subsequences
        :param chars: list of all unique chars
        :return: x, y
        """
    step = 5
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
    return x, y


def network_model(x, y, epochs):
    """
        Creating and training model

        :param x: sequence of characters
        :param y: sequence of characters
        :param epochs: number of epochs
        :return: model
        """
    model = Sequential()
    model.add(LSTM(128, input_shape=(maxlen, len(chars)), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))
    optimizer = RMSprop(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    filepath = "test.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss',
                                 verbose=1, save_best_only=True,
                                 mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                                  patience=1, min_lr=0.001)
    history = model.fit(x, y, batch_size=128, epochs=epochs, callbacks=[checkpoint, reduce_lr])
    return model, history


def plotLoss(history):
    """
    plots graph of model loss change in epochs
    """
    import matplotlib.pyplot as plt
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()


if __name__ == "__main__":

    number_of_slogans = 10
    max_slogan_length = 40
    diversity = 0.4
    epochs = 100
    word = "ABC"
    char_indices, indices_char, text, chars, maxlen = get_data('data.txt')
    x, y = get_x_y(text, maxlen, chars)
    model, history = network_model(x, y, epochs)
    plotLoss(history)
    for _ in range(number_of_slogans):
        print(generate_slogan(model, text, maxlen, chars, char_indices, indices_char, max_slogan_length, diversity).replace("*", word))
