from keras.models import Sequential
from keras.layers import Dense, Activation, SimpleRNN, LSTM
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import random
import pandas as pd


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_text(model, all_slogans_as_text, maxlen, chars, char_indices, indices_char, length, diversity, end_after_pipe_character = True):
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


def get_saved_model(maxlen, chars, filepath):
    model = Sequential()
    model.add(LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.01)
    model.load_weights(filepath)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model


def get_data(filepath):
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
    return x,y


def network_model(x, y, epochs):
    model = Sequential()
    model.add(LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    filepath = "test.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss',
                                 verbose=1, save_best_only=True,
                                 mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                                  patience=1, min_lr=0.001)
    model.fit(x, y, batch_size=128, epochs=epochs, callbacks=[checkpoint, reduce_lr])
    return model


if __name__ == "__main__":

    number_of_slogans = 10
    max_slogan_length = 30
    diversity = 0.1
    epochs = 15
    word = "ABC"
    char_indices, indices_char, text, chars, maxlen = get_data('data1.txt')
    x, y = get_x_y(text, maxlen, chars)
    model = network_model(x, y, epochs)

    for _ in range(number_of_slogans):
        print(generate_text(model, text, maxlen, chars, char_indices, indices_char, max_slogan_length, diversity).replace("*", word))
