import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras import backend as K
from keras.callbacks import Callback
from keras.initializations import uniform
from keras.layers.recurrent import SimpleRNN
import time, pickle

START = ''
DATA = "../datasets/"
TEXT8 = DATA + "text8"
SHAKESPEARE = DATA + "shakespeare/s.txt"
MODEL_PATH = DATA + "model.keras"
TRAIN_PERCENT = 0.9999

def per_epoch(samples, batch_size):
    with open(TEXT8, "rb") as f:
        data = list(f.readlines()[0])
        return len(data)/samples/batch_size

def ternary_choice(val):
    def init(shape, scale=None, name=None):
        mat = np.random.choice([0, -val, val], shape)
        return tf.Variable(mat, dtype=tf.float32)
    return init

def nary_uniform(shape, scale=1.5, name=None):
    return K.random_uniform_variable(shape, -scale, scale, name=name)

def printProgress(place, place_per, how_often, loss):
    if place % how_often == 0:
        x = place // place_per
        print("{0} percent through epoch {1}. Loss is {2}.".format(100*place/float(place_per), x, loss))

def trinaryQuant(x):
    orig = x
    signs = np.sign(x)
    def hardSigmoid(x):
        return np.clip((x + 1)/2, 0, 1)
    x = hardSigmoid(np.abs(x))
    probs = np.random.rand(*x.shape)
    x[x < probs] = 0
    x[x >= probs] = 1
    return x * signs

def identity(x):
    return x

def det_tri_quant(x):
    x[x <= -.5] = -1
    x[x >= .5] = 1
    x[(-.5 < x) * (x < .5)] = 0
    return x

def one_hot(text, mapping, num_classes):
    data = np.zeros((text.shape[0], text.shape[1], num_classes))
    for (x,y), value in np.ndenumerate(text):
        data[x, y, mapping[value]] = 1
    return data

def sample(probs):
    a = np.exp(probs)/np.sum(np.exp(probs))
    return np.argmax(np.random.multinomial(1, a, 1))

def char_mapping(path):
    with open(path, "rb") as f:
        read_data = f.readlines()
    chars = [START] + list(set(read_data[0]))
    char_to_labels = {ch:i for i, ch in enumerate(chars)}
    labels_to_char = {i:ch for i, ch in enumerate(chars)}
    return len(chars), char_to_labels, labels_to_char

def p_char_mapping(path):
    with open(path, "rb") as f:
        read_data = pickle.load(f)[0]
    chars = [START] + list(set(read_data))
    char_to_labels = {ch:i for i, ch in enumerate(chars)}
    labels_to_char = {i:ch for i, ch in enumerate(chars)}
    return len(chars), char_to_labels, labels_to_char

def text_8_generator(CHAR_NUM, NB_SAMPLES):
    CHAR_NUM = CHAR_NUM + 1
    with open(TEXT8, "rb") as f:
        read_data = f.readlines()
    char_list = [START] + list(read_data[0])
    char_list = char_list[:int(TRAIN_PERCENT*len(char_list))]
    HOW_FAR = len(char_list)/NB_SAMPLES
    char_list = np.array(char_list)
    i = 1
    while True:
        X = np.zeros((NB_SAMPLES, CHAR_NUM), dtype='|S1')
        for s in np.arange(NB_SAMPLES):
            X[s, :] = char_list[s*HOW_FAR + (i-1)*CHAR_NUM:s*HOW_FAR + i*CHAR_NUM]
        yield X
        i += 1
        if ((NB_SAMPLES-1)*HOW_FAR + i*CHAR_NUM) > len(char_list) - 1:
            i = 1

def test_8_generator(CHAR_NUM, NB_SAMPLES, value = None):
    CHAR_NUM = CHAR_NUM + 1
    with open(TEXT8, "rb") as f:
        read_data = f.readlines()
    char_list = [START] + list(read_data[0])
    char_list = char_list[int(TRAIN_PERCENT*len(char_list))+1:]
    HOW_FAR = len(char_list)/NB_SAMPLES
    char_list = np.array(char_list)
    i = 1
    while True:
        X = np.zeros((NB_SAMPLES, CHAR_NUM), dtype='|S1')
        for s in np.arange(NB_SAMPLES):
            X[s, :] = char_list[s*HOW_FAR + (i-1)*CHAR_NUM:s*HOW_FAR + i*CHAR_NUM]
        i += 1
        if value:
            value = yield X, (((NB_SAMPLES-1)*HOW_FAR + i*CHAR_NUM) > len(char_list) - 1)
        else:
            value = yield X
        if ((NB_SAMPLES-1)*HOW_FAR + i*CHAR_NUM) > len(char_list) - 1:
            i = 1

def data_target_generator(g, c_to_l, INPUT_DIM, value = None):
    while True:
        if value:
            text, done = g.send(value)
        else:
            text = g.send(value)
        data = one_hot(text, c_to_l, INPUT_DIM)
        X = data[:, :-1, :]
        y = data[:, 1:, :]
        if value:
            value = yield (X, y), done
        else:
            value = yield (X, y)

class Timer:

    def __init__(self):
        self.silent = False

    def tic(self, msg = None): 
        self.t = time.time()
        self.ticked = True
        self.msg = msg

    def toc(self):
        if not self.ticked:
            raise Exception()
        if not self.silent:
            if not self.msg:
                print("Time elapsed: {0}".format(time.time() - self.t))
            else:
                print("Time elapsed since {0}: {1}".format(self.msg, time.time() - self.t))
        self.ticked = False
        self.msg = None

    def silence(self):
        self.silent = not self.silent

T = Timer()

def text_generator(path, CHAR_NUM, NB_SAMPLES, value = None):
    CHAR_NUM = CHAR_NUM + 1
    with open(path, "rb") as f:
        read_data = pickle.load(f)[0]
    char_list = [START] + list(read_data)
    HOW_FAR = len(char_list)/NB_SAMPLES
    char_list = np.array(char_list)
    i = 1
    while True:
        X = np.zeros((NB_SAMPLES, CHAR_NUM), dtype='|S1')
        for s in np.arange(NB_SAMPLES):
            X[s, :] = char_list[s*HOW_FAR + (i-1)*CHAR_NUM:s*HOW_FAR + i*CHAR_NUM]
        yield X
        i += 1
        if ((NB_SAMPLES-1)*HOW_FAR + i*CHAR_NUM) > len(char_list) - 1:
            i = 1

def data_len(path, CHAR_NUM, NB_SAMPLES):
    with open(path, "rb") as f:
        read_data = pickle.load(f)[0]
    return len(list(read_data)) / CHAR_NUM / NB_SAMPLES

def small_generators(BATCH_SIZE, TIMESTEPS):
	nb_classes = 2
	X_train, y_train = load_data("datatraining.txt")
	X_test, y_test = load_data("datatest.txt")
	y_train = to_categorical(y_train, nb_classes)
	y_test = to_categorical(y_test, nb_classes)
	train_g = small_data_generator(X_train, y_train, BATCH_SIZE, TIMESTEPS)
	test_g = small_data_generator(X_test, y_test, BATCH_SIZE, TIMESTEPS)
	return train_g, test_g, X_train.shape, X_test.shape
