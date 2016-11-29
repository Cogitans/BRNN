import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras import backend as K
from keras.callbacks import Callback
from keras.initializations import uniform
from keras.layers.recurrent import SimpleRNN
from scipy.io import wavfile
import time, pickle, os

START = ''
DATA = "../datasets/"
TEXT8 = DATA + "text8"
SHAKESPEARE = DATA + "shakespeare/s.txt"
MODEL_PATH = DATA + "model.keras"
TRAIN_PERCENT = 0.9999

def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)
	return path

def per_epoch(samples, batch_size):
    with open(TEXT8, "rb") as f:
        data = list(f.readlines()[0])
        return len(data)/samples/batch_size

def ternary_choice(val):
    def init(shape, scale=None, name=None):
        mat = np.random.choice([0, -val, val], shape)
        return tf.Variable(mat, dtype=tf.float32, name=name)
    return init

def nary_uniform(shape, scale=1.5, name=None):
    return K.random_uniform_variable(shape, -scale, scale, name=name)

def printProgress(place, place_per, how_often, loss):
    if place % how_often == 0:
        x = place // place_per
	while place > place_per:
		place -= place_per
        print("{0} percent through epoch {1}. Loss is {2}.".format(100*place/float(place_per), x, loss))


def one_hot(text, mapping, num_classes):
    while num_classes % 8 != 0:
	num_classes += 1
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

def text_generator(path, CHAR_NUM, NB_SAMPLES, value = None, percent = None, from_back = False):
    CHAR_NUM = CHAR_NUM + 1
    with open(path, "rb") as f:
        read_data = pickle.load(f)[0]
    char_list = [START] + list(read_data)
    if percent != None:
	if from_back:
		char_list = char_list[-int(len(char_list)*percent):]
	else:
		char_list = char_list[:int(len(char_list)*percent)]
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

def data_len(path, CHAR_NUM, NB_SAMPLES, percent = None):
    with open(path, "rb") as f:
        read_data = list(pickle.load(f)[0])
    if percent is not None:
	read_data = read_data[:int(len(read_data)*percent)]
    return len(read_data) / CHAR_NUM / NB_SAMPLES

def small_generators(BATCH_SIZE, TIMESTEPS):
	nb_classes = 2
	X_train, y_train = load_data("datatraining.txt")
	X_test, y_test = load_data("datatest.txt")
	y_train = to_categorical(y_train, nb_classes)
	y_test = to_categorical(y_test, nb_classes)
	train_g = small_data_generator(X_train, y_train, BATCH_SIZE, TIMESTEPS)
	test_g = small_data_generator(X_test, y_test, BATCH_SIZE, TIMESTEPS)
	return train_g, test_g, X_train.shape, X_test.shape

########################
# FOR MUSIC GENERATION #
########################

def music_generator(path, n_samples, n_timesteps, percent = None, from_back = False):
    n_timesteps += 1
    sample_rate, data = wavfile.read(path)
    nb_p, data_w = data.shape
    if percent is not None:
        if from_back:
            data = data[int(-percent*nb_p):]
        else:
            data = data[:int(percent*nb_p)]
    nb_p, data_w = data.shape
    offset = nb_p / n_samples 
    i = 1
    while True:
        X = np.zeros((n_samples, n_timesteps, data_w))
        for s in np.arange(n_samples):
            X[s, :, :] = data[s*offset + (i-1)*n_timesteps:s*offset + i*n_timesteps, :] 
        yield X 
        if (n_samples-1)*offset + n_timesteps*(i+1) > nb_p:
            i = 1

def music_len(path, n_samples, n_timesteps, percent = None):
    n_timesteps += 1
    sample_rate, data = wavfile.read(path)
    nb_p, data_w = data.shape
    if percent is not None:
        data = data[-int(percent*nb_p):]
    nb_p, data_w = data.shape
    return nb_p / n_samples / n_timesteps, data_w


def music_pair_generator(g):
    while True:
        raw = g.next()
        X = raw[:, :-1, :]
        y = raw[:, 1:, :]
        yield (X, y)