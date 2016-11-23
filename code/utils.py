import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import backend as K
from keras.callbacks import Callback
from keras.initializations import uniform
from keras.layers.recurrent import SimpleRNN

START = ''
DATA = "../datasets/"
TEXT8 = DATA + "text8"
MODEL_PATH = DATA + "model.keras"
TRAIN_PERCENT = 0.9

def per_epoch(samples, batch_size):
    with open(TEXT8, "rb") as f:
        data = list(f.readlines()[0])
        return len(data)/samples/batch_size

def ternary_choice(val):
	def init(shape, scale=None, name=None):
		return tf.random_uniform(shape, -val, val)
	return init		

def nary_uniform(shape, scale=1.5, name=None):
    return K.random_uniform_variable(shape, -scale, scale, name=name)

def printProgress(place, place_per, how_often, loss):
    if place % how_often == 0:
        x = place // place_per
        print("{0}% percent through epoch {1}. Loss is {2}.".format(100*place/float(place_per), x, loss))

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

def test_8_generator(CHAR_NUM, NB_SAMPLES):
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
        yield X
        i += 1
        if ((NB_SAMPLES-1)*HOW_FAR + i*CHAR_NUM) > len(char_list) - 1:
            i = 1

def data_target_generator(g, c_to_l, INPUT_DIM):
    while True:
        text = g.next()
        data = one_hot(text, c_to_l, INPUT_DIM)
        X = data[:, :-1, :]
        y = data[:, 1:, :]
        yield (X, y)

class Clockwork(SimpleRNN):
    """From githubnemo"""

    '''Clockwork Recurrent Neural Network - Koutnik et al. 2014.
    A Clockwork RNN splits a SimpleRNN layer into modules of equal size.'''
class Clockwork(SimpleRNN):
    """From githubnemo"""

    '''Clockwork Recurrent Neural Network - Koutnik et al. 2014.
    A Clockwork RNN splits a SimpleRNN layer into modules of equal size.
    Each module is activated during its associated clock period.
    This results in "fast" modules capturing short-term dependencies
    while "slow" modules capture long-term dependencies.
    Periods are assigned in ascending order from left to right.
    Modules are only connected from right to left, allowing
    "slow" modules with larger periods to influence the "faster"
    modules with smaller periods.
    # Arguments
        output_dim: Dimension of internal (module) projections and the
                    final output. Module size is defined by
                    `output_dim / len(periods)`.
        periods: The activation periods of the modules. The number of
                 periods defines the number of modules.
    # References
        - [A Clockwork RNN](http://arxiv.org/abs/1402.3511)
    '''
    def __init__(self, output_dim, periods=[1], **kwargs):
        self.output_dim = output_dim

        assert len(periods) > 0 and output_dim % len(periods) == 0, (
            'Output dimension ({}) must be divisible '.format(output_dim) +
            'by the number of periods ({}) since modules are equally sized '.format(len(periods)) +
            'and each module must have its own period.')
        self.periods = np.asarray(sorted(periods))

        super(Clockwork, self).__init__(output_dim, **kwargs)

    def build(self, input_shape):
        input_dim = input_shape[2]
        n = self.output_dim // len(self.periods)

        module_mask = np.zeros((self.output_dim, self.output_dim), K.floatx())
        periods = np.zeros((self.output_dim,), np.int16)

        for i, t in enumerate(self.periods):
            module_mask[i*n:, i*n:(i+1)*n] = 1
            periods[i*n:(i+1)*n] = t

        module_mask = K.variable(module_mask, name='module_mask')
        self.periods = K.variable(periods, name='periods')

        super(Clockwork, self).build(input_shape)

        # Make sure modules are shortcut from slow to high periods.
        # Placed after super().build since it fills U with values which
        # we want to shadow.
        self.U *= module_mask

        # track previous state as well as time step (for periodic activation)
        if self.stateful:
            self.reset_states()
        else:
            self.states = [None]#, None]

    def get_initial_states(self, x):
        initial_states = super(Clockwork, self).get_initial_states(x)
      #  if self.go_backwards:
      #      input_length = self.input_spec[0].shape[1]
       #     initial_states[-1] = float(input_length)
      #  else:
        #    initial_states[-1] = K.variable(0.)
        return initial_states

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')

        if self.go_backwards:
            initial_time = self.input_spec[0].shape[1]
        else:
            initial_time = 0.

        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
            #K.set_value(self.states[1], initial_time)
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim))]#,
                          # K.variable(initial_time)]


    def get_constants(self, x):
        return super(Clockwork, self).get_constants(x) + [self.periods]

    def step(self, x, states):
        # B_U and B_W are dropout weights
        prev_output, time_step, B_U, B_W, periods = states

        if self.consume_less == 'cpu':
            h = x
        else:
            h = K.dot(x * B_W, self.W) + self.b

        output = self.activation(h + K.dot(prev_output * B_U, self.U))

        # note: switch evaluates the expression for each element so only
        # the modules which period matches the time step is activated here.
        output = K.switch(K.equal(time_step % periods, 0.), output, prev_output)
