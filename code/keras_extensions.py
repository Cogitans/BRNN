import numpy as np
from keras.callbacks import Callback
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras import backend as K
import matplotlib.pyplot as plt
import time
import tensorflow as tf

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

        module_mask = tf.Variable(module_mask, name='module_mask', trainable=False,dtype = tf.float32)
        self.periods = tf.Variable(periods, name='periods', trainable=False, dtype=tf.float32)

        super(Clockwork, self).build(input_shape)
	
	#####
#        self.input_spec = [InputSpec(shape=input_shape)]
#        if self.stateful:
#            self.reset_states()
#        else:
            # initial states: all-zero tensor of shape (output_dim)
#            self.states = [None]
#        input_dim = input_shape[2]
#        self.input_dim = input_dim
#
#        self.W = self.init((input_dim, self.output_dim),
#                           name='{}_W'.format(self.name))
#        self.U = self.inner_init((self.output_dim, self.output_dim),
#                                 name='{}_U'.format(self.name))
	self.b = self.init((self.output_dim,), name="{}_b".format(self.name))
#
 #       self.regularizers = []
#        if self.W_regularizer:
 #           self.W_regularizer.set_param(self.W)
#            self.regularizers.append(self.W_regularizer)
 #       if self.U_regularizer:
  #          self.U_regularizer.set_param(self.U)
   #         self.regularizers.append(self.U_regularizer)
    #    if self.b_regularizer:
     #       self.b_regularizer.set_param(self.b)
      #      self.regularizers.append(self.b_regularizer)

       # self.trainable_weights = [self.W, self.U, self.b]

        #if self.initial_weights is not None:
         #   self.set_weights(self.initial_weights)
          #  del self.initial_weights
	#####

        # Make sure modules are shortcut from slow to high periods.
        # Placed after super().build since it fills U with values which
        # we want to shadow.
        self.U *= module_mask

        # track previous state as well as time step (for periodic activation)
        if self.stateful:
            self.reset_states()
        else:
            self.states = [None, None]

    def get_initial_states(self, x):
        initial_states = super(Clockwork, self).get_initial_states(x)
        if self.go_backwards:
           input_length = self.input_spec[0].shape[1]
           initial_states[-1] = float(input_length)
        else:
           initial_states[-1] = tf.Variable(np.zeros(initial_states[0].get_shape()), name="Timestep",dtype=tf.float32,trainable=False)
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
            K.set_value(self.states[1], 
                        np.full(((input_shape[0], self.output_dim)), initial_time))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim)),
                          K.variable(np.full(((input_shape[0], self.output_dim)), initial_time))]

    def preprocess_input(self, x):
	return x


    def get_constants(self, x):
        return super(Clockwork, self).get_constants(x) + [self.periods]

    def step(self, x, states):
        # B_U and B_W are dropout weights
        prev_output, time_step, B_U, B_W, periods = states
        time = tf.cast(time_step[0, 0], tf.int32)
        h = K.dot(x * B_W, self.W) + self.b

        output = self.activation(h + K.dot(prev_output * B_U, self.U))
        # note: switch evaluates the expression for each element so only
        # the modules which period matches the time step is activated here.
 	with tf.device('/cpu:0'):
 	       output = tf.select(tf.pack([K.equal(tf.cast(tf.mod(time, tf.cast(periods, tf.int32)), tf.float32), 0.0)]*output.get_shape()[0]), output, prev_output)
        return output, [output, time_step + 1]

class MUT2(GRU):

    def preprocess_input(self, x):
        return x

    def step(self, x, states):
        h_tm1 = states[0]  # previous memory
        B_U = states[1]  # dropout matrices for recurrent units
        B_W = states[2]

        x_z = K.dot(x, self.W_z) + K.dot(h_tm1, self.U_z) + self.b_z
        x_r = x + K.dot(h_tm1, self.U_r) + self.b_r
        z = K.sigmoid(x_z)
        r = K.sigmoid(x_r)
        h_temp = K.tanh(K.dot(r * h_tm1, self.U_h) + K.dot(x, self.W_h) + self.b_h)
        h = h_temp * z + h_tm1 * (1 - z)
        return h, [h]

class MUT1(GRU):

    def preprocess_input(self, x):
        return x

    def step(self, x, states):
        h_tm1 = states[0]  # previous memory
        B_U = states[1]  # dropout matrices for recurrent units
        B_W = states[2]

        x_z = K.dot(x, self.W_z) + self.b_z
        x_r = K.dot(x, self.W_r) + K.dot(h_tm1, self.U_r) + self.b_r
        z = K.sigmoid(x_z)
        r = K.sigmoid(x_r)
        h_temp = K.tanh(K.dot(r * h_tm1, self.U_h) + K.tanh(x) + self.b_h)
        h = h_temp * z + h_tm1 * (1 - z)
        return h, [h]
