import numpy as np
import tensorflow as tf
from utils import *
from quantifications import *
from keras_extensions import *
from keras.layers.core import Dense, TimeDistributedDense
from keras import backend as K
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers.wrappers import TimeDistributed
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy as accuracy
import time
import pickle
import matplotlib.pyplot as plt
sess = tf.Session()
K.set_session(sess)
K.manual_variable_initialization(True)

#### Constants Definition ####

DATA = "../datasets/"
SAVE = "../results/"
TEXT = DATA + "shakespeare/s.txt"
MODEL_PATH = DATA + "model.keras"
SAVE_PATH = SAVE + "saved_quick.tf"
LOSS_PATH = mkdir(SAVE + "shakeweight/")

batch_size = 512
HIDDEN_SIZE = 516
num_batch = None
how_often = 50


##############################

#### Data Establishing ####
def run(LR, val, RNN_TYPE, TIMESTEPS = 1, quant = None, GPU_FLAG=True, NUM_EPOCH = 1, NUM_BATCH = None, SAVE_WEIGHTS = False, VERBOSE = True, WHICH = None):
	if quant is None: assert val is np.inf
	quantify = identity if quant is None else quant(val)
	num_timesteps = TIMESTEPS
	num_classes, c_to_l, l_to_c = p_char_mapping(TEXT)

	t_g = text_generator(TEXT, num_timesteps, batch_size, percent = .95)
	test_g = text_generator(TEXT, num_timesteps, batch_size, percent = .05, from_back = True)

	x_y_generator = data_target_generator(t_g, c_to_l, num_classes)
	test_generator = data_target_generator(test_g, c_to_l, num_classes)

	num_batch_in_epoch = data_len(TEXT, batch_size, num_timesteps, percent = .95)
	num_test = data_len(TEXT, batch_size, num_timesteps, percent = .05)
	num_batch = NUM_EPOCH * num_batch_in_epoch if not NUM_BATCH else NUM_BATCH
	how_often = num_batch_in_epoch // 100
	
	_init = "he_normal" if val == np.inf else ternary_choice(val)
	i_init = "identity" 

	loaded_weights = pickle.load(open("../results/weights/weights.weights"))[0]
	###########################

	#### Model Definition ####

	i = tf.placeholder(tf.float32, shape=(batch_size, num_timesteps, num_classes+1), name="X")
	labels = tf.placeholder(tf.float32, shape=(batch_size, num_timesteps, num_classes+1), name="y")
	with tf.device('/gpu:0') if GPU_FLAG else tf.device('/cpu:0'):
		if RNN_TYPE == Clockwork:
			layer1 = RNN_TYPE(HIDDEN_SIZE, init = _init, inner_init= i_init, periods=[1, 2, 4, 8, 16, 32], stateful=True, return_sequences=True)
			h1 = layer1(i)
		else:
			layer1 = RNN_TYPE(HIDDEN_SIZE, init = _init, inner_init = i_init ,stateful=True, return_sequences=True)
			h1 = layer1(i)
	with tf.device('/gpu:1') if GPU_FLAG else tf.device('/cpu:0'):
		if RNN_TYPE == Clockwork:
			layer2 = RNN_TYPE(HIDDEN_SIZE, init = _init, inner_init= i_init, periods=[1, 2, 4, 8, 16, 32], stateful=True, return_sequences=True)
			h2 = layer2(h1)
		else:
			layer2 = RNN_TYPE(HIDDEN_SIZE, init= _init, inner_init= i_init, stateful=True, return_sequences=True)
			h2 = layer2(h1)
	with tf.device('/gpu:2') if GPU_FLAG else tf.device('/cpu:0'):
		layer3 = TimeDistributed(Dense(num_classes+1, init = _init))
		o = layer3(h2)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(o, labels))
	acc_value = accuracy(labels, o)	

	update_ops = []
	for old_value, new_value in layer1.updates + layer2.updates:
		update_ops.append(tf.assign(old_value, new_value).op)

	name_v = {v.name: v for v in tf.trainable_variables()}
	intitial_assignments = []
	for (name, val) in loaded_weights:
		intitial_assignments.append(name_v[name].assign_add(tf.constant(val)))

	layer_2_activations = []


	init_op = tf.initialize_all_variables()
	with sess.as_default():
		init_op.run()
		sess.run(intitial_assignments)
		for batch in np.arange(num_batch):
			X, y = x_y_generator.next()
			acts = h2.eval(feed_dict={i: X})
			layer_2_activations.append(acts)
			sess.run(update_ops, feed_dict={i: X})
	with open(LOSS_PATH + "layer_2.a", "wb"):
		pickle.dump([layer_2_activations], f)

run(np.inf, np.inf, GRU, quant=None, GPU_FLAG = False, NUM_EPOCH = 1)

