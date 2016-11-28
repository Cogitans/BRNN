import numpy as np
import tensorflow as tf
from utils import *
from quantifications import *
from keras_extensions import *
from keras.layers.core import Dense
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
LOSS_PATH = mkdir(SAVE + "shakespeare_base2_det/")

batch_size = 512
HIDDEN_SIZE = 516
num_batch = None
how_often = 50


##############################

#### Data Establishing ####
def run(LR, val, RNN_TYPE, TIMESTEPS = 128, GPU_FLAG=True, NUM_EPOCH = 1, NUM_BATCH = None, SAVE_WEIGHTS = False, VERBOSE = True):
	quantify = identity if val == np.inf else deterministic_ternary(val)
	num_timesteps = TIMESTEPS
	num_classes, c_to_l, l_to_c = p_char_mapping(TEXT)

	t_g = text_generator(TEXT, num_timesteps, batch_size, percent = .95)
	test_g = text_generator(TEXT, num_timesteps, batch_size, percent = .05, from_back = True)

	x_y_generator = data_target_generator(t_g, c_to_l, num_classes)
	test_generator = data_target_generator(test_g, c_to_l, num_classes)

	num_batch_in_epoch = data_len(TEXT, batch_size, num_timesteps, percent = .95)
	num_test = data_len(TEXT, batch_size, num_timesteps, percent = .05)
	num_batch = NUM_EPOCH * num_batch_in_epoch if not NUM_BATCH else NUM_BATCH
	how_often = num_batch_in_epoch // 4
	
	_init = ternary_choice(val) if val is not np.inf else "he_normal"
	i_init = ternary_choice(val) if val is not np.inf else "identity"
	###########################

	#### Model Definition ####

	i = tf.placeholder(tf.float32, shape=(batch_size, num_timesteps, num_classes+1), name="X")
	labels = tf.placeholder(tf.float32, shape=(batch_size, num_timesteps, num_classes+1), name="y")
	with tf.device('/gpu:0'):# if RNN_TYPE != Clockwork and GPU_FLAG else tf.device('/cpu:0'):
		if RNN_TYPE == Clockwork:
			layer1 = RNN_TYPE(HIDDEN_SIZE, init = _init, inner_init= i_init, periods=[1, 2, 4, 8, 16, 32], stateful=True, return_sequences=True)
			h1 = layer1(i)
		else:
			layer1 = RNN_TYPE(HIDDEN_SIZE, init = _init, inner_init = i_init ,stateful=True, return_sequences=True)
			h1 = layer1(i)
	with tf.device('/gpu:1'):# if RNN_TYPE != Clockwork and GPU_FLAG else tf.device('/cpu:0'):
		if RNN_TYPE == Clockwork:
			layer2 = RNN_TYPE(HIDDEN_SIZE, init = _init, inner_init= i_init, periods=[1, 2, 4, 8, 16, 32], stateful=True, return_sequences=True)
			h2 = layer2(h1)
		else:
			layer2 = RNN_TYPE(HIDDEN_SIZE, init= _init, inner_init= i_init, stateful=True, return_sequences=True)
			h2 = layer2(h1)
	with tf.device('/gpu:2'):
		layer3 = TimeDistributed(Dense(num_classes+1))
		o = layer3(h2)
	loss = tf.reduce_mean(K.categorical_crossentropy(o, labels, True))
	acc_value = accuracy(labels, o)	

	global_step = tf.Variable(0, trainable=False)

	real_valued = []
	var_to_real = {}
	for w in tf.trainable_variables():
		v = tf.Variable(w.initialized_value(), dtype=tf.float32, trainable=False)
		var_to_real[w] = v 
		real_valued.append(v)

	update_ops = []
	for old_value, new_value in layer1.updates + layer2.updates + layer3.updates:
		update_ops.append(tf.assign(old_value, new_value).op)


	##########################

	#### Train loop ####
	learning_rate = tf.placeholder(tf.float32, shape=(), name="LR")
	optimizer = tf.train.AdamOptimizer(learning_rate)
	grads_vars = optimizer.compute_gradients(loss, colocate_gradients_with_ops = True if RNN_TYPE is not Clockwork else False)
	grads = []
	for k, (g, v) in enumerate(grads_vars):
		grads.append(g)
	clipped_grads, global_norm = tf.clip_by_global_norm(grads, 1)
	for k, (grad, var) in enumerate(grads_vars):
		grads_vars[k] = (clipped_grads[k], var_to_real[var])#(tf.clip_by_value(grad, -1, 1) if grad is not None else grad, var_to_real[var])
	app = optimizer.apply_gradients(grads_vars, global_step = global_step)
	assignments = [tf.assign(w, quantify(var_to_real[w])).op for w in tf.trainable_variables()]
	clips = [tf.assign(w, tf.clip_by_value(w, -val, val)).op for w in real_valued]

	saver = tf.train.Saver()
	losses = []
	accuracies = []
	#T.silence()
	lr = LR

	init_op = tf.initialize_all_variables()
	with sess.as_default():
		init_op.run()
		for batch in np.arange(num_batch):
			sess.run(assignments)
			X, y = x_y_generator.next()
			print(tf.trainable_variables()[0].eval())
			app.run(feed_dict={i: X, labels: y, learning_rate: lr})
			for op in update_ops:
				op.run(feed_dict={i: X})
			for w in clips:
				w.run()
			if batch % how_often == 0:
				validation_loss = 0.0
				validation_acc = 0.0
				count = 0
				while count < num_test:
					test_X, test_y = test_generator.next()
					curr_loss, acc = sess.run([loss, acc_value], {i: test_X, labels: test_y})
					validation_loss += curr_loss
					validation_acc += acc
					count += 1
				losses.append(validation_loss / count)
				accuracies.append(validation_acc / count)
				if SAVE_WEIGHTS: saver.save(sess, SAVE_PATH)
				if VERBOSE: 
					printProgress(batch, num_batch_in_epoch, how_often, losses[-1])
					print("Accuracy at last batch: {0}".format(validation_acc / count))
				with open(LOSS_PATH + "{0}_{1}_{2}_{3}.w".format(LR, val, RNN_TYPE.__name__, TIMESTEPS), "wb") as f:
					pickle.dump([losses, accuracies], f)
				if (validation_acc / count) < 0.005 and batch > num_batch * 50:
					print("Returning early due to failure.")
					return

#for lr in [1e-1, 1e-2, 1e-6]:
#	for val in [np.inf]:
#		for rnn in [SimpleRNN, GRU, Clockwork]:
#			print("\tBeginning run with LR = {0}, val = {1}, type of RNN = {2}".format(lr, val, rnn.__name__))
#			run(lr, val, rnn, NUM_EPOCH = 1)
run(1e-4, 1, SimpleRNN, NUM_EPOCH = 20)
run(1e-4, 1, GRU, NUM_EPOCH = 20)
run(1e-4, 1, LSTM, NUM_EPOCH = 20)
run(1e-4, 1, Clockwork, NUM_EPOCH = 20)
