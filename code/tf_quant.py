import numpy as np
import tensorflow as tf
from utils import *
from quantifications import *
from keras_extensions import *
from keras import backend as K
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy as accuracy
import time
import pickle
import matplotlib.pyplot as plt
sess = tf.Session()
K.set_session(sess)

#### Constants Definition ####

DATA = "../datasets/"
SAVE = "../results/"
TEXT8 = DATA + "text8"
MODEL_PATH = DATA + "model.keras"
SAVE_PATH = SAVE + "saved_quick.tf"
LOSS_PATH = SAVE + "testing/"

batch_size = 64
HIDDEN_SIZE = 400
num_batch = None
how_often = 25


##############################

#### Data Establishing ####
def run(LR, val, RNN_TYPE, TIMESTEPS = 64, GPU_FLAG=True, NUM_BATCH = None, SAVE_WEIGHTS = False, VERBOSE = True):
	quantify = deterministic_ternary(val)
	num_timesteps = TIMESTEPS
	text_generator = text_8_generator(num_timesteps, batch_size)
	test_generator = test_8_generator(num_timesteps, batch_size)
	num_classes, c_to_l, l_to_c = char_mapping(TEXT8)
	x_y_generator = data_target_generator(text_generator, c_to_l, num_classes)
	text_x_y_generator = data_target_generator(test_generator, c_to_l, num_classes)
	num_batch_in_epoch = per_epoch(batch_size, num_timesteps)
	num_batch = 1 * num_batch_in_epoch / 10 if not NUM_BATCH else NUM_BATCH

	###########################

	#### Model Definition ####

	i = tf.placeholder(tf.float32, shape=(batch_size, num_timesteps, num_classes), name="X")
	labels = tf.placeholder(tf.float32, shape=(batch_size, num_timesteps, num_classes), name="y")
	with tf.device('/gpu:0') if RNN_TYPE != Clockwork and GPU_FLAG else tf.device('/cpu:0'):
		if RNN_TYPE == Clockwork:
			layer1 = RNN_TYPE(HIDDEN_SIZE, init=ternary_choice(val), inner_init=ternary_choice(val), periods=[1, 2, 4, 8, 16], stateful=True, return_sequences=True)
			h1 = layer1(i)
		else:
			layer1 = RNN_TYPE(HIDDEN_SIZE, init=ternary_choice(val), inner_init=ternary_choice(val),stateful=True, return_sequences=True)
			h1 = layer1(i)
	with tf.device('/gpu:1') if RNN_TYPE != Clockwork and GPU_FLAG else tf.device('/cpu:0'):
		if RNN_TYPE == Clockwork:
			layer2 = RNN_TYPE(num_classes, init=ternary_choice(val), inner_init=ternary_choice(val), periods=[1, 2, 4, 8, 16, 32, 64], stateful=True, return_sequences=True)
			o = layer2(h1)
		else:
			layer2 = RNN_TYPE(num_classes, init=ternary_choice(val), inner_init=ternary_choice(val),stateful=True, return_sequences=True)
			o = layer2(h1)
	loss = tf.reduce_mean(categorical_crossentropy(labels, o))
	acc_value = accuracy(labels, o)

	real_valued = []
	var_to_real = {}
	for w in tf.trainable_variables():
		v = tf.Variable(w.initialized_value(), dtype=tf.float32, trainable=False)
		var_to_real[w] = v 
		real_valued.append(v)

	update_ops = []
	for old_value, new_value in layer1.updates + layer2.updates:
		update_ops.append(tf.assign(old_value, new_value).op)


	##########################

	#### Train loop ####

	optimizer = tf.train.AdamOptimizer(LR)
	grads_vars = optimizer.compute_gradients(loss)
	for k, (grad, var) in enumerate(grads_vars):
		grads_vars[k] = (grad, var_to_real[var])
	app = optimizer.apply_gradients(grads_vars)
	with tf.device('/gpu:3') if RNN_TYPE != Clockwork and GPU_FLAG else tf.device('/cpu:0'):
		assignments = [tf.assign(w, quantify(var_to_real[w])).op for w in tf.trainable_variables()]
		clips = [tf.assign(w, tf.clip_by_value(w, -val, val)).op for w in real_valued]

	saver = tf.train.Saver()
	losses = []

	T.silence()

	init_op = tf.initialize_all_variables()
	with sess.as_default():
		init_op.run()
		for w in tf.trainable_variables():
			tf.assign(var_to_real[w], w).op.run()
		for batch in np.arange(num_batch):
			T.tic("assigning quantifications")
			for assign in assignments:
				assign.run()
			T.toc()
			T.tic("gathering data")
			X, y = x_y_generator.next()
			test_X, test_y = text_x_y_generator.next()
			T.toc()
			T.tic("computing forward pass")
			app.run(feed_dict={i: X, labels: y})
			# for g, v in grads_vars:
			# 	print(g.eval(feed_dict={i: X, labels: y}))
			T.toc()
			T.tic("updating statefulness")
			for op in update_ops:
				op.run(feed_dict={i: X, labels: y})
			T.toc()
			T.tic("restricting real-valued weights")
			for w in clips:
				w.run()
			T.toc()
			if batch % how_often == 0:
				T.tic("calculating loss")
				validation_loss = 0.0
				count = 0
				while True:
					(test_X, test_Y), done = text_x_y_generator.send(True)
					curr_loss = loss.eval(feed_dict={i: test_X, labels: test_y})
					validation_loss += curr_loss
					count += 1
					if done:
						break
				losses.append(validation_loss / count)
				T.toc()
				if SAVE_WEIGHTS: saver.save(sess, SAVE_PATH)
				if VERBOSE: printProgress(batch, num_batch_in_epoch, how_often, losses[-1])
				T.tic("saving loss")
				with open(LOSS_PATH + "{0}_{1}_{2}_{3}.w".format(LR, val, RNN_TYPE.__name__, TIMESTEPS), "wb") as f:
					pickle.dump([losses], f)
				T.toc()

# for lr in [1e-4]:
# 	for val in [1]:
# 		for rnn in [SimpleRNN, GRU, Clockwork]:
# 			for timestep in [8, 16, 32, 64]:
# 				print("\tBeginning run with LR = {0}, val = {1}, type of RNN = {2}, timestep = {3}".format(lr, val, rnn.__name__, timestep))
# 				run(lr, val, rnn, timestep, False)



run(1e-6, 1, GRU)
