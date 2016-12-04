import numpy as np
import tensorflow as tf
from utils import *
from quantifications import *
from keras_extensions import *
from keras.layers.core import Dense, TimeDistributedDense
from keras import backend as K
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers.wrappers import TimeDistributed
from keras.objectives import categorical_crossentropy, mean_squared_error
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
MOLDAU = DATA + "smetana/smetana.wav"
MODEL_PATH = DATA + "model.keras"
SAVE_PATH = SAVE + "saved_quick.tf"
LOSS_PATH = mkdir(SAVE + "generation_new/")

batch_size = 1
HIDDEN_SIZE = 513
num_batch = None
how_often = 50


##############################

#### Data Establishing ####
def run(LR, val, RNN_TYPE, TIMESTEPS = None, quant = None, GPU_FLAG=True, NUM_EPOCH = 1, NUM_BATCH = None, SAVE_WEIGHTS = False, VERBOSE = True, WHICH = None):
	if quant is None: assert val is np.inf
	quantify = identity if quant is None else quant(val)
	num_timesteps = TIMESTEPS
	
	g = music_generator(MOLDAU, batch_size, num_timesteps, percent = .00005, offset = 0.1)
	test_g = music_generator(MOLDAU, batch_size, num_timesteps, percent = .00005, offset = 0.1)

	x_y_generator = music_pair_generator(g)
	test_generator = music_pair_generator(test_g)

	num_batch_in_epoch, num_classes, train_timesteps = music_len(MOLDAU, batch_size, num_timesteps, percent = .00005, offset = 0.1)
	num_test, _, test_timesteps = music_len(MOLDAU, batch_size, num_timesteps, percent = .00005, offset = 0.1)
	num_batch = NUM_EPOCH * num_batch_in_epoch if not NUM_BATCH else NUM_BATCH
	how_often = 1
	
	_init = "he_normal" if val == np.inf else ternary_choice(val)
	i_init = "identity" if val == np.inf else scale_identity(val)
	if num_timesteps is None:
		num_timesteps = train_timesteps - 1
	###########################

	#### Model Definition ####

	i = tf.placeholder(tf.float32, shape=(batch_size, num_timesteps, num_classes), name="X")
	labels = tf.placeholder(tf.float32, shape=(batch_size, num_timesteps, num_classes), name="y")
	with tf.device('/gpu:0') if GPU_FLAG else tf.device('/cpu:0'):
		if RNN_TYPE == Clockwork:
			layer1 = RNN_TYPE(HIDDEN_SIZE, init = _init, inner_init= i_init, periods=[1, 2, 4, 8, 16, 32, 64, 128, 256], stateful=True, return_sequences=True)
			h1 = layer1(i)
		else:
			layer1 = RNN_TYPE(HIDDEN_SIZE, init = _init, inner_init = i_init, return_sequences=True)
			h1 = layer1(i)
	with tf.device('/gpu:1') if GPU_FLAG else tf.device('/cpu:0'):
		if RNN_TYPE == Clockwork:
			layer2 = RNN_TYPE(HIDDEN_SIZE, init = _init, inner_init= i_init, periods=[1, 2, 4, 8, 16, 32, 64, 128, 256], stateful=True, return_sequences=True)
			h2 = layer2(h1)
		else:
			layer2 = RNN_TYPE(HIDDEN_SIZE, init= _init, inner_init= i_init, return_sequences=True)
			h2 = layer2(h1)
	with tf.device('/gpu:2') if GPU_FLAG else tf.device('/cpu:0'):
		layer3 = TimeDistributed(Dense(num_classes, init = _init))
		o = layer3(h2)
	loss = tf.reduce_mean(mean_squared_error(labels, o))
	acc_value = accuracy(labels, o)	

	global_step = tf.Variable(0, trainable=False)

	to_quantize = []
	if WHICH == "all":
		to_quantize = tf.trainable_variables()
	elif WHICH == "hidden":
		for v in tf.trainable_variables():
			if "U" in v.name.split("_"):
				to_quantize.append(v)	
	elif WHICH == "input":
		for v in tf.trainable_variables():
			if "W" in v.name.split("_"):
				to_quantize.append(v)
	real_valued = []
	var_to_real = {}
	for w in to_quantize:
		v = tf.Variable(w.initialized_value(), dtype=tf.float32, trainable=False)
		var_to_real[w] = v 
		real_valued.append(v)

	update_ops = []
#	for old_value, new_value in layer1.updates + layer2.updates:
#		update_ops.append(tf.assign(old_value, new_value).op)


	##########################

	#### Train loop ####
	learning_rate = tf.placeholder(tf.float32, shape=(), name="LR")
	optimizer = tf.train.AdamOptimizer(learning_rate)
	grads_vars = optimizer.compute_gradients(loss, colocate_gradients_with_ops = True if RNN_TYPE is not Clockwork else False)
	grads = []
	vars_ = []
	for k, (g, v) in enumerate(grads_vars):
		vars_.append(v)
		grads.append(g)
	
	clipped_grads, global_norm = tf.clip_by_global_norm(grads, 1)
	for k, (grad, var) in enumerate(grads_vars):
		grads_vars[k] = (clipped_grads[k], var_to_real[var] if var in to_quantize else var)
	app = optimizer.apply_gradients(grads_vars, global_step = global_step)
	assignments = [tf.assign(w, quantify(var_to_real[w])) for w in to_quantize]
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
			app.run(feed_dict={i: np.zeros_like(X), labels: y, learning_rate: lr})
			sess.run(update_ops, feed_dict={i: X})
			sess.run(clips)
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
				with open(LOSS_PATH + "{0}_{1}_{2}_{3}_{4}.w".format(val, RNN_TYPE.__name__, TIMESTEPS, WHICH, quant.__name__ if quant is not None else ""), "wb") as f:
					pickle.dump([losses, accuracies], f)
				if (validation_acc / count) < 0.005 and batch > num_batch / 2 or (validation_acc == 0 and batch > num_batch / 5):
					print("Returning early due to failure.")
					return

run(1e-4, 1e-1, SimpleRNN, quant = deterministic_ternary, NUM_EPOCH = 200)
run(1e-4, 1e-1, GRU, quant = deterministic_ternary, NUM_EPOCH = 200)
run(1e-4, 1e-1, LSTM, quant = deterministic_ternary, NUM_EPOCH = 200)
run(1e-3, 1e-1, Clockwork, quant = deterministic_ternary, NUM_EPOCH = 200)
