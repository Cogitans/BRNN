import numpy as np 
import os
import matplotlib.pyplot as plt
from keras.models import Sequential, Model, load_model
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers.core import Dense, Activation, Reshape, Flatten
from keras.optimizers import SGD, RMSprop, Adagrad, Adam
from keras.layers.embeddings import Embedding
from keras.callbacks import Callback, ReduceLROnPlateau
from keras.regularizers import l2
from keras_extensions import *
from keras.objectives import categorical_crossentropy
from keras.utils.np_utils import to_categorical
from utils import *
import tensorflow as tf
sess = tf.Session()
from keras import backend as K
K.set_session(sess)

DATA = "../datasets/"
TEXT8 = DATA + "text8"
MODEL_PATH = DATA + "model.keras"


BATCH_SIZE = 256
TIMESTEPS = 64
BATCH_PER_EPOCH = 5
NUM_EPOCH = 100

EVERY = 10
START = ''
LOAD = False
HIDDEN_DIM_1 = 500
HIDDEN_DIM_2 = 250
LEARNING_RATE = 1e-4

def text_8_generator(CHAR_NUM, NB_SAMPLES):
	CHAR_NUM = CHAR_NUM + 1
	with open(TEXT8, "rb") as f:
		read_data = f.readlines()
		char_list = [START] + list(read_data[0])
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

def data_target_generator(g, num_chars, c_to_l):
	while True:
		text = g.next()
		data = one_hot(text, c_to_l, INPUT_DIM)
		X = data[:, :-1, :]
		y = data[:, 1:, :]
		yield (X, y)


def build_model(INPUT_DIM, BATCH_SIZE, TIMESTEPS):
	with tf.device('/cpu:0'):
		x1 = tf.placeholder(tf.float32, shape=(BATCH_SIZE, TIMESTEPS, INPUT_DIM))
		x2 = tf.placeholder(tf.float32, shape=(BATCH_SIZE, TIMESTEPS, INPUT_DIM))
		model = Sequential()
		model.add(QuantGRU(HIDDEN_DIM_1, batch_input_shape=(BATCH_SIZE, TIMESTEPS, INPUT_DIM), quant=trinaryQuant, return_sequences=True, stateful=True))
		model.add(QuantGRU(HIDDEN_DIM_2, quant=trinaryQuant, return_sequences=True, stateful=True))
		model.add(QuantGRU(INPUT_DIM, quant=trinaryQuant, return_sequences=True, stateful=True))
	with tf.device('/gpu:0'):
		output_0 = model(x1)
	with tf.device('/gpu:1'):
		output_1 = model(x2)
	with tf.device('/cpu:0'):
		preds = 0.5 * (output_0 + output_1)
	return preds

def one_hot(text, mapping):
	data = np.zeros((text.shape[0], text.shape[1], INPUT_DIM))
	for (x,y), value in np.ndenumerate(text):
		data[x, y, mapping[value]] = 1
	return data

def data_target_generator(g, num_chars, c_to_l):
	while True:
		text = g.next()
		data = one_hot(text, c_to_l)
		X = data[:, :-1, :]
		y = data[:, 1:, :]
		yield (X, y)

generator = text_8_generator(TIMESTEPS, BATCH_SIZE)
INPUT_DIM, c_to_l, l_to_c = char_mapping()
with tf.device('/cpu:0'):
	x1 = tf.placeholder(tf.float32, shape=(BATCH_SIZE, TIMESTEPS, INPUT_DIM))
	x2 = tf.placeholder(tf.float32, shape=(BATCH_SIZE, TIMESTEPS, INPUT_DIM))
	model = Sequential()
	model.add(QuantGRU(HIDDEN_DIM_1, batch_input_shape=(BATCH_SIZE, TIMESTEPS, INPUT_DIM), quant=trinaryQuant, return_sequences=True, stateful=True))
	model.add(QuantGRU(HIDDEN_DIM_2, quant=trinaryQuant, return_sequences=True, stateful=True))
	model.add(QuantGRU(INPUT_DIM, quant=trinaryQuant, return_sequences=True, stateful=True))
with tf.device('/gpu:0'):
	output_0 = model(x1)
with tf.device('/gpu:0'):
	output_1 = model(x2)
#prediction_model = build_model(INPUT_DIM, 1, 1)

train_generator = data_target_generator(generator, INPUT_DIM, c_to_l)

labels1 = tf.placeholder(tf.float32, shape=(BATCH_SIZE, TIMESTEPS, INPUT_DIM))
labels2 = tf.placeholder(tf.float32, shape=(BATCH_SIZE, TIMESTEPS, INPUT_DIM))
loss1 = tf.reduce_mean(categorical_crossentropy(labels1, output_0))
loss2 = tf.reduce_mean(categorical_crossentropy(labels2, output_1))
loss = 0.5 * (loss1 + loss2)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

with sess.as_default():
	for _ in range(50):
		X1, y1 = train_generator.next()
		X2, y2 = train_generator.next()
		train_step.run(feed_dict={x1: X1, labels1: y1, x2: X2, labels2: y2})
		print loss.eval(feed_dict={x1: X1, labels1: y1, x2: X2, labels2: y2})

		
		

def generate(seed=None, predicate=lambda x: len(x) < 100):
	prediction_model.set_weights(model.get_weights())
	generated = np.zeros((1, 1), dtype='|S1')
	generated.fill(START)
	sentences = ["" for _ in np.arange(1)]

	while predicate(sentences[0]):
		X = one_hot(generated, c_to_l)
		probs = prediction_model.predict(X, verbose=0)
		for s in np.arange(1):
			next_char = l_to_c[sample(probs[s, 0, :])]
			generated[s, 0] = next_char
			sentences[s] += next_char
	return sentences

class Sample(Callback):
    def on_epoch_end(self, batch, logs):
        if batch % EVERY == 0 and batch != 0:
            print("".join(generate()))


history = model.fit_generator(train_generator, BATCH_SIZE, NUM_EPOCH, verbose=1, callbacks=[Sample(), QuanizationUpdate(), ReduceLROnPlateau(monitor='loss')])

model.save(MODEL_PATH)
loss = history.history['loss']
plt.plot(np.arange(len(loss)), loss)
plt.show()
plt.savefig("../imgs/loss.png")

