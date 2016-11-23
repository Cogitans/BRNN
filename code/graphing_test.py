import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Input, Dense
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.callbacks import Callback
import tensorflow as tf
from keras import backend as K
import os
import pickle
import shutil
tf.python.control_flow_ops = tf


batch_size = 128
nb_classes = 2
nb_epoch = 50
path = "./graph.struct"

def load_data():
	PATH = "../datasets/"
	train_data = []
	train_labels = []
	with open(PATH + "datatraining.txt", "rb") as f:
		lines = f.readlines()
		for line in lines[1:]:
			line_arr = line.strip().split(",")
			train_labels.append(int(line_arr[-1]))
			train_data.append(line_arr[2:-1])
	test_data = []
	test_labels = []
	with open(PATH + "datatest.txt", "rb") as f:
		lines = f.readlines()
		for line in lines[1:]:
			line_arr = line.strip().split(",")
			test_labels.append(int(line_arr[-1]))
			test_data.append(line_arr[2:-1])
	return (np.array(train_data), np.array(train_labels)), (np.array(test_data), np.array(test_labels))

(X_train, y_train), (X_test, y_test) = load_data()

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
i = Input(shape=(5,))
d = Dense(2, activation='tanh')(i)
d = Dense(2, activation='softmax')(d)
model = Model(input=i, output=d)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=1e-2),
              metrics=['accuracy'])

weights = model.trainable_weights
flattened_weights = [tf.reshape(T, [-1]) for T in weights]
linear_weights = tf.concat(0, flattened_weights)
gradients = model.optimizer.get_gradients(model.total_loss, weights) 

input_tensors = [model.inputs[0], # input data
				 model.sample_weights[0],
                 model.targets[0], # labels
                 K.learning_phase(), # train or test mode
]


flattened_grads = [tf.reshape(T, [-1]) for T in gradients]
linear_grads = tf.concat(0, flattened_grads)
get_gradients = K.function(inputs=input_tensors, outputs=[linear_grads])
dim = linear_grads.get_shape()[0]
print(dim)
hessian = []
for grad in np.arange(int(dim)):
	if grad % 10 == 0:
		print(grad)
	subtensor = tf.slice(linear_grads, [grad], [1])
	subweight = tf.slice(weights[0], [grad, grad], [1, 1])
	print(subweight)
	hessian += K.gradients(subtensor, weights[0])
	print(hessian)
	print(K.gradients(subtensor, subweight))
	quit()
flattened_hessian = [tf.reshape(T, [-1]) for T in hessian]
Hessian = tf.concat(0, flattened_hessian)

get_hessian = K.function(inputs=input_tensors, outputs=[Hessian])

inputs = [X_train, # X
		  [1 for _ in np.arange(Y_train.shape[0])],
          Y_train, # sample weights
          0 # learning phase in TEST mode
		]



#### FOR FULL HESSIAN ####

# weights = model.trainable_weights
# gradients = model.optimizer.get_gradients(model.total_loss, weights) 

# input_tensors = [model.inputs[0], # input data
# 				 model.sample_weights[0],
#                  model.targets[0], # labels
#                  K.learning_phase(), # train or test mode
# ]


# flattened_grads = [tf.reshape(T, [-1]) for T in gradients]
# grads = tf.concat(0, flattened_grads)
# get_gradients = K.function(inputs=input_tensors, outputs=[grads])
# dim = grads.get_shape()[0]
# print(dim)
# hessian = []
# for grad in np.arange(int(dim)):
# 	if grad % 10 == 0:
# 		print(grad)
# 	subtensor = tf.slice(grads, [grad], [1])
# 	hessian += K.gradients(subtensor, weights)
# flattened_hessian = [tf.reshape(T, [-1]) for T in hessian]
# Hessian = tf.concat(0, flattened_hessian)

# get_hessian = K.function(inputs=input_tensors, outputs=[Hessian])

# inputs = [X_train, # X
# 		  [1 for _ in np.arange(Y_train.shape[0])],
#           Y_train, # sample weights
#           0 # learning phase in TEST mode
# ]
#############################
class SaveCallback(Callback):
	def on_epoch_end(self, batch, logs):
		hessian_values = [i[1] for i in zip(weights, get_hessian(inputs))]
		gradient_Values = [i[1] for i in zip(weights, get_gradients(inputs))]
		weight_values = self.model.get_weights()
		with open("../datasets/diagonal/"+str(batch)+".diagonal.save", "wb") as f:
			pickle.dump([hessian_values, gradient_Values, weight_values], f)


history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch, callbacks=[SaveCallback()],
                    verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
