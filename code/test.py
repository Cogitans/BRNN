import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Input, Dense
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.callbacks import Callback
from keras.datasets import mnist
import pickle

(X_train, y_train), (X_test, y_test) = mnist.load_data()

num_batches = 10 
num_epoch = 1
nb_classes = 10

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

X_batches, y_batches = np.array_split(X_train, num_batches, axis=0), np.array_split(Y_train, num_batches, axis=0)

model = Sequential()
model.add(Dense(150, input_shape=(784,)))
model.add(Dense(100, activation='tanh'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=1e-2),
              metrics=['accuracy'])

weights = []

for epoch in np.arange(num_epoch):
	for batch in np.arange(len(X_batches)):
		weights.append(model.get_weights())
		X, y = X_batches[batch], y_batches[batch]
		model.train_on_batch(X, y)
	print("\n"+"Loss/Accuracy at epoch {0}: ".format(epoch) + str(model.evaluate(X_test, Y_test)))

with open("weights.pickle", "wb") as f:
	pickle.dump([weights], f)