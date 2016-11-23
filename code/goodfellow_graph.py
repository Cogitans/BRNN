import pickle
import math
from keras.datasets import mnist
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import numpy.linalg as la
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from mpl_toolkits.mplot3d import Axes3D

num_alphas = 10
MAKE = False

with open("weights.pickle", "rb") as f:
	weights = pickle.load(f)[0]

def generate_range():
	def array_minus(e, s):
		assert len(e) == len(s)
		return [e[i] - s[i] for i in np.arange(len(e))]

	def array_norm(ws):
		sums = math.sqrt(sum([la.norm(w)**2 for w in ws]))
		return [w/sums for w in ws]

	def array_project(ws, to_projs):
		return [np.dot(ws[i].flatten(), to_projs[i].flatten())*to_projs[i] for i in np.arange(len(ws))]

	def array_residual(ws, projects):
		return [ws[i] - projects[i] for i in np.arange(len(ws))]

	def array_mult(arr, alpha):
		return [w*alpha for w in arr]

	def array_add(arr1, arr2):
		assert len(arr1) == len(arr2)
		return [arr1[i] + arr2[i] for i in np.arange(len(arr1))]

	def array_alphatize(start, end, steps):
		alphas = np.arange(0, 1, 1.0/steps)
		return [array_add(start, array_mult(end, alpha)) for alpha in alphas]



	start_w, end_w = weights[0], weights[-1]
	line = array_minus(end_w, start_w)
	normed_line = array_norm(line)

	x_projections = [array_project(w, normed_line) for w in weights]
	y_residuals = [array_residual(w, normed_line) for w in weights]
	return [array_alphatize(x_projections[i], y_residuals[i], num_alphas) for i in np.arange(len(x_projections))]


(X_train, y_train), (X_test, y_test) = mnist.load_data()

num_batches = 100 
num_epoch = 2
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
              optimizer=Adam(lr=1e-2))
if MAKE:
	ranges = generate_range()
	losses = []
	print(len(ranges), len(ranges[0]))
	for x in np.arange(len(ranges)):
		print(x)
		losses.append([])
		for y in np.arange(len(ranges[x])):
			print(y)
			model.set_weights(ranges[x][y])
			losses[-1].append(model.evaluate(X_test, Y_test))

	with open("losses.f", "wb") as f:
		pickle.dump([losses], f)
else:
	with open("losses.f", "rb") as f:
		losses = pickle.load(f)[0]

x, y = np.arange(len(losses)), np.arange(len(losses[0]))
X, Y = np.meshgrid(x, y)
zs = np.array([losses[x][y] for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)


fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                      linewidth=0, antialiased=False)

true_path = [L[-1] for L in losses]
x_true, y_true = [x[-1] for x in X], [y[-1] for y in Y]

line = ax.plot(y_true, x_true, true_path)

#fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()