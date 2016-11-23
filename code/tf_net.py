import tensorflow as tf 
import numpy as np

def to_categorical(y, nb_classes=None):
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y


BATCH_SIZE = 128
nb_classes = 2
NUM_EPOCH = 10

def load_data(name):
	PATH = "../datasets/"
	train_data = []
	train_labels = []
	with open(PATH + name, "rb") as f:
		lines = f.readlines()
		for line in lines[1:]:
			line_arr = line.strip().split(",")
			train_labels.append(int(line_arr[-1]))
			train_data.append(line_arr[2:-1])
	return np.array(train_data), np.array(train_labels)

X_train, y_train = load_data("datatraining.txt")
X_test, y_test = load_data("datatest.txt")
y_train = to_categorical(y_train, nb_classes)
y_test = to_categorical(y_test, nb_classes)

def data_generator(X, y):
	assert X.shape[0] == y.shape[0]
	assert BATCH_SIZE < X.shape[0]
	i = 0
	while True:
		yield X[i*BATCH_SIZE:(i+1)*BATCH_SIZE, :], y[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
		i += 1
		if (i+1)*BATCH_SIZE >= X.shape[0]:
			i = 0


INPUT_SIZE = 5
W1_SHAPE = (20, 5)
W2_SHAPE = (10, 20)
SOFTMAX_SHAPE = (2, 10)

def unpackable_tensor(shape):
	x, y = shape
	unpacked = []
	for i in np.arange(x):
		temp = []
		for j in np.arange(y):
			temp.append(tf.Variable(tf.random_normal((1,), stddev=0.01)))
		unpacked.append(temp)
	tensor_pack = tf.pack(tf.pack(unpacked))
	return np.array(unpacked), tf.transpose(tf.squeeze(tensor_pack))

W1_unpacked, W1 = unpackable_tensor(W1_SHAPE)
b1_unpacked, b1 = unpackable_tensor((W1_SHAPE[0],1))
W2_unpacked, W2 = unpackable_tensor(W2_SHAPE)
b2_unpacked, b2 = unpackable_tensor((W2_SHAPE[0],1))
S_unpacked, S = unpackable_tensor(SOFTMAX_SHAPE)
bS_unpacked, bS = unpackable_tensor((SOFTMAX_SHAPE[0],1))

X, y = tf.placeholder(tf.float32, [None, 5]), tf.placeholder(tf.float32, [None, 2])
h1 = tf.matmul(X, W1) + b1
a1 = tf.tanh(h1)
h2 = tf.matmul(a1, W2) + b2
a2 = tf.tanh(h2)
h3 = tf.matmul(a2, S) + bS
loss = tf.nn.softmax_cross_entropy_with_logits(h3, y)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

with tf.Session() as sess:
	tf.initialize_all_variables().run()
	data_g = data_generator(X_train, y_train)
	for i in range(NUM_EPOCH):
		batch_xs, batch_ys = data_g.next()
		sess.run(train_step, feed_dict={X: batch_xs, y: batch_ys})
	print(loss.get_shape())
	print(y.get_shape())
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(loss,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print(sess.run(accuracy, feed_dict={X: X_test, y: y_test}))