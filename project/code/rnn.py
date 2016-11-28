import numpy as np 
import os
import matplotlib.pyplot as plt
from keras.models import Sequential, Model, load_model
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers.core import Dense, Activation, Reshape, Flatten
from keras.optimizers import SGD, RMSprop, Adagrad, Adam
from keras.layers.embeddings import Embedding
from keras.callbacks import Callback


DATA = """FIX"""
TEXT8 = """FIX"""
MODEL_PATH = """FIX"""

SHAKESPEARE = "../datasets/shakespeare/"
PLAY_PATH = SHAKESPEARE + "play_dict.p"

BATCH_SIZE = 32
TIMESTEPS = 64
BATCH_PER_EPOCH = 1
NUM_EPOCH = 500
START = ''

def char_mapping():
	"""
	MUST BE FILLED IN
	Should return a tuple of (#different inputs (chars) in dataset, mapping from char -> index, mapping from index -> char)
	"""
	total_chars = 0
    char_to_idx = {}
    idx_to_char = {}
    plays = pickle.load(open(PLAY_PATH, "rb"))[0]
    for play in plays:
        p = d[play][0]
        for i in xrange(len(p)):
            sentence = p[i].lower()
            for c in sentence:
                if c not in char_to_idx:
                    char_to_idx[c] = total_chars
                    idx_to_char[total_chars] = c
                    total_chars += 1
    return total_chars, char_to_idx, idx_to_char


def build_model(INPUT_DIM, BATCH_SIZE, TIMESTEPS):
	HIDDEN_DIM = 128
	LEARNING_RATE = 3e1
	adam = Adam(lr=LEARNING_RATE)

	model = Sequential()
	model.add(Embedding(INPUT_DIM, HIDDEN_DIM, batch_input_shape=(BATCH_SIZE, TIMESTEPS)))
	model.add(GRU(HIDDEN_DIM, return_sequences=True, stateful=True))
	model.add(GRU(INPUT_DIM, return_sequences=True, stateful=True))
	model.compile(optimizer=adam, loss='mse')
	return model 

def one_hot(text, mapping):
	"""
	Takes in a matrix of characters and outputs a matrix of one-hot indexes of same shape
	according to the char -> index dictionary "mapping"
	"""
	data = np.zeros(text.shape)
	for (x,y), value in np.ndenumerate(text):
		data[x, y] = mapping[value]
	return data

def data_target_generator(num_chars, c_to_l):
	"""
	MUST BE FILLED IN (arguments potentially changed), you should use one_hot
	Should run forever, yielding (X, Y), where:
	X is a one-hot data matrix of dimensions (#samples, #timesteps)
	Y is a target matrix (0 or 1) of dimensions (#samples, #timesteps)
	"""
	while True:



INPUT_DIM, c_to_l, l_to_c = char_mapping()
model = build_model(INPUT_DIM, BATCH_SIZE, TIMESTEPS)
train_generator = data_target_generator(INPUT_DIM, c_to_l)

history = model.fit_generator(train_generator, BATCH_SIZE, NUM_EPOCH, verbose=2)
model.save(MODEL_PATH)
loss = history.history['loss']
plt.plot(np.arange(len(loss)), loss)
plt.show()
plt.savefig("""FIX""")


"""RUN THE FOLLOWING IN PYTHON TO UNDERSTAND WHAT A MATRIX OF (#SAMPLES, #TIMESTEPS) SHOULD
LOOK LIKE FOR A STATEFUL RNN "(T is the 'input data')"""


"""
T = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y', 'z','1','2','3','4','5','6']

SEQ_LEN = 4
BATCH_SIZE = 2
BATCH_CHARS  = len(T) / BATCH_SIZE

x = [[0,0,0,0],[0,0,0,0]]

print 'Sequence: ', '  '.join(str(c) for c in T)
for i in range(0, BATCH_CHARS - SEQ_LEN +1, SEQ_LEN):
    print 'BATCH', i/SEQ_LEN
    for batch_idx in range(BATCH_SIZE):
        start = batch_idx * BATCH_CHARS + i          
        print '\tsequence', batch_idx, 'of batch:',
        for j in range(SEQ_LEN):
            x[batch_idx][j] = T[start+j]
            print T[start+j],
    #here we would yield x (the batch) if this were an iterator
        print x
