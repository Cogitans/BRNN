import numpy as np 
import os
# import matplotlib.pyplot as plt
from keras.models import Sequential, Model, load_model
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers.core import Dense, Activation, Reshape, Flatten, Merge
from keras.optimizers import SGD, RMSprop, Adagrad, Adam
from keras.layers.embeddings import Embedding
from keras.callbacks import Callback
from keras.preprocessing import sequence
from data_gen import *
import tensorflow as tf
from keras import backend as K
sess = tf.Session()
K.set_session(sess)

DATA = """FIX"""
TEXT8 = """FIX"""
MODEL_PATH = """FIX"""


BATCH_SIZE = 32
NUM_SAMPLES = 32
TIMESTEPS = 64
BATCH_PER_EPOCH = 1
NUM_EPOCH = 200


def inf_generator(generator):
    while True:
        try:
            for elem in generator():
                yield elem
        except StopIteration:
            continue

def word_mapping(raw_generator):
    """
    Returns a tuple of:
        - distince # of words in dataset
        - mapping from word -> index
        - mapping from index -> word
    Note: 0 is reserved for padding
          1 is reserved for unknown words at test time
    """
    total_words = 2
    word_to_idx = {}
    idx_to_word = {}
    for line in raw_generator():
        sentence = line[0].lower()
        for word in sentence.split(' '):
            if word not in word_to_idx:
                    word_to_idx[word] = total_words
                    idx_to_word[total_words] = word
                    total_words += 1
    return total_words, word_to_idx, idx_to_word

def one_hot(text, mapping):
    """
    Takes in a matrix of characters and outputs a matrix of one-hot indexes of same shape
    according to the word -> index dictionary "mapping"
    """
    data = np.zeros(len(text))
    for i in range(len(text)):
        if text[i] not in mapping:
            data[i] = 1
        else:
            data[i] = mapping[text[i]]
    return data

def shakespeare_pair_train_target_generator(word2idx):
    """
    Should run forever, yielding (X, Y), where:
    X is a one-hot data matrix of dimensions (#samples, #timesteps)
    Y is a target matrix (0 or 1) of dimensions (#samples, #timesteps)
    """
    i = 1
    X = np.zeros((NUM_SAMPLES, TIMESTEPS))
    Y = np.zeros((NUM_SAMPLES, TIMESTEPS, 2))
    for line_1, line_2, did_speaker_change in inf_generator(shakespeare_soft_train_gen):
        line_1, line_2 = line_1.lower(), line_2.lower()
        line_1_split, line_2_split = line_1.split(' '), line_2.split(' ')
        words = one_hot(line_1_split + line_2_split, word2idx)
        words = sequence.pad_sequences([words], maxlen=TIMESTEPS)[0]
        
        targets = np.zeros((len(words), 2))
        targets[:, 0] = 1

        if did_speaker_change:
            targets[len(line_1_split), :] = [0, 1]

        X[i%NUM_SAMPLES, :] = words
        Y[i%NUM_SAMPLES, :] = targets
        
        if i % NUM_SAMPLES == 0:
            yield X, Y
            X = np.zeros((NUM_SAMPLES, TIMESTEPS))
            Y = np.zeros((NUM_SAMPLES, TIMESTEPS, 2))
        
        i += 1

def shakespeare_soft_train_target_generator(word2idx):
    """
    Should run forever, yielding (X, Y), where:
    X is a one-hot data matrix of dimensions (#samples, #timesteps)
    Y is a target matrix (0 or 1) of dimensions (#samples, #timesteps)
    """
    i = 1
    X = np.zeros((NUM_SAMPLES, TIMESTEPS))
    Y = np.zeros((NUM_SAMPLES, TIMESTEPS, 2))
    for line_1, line_2, did_speaker_change in inf_generator(shakespeare_soft_train_gen):
        line_1, line_2 = line_1.lower(), line_2.lower()
        line_1_split, line_2_split = line_1.split(' '), line_2.split(' ')
        words = one_hot(line_1_split + line_2_split, word2idx)
        words = sequence.pad_sequences([words], maxlen=TIMESTEPS)[0]
        
        targets = np.zeros((len(words), 2))
        targets[:, 0] = 1

        if did_speaker_change:
            targets[len(line_1_split), :] = [0, 1]

        X[i%NUM_SAMPLES, :] = words
        Y[i%NUM_SAMPLES, :] = targets
        
        if i % NUM_SAMPLES == 0:
            yield X, Y
            X = np.zeros((NUM_SAMPLES, TIMESTEPS))
            Y = np.zeros((NUM_SAMPLES, TIMESTEPS, 2))
        
        i += 1

def shakespeare_soft_test_target_generator(word2idx):
    """
    Should run forever, yielding (X, Y), where:
    X is a one-hot data matrix of dimensions (#samples, #timesteps)
    Y is a target matrix (0 or 1) of dimensions (#samples, #timesteps)
    """
    i = 1
    X = np.zeros((NUM_SAMPLES, TIMESTEPS))
    Y = np.zeros((NUM_SAMPLES, TIMESTEPS, 2))
    for line_1, line_2, did_speaker_change in shakespeare_soft_test_gen():
        line_1, line_2 = line_1.lower(), line_2.lower()
        line_1_split, line_2_split = line_1.split(' '), line_2.split(' ')
        words = one_hot(line_1_split + line_2_split, word2idx)
        words = sequence.pad_sequences([words], maxlen=TIMESTEPS)[0]
        
        targets = np.zeros((len(words), 2))
        targets[:, 0] = 1

        if did_speaker_change:
            targets[len(line_1_split), :] = [0, 1]

        X[i%NUM_SAMPLES, :] = words
        Y[i%NUM_SAMPLES, :] = targets
        
        if i % NUM_SAMPLES == 0:
            yield X, Y
            X = np.zeros((NUM_SAMPLES, TIMESTEPS))
            Y = np.zeros((NUM_SAMPLES, TIMESTEPS, 2))
        
        i += 1

def wilde_soft_train_target_generator(word2idx):
    """
    Should run forever, yielding (X, Y), where:
    X is a one-hot data matrix of dimensions (#samples, #timesteps)
    Y is a target matrix (0 or 1) of dimensions (#samples, #timesteps)
    """
    i = 1
    X = np.zeros((NUM_SAMPLES, TIMESTEPS))
    Y = np.zeros((NUM_SAMPLES, TIMESTEPS, 2))
    for line_1, line_2, did_speaker_change in inf_generator(wilde_soft_gen):
        line_1, line_2 = line_1.lower(), line_2.lower()
        line_1_split, line_2_split = line_1.split(' '), line_2.split(' ')
        words = one_hot(line_1_split + line_2_split, word2idx)
        words = sequence.pad_sequences([words], maxlen=TIMESTEPS)[0]
        
        targets = np.zeros((len(words), 2))
        
        if did_speaker_change:
            targets[len(line_1_split), :] = [0, 1]
        X[i%NUM_SAMPLES, :] = words
        Y[i%NUM_SAMPLES, :] = targets
        
        if i % NUM_SAMPLES == 0:
            yield X, Y
            X = np.zeros((NUM_SAMPLES, TIMESTEPS))
            Y = np.zeros((NUM_SAMPLES, TIMESTEPS, 2))
        
        i += 1

def build_model(input_dim):
    HIDDEN_DIM = 128
    LEARNING_RATE = 3e-1
    adam = Adam(lr=LEARNING_RATE)

    model = Sequential()
    model.add(Embedding(input_dim, HIDDEN_DIM, batch_input_shape=(BATCH_SIZE, TIMESTEPS)))
    model.add(GRU(HIDDEN_DIM, return_sequences=True, stateful=True))
    model.add(GRU(2, activation='softmax', return_sequences=True, stateful=True))
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model 


def many_to_one_model():
    data_dim, word2idx, indx2word = word_mapping(shakespeare_raw_train_gen)
    timesteps = 32
    hidden_dim = 2048
    adam = Adam(lr=3e-3)
    with tf.device('/gpu:0'):
    	encoder_a = Sequential()
   	encoder_a.add(Embedding(data_dim, hidden_dim, batch_input_shape=(BATCH_SIZE, timesteps)))
    	encoder_a.add(GRU(hidden_dim, stateful=True))
    with tf.device('/gpu:1'):
    	encoder_b = Sequential()
   	encoder_b.add(Embedding(data_dim, hidden_dim, batch_input_shape=(BATCH_SIZE, timesteps)))
    	encoder_b.add(GRU(hidden_dim, stateful=True))
	
	model = Sequential()
    	model.add(Merge([encoder_a, encoder_b], mode=lambda x: x[0] - x[1], output_shape=lambda x: x[0]))
    	model.add(Dense(512, activation='relu', init='glorot_normal'))
    	model.add(Dense(2, activation='softmax'))

    model.compile(loss='binary_crossentropy',
                    optimizer=adam,
                    metrics=['accuracy'])

    def gen(train=True):

        i = 1
        X_a = np.zeros((NUM_SAMPLES, timesteps))
        X_b = np.zeros((NUM_SAMPLES, timesteps))
        Y = np.zeros((NUM_SAMPLES, 2))
        sample_weights = np.ones(NUM_SAMPLES)
        if train:
            generator = shakespeare_soft_train_gen
        else:
            generator = shakespeare_soft_test_gen

        for line_1, line_2, did_speaker_change in inf_generator(generator):
            line_1, line_2 = line_1.lower(), line_2.lower()
            line_1_split, line_2_split = line_1.split(' '), line_2.split(' ')
            words_1 = one_hot(line_1_split, word2idx)
            words_1 = sequence.pad_sequences([words_1], maxlen=timesteps)[0]
            words_2 = one_hot(line_2_split, word2idx)
            words_2 = sequence.pad_sequences([words_2], maxlen=timesteps)[0]

            if did_speaker_change:
                Y[i%NUM_SAMPLES] = [1, 0]
                sample_weights[i%NUM_SAMPLES] = 2
            else:
                Y[i%NUM_SAMPLES] = [0, 1]

            X_a[i%NUM_SAMPLES, :] = words_1
            X_b[i%NUM_SAMPLES, :] = words_2
            
            if i % NUM_SAMPLES == 0:
                yield ([X_a, X_b], Y, sample_weights)
                X_a = np.zeros((NUM_SAMPLES, timesteps))
                X_b = np.zeros((NUM_SAMPLES, timesteps))
                Y = np.zeros((NUM_SAMPLES, 2))
                sample_weights = np.ones(NUM_SAMPLES)
            
            i += 1

    model.fit_generator(gen(train=True), BATCH_SIZE, NUM_EPOCH)
    results = model.evaluate_generator(gen(train=False), NUM_SAMPLES)
    print 'Test Time Results'
    print model.metrics_names
    print results

############################
### ORIGINAL MODEL START ###
############################
# num_words, word2idx, indx2word = word_mapping(shakespeare_raw_train_gen)
# train_generator = shakespeare_soft_train_target_generator(word2idx)
# test_generator = shakespeare_soft_test_target_generator(word2idx)
# model = build_model(num_words)

# history = model.fit_generator(train_generator, BATCH_SIZE, NUM_EPOCH, verbose=2)
# results = model.evaluate_generator(test_generator, NUM_SAMPLES)

# print 'Test Time Results'
# print model.metrics_names
# print results
##########################
### ORIGINAL MODEL END ###
##########################


############################
### MANY TO ONE    START ###
############################
many_to_one_model()
##########################
### MANY TO ONE    END ###
##########################
