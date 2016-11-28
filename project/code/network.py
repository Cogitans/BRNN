from keras.preprocessing import sequence, text
from keras.preprocessing.text import one_hot, Tokenizer, text_to_word_sequence
from keras.optimizers import SGD, RMSprop, Adagrad, Adam
from keras.utils import np_utils, generic_utils
from keras.models import Sequential, Model, load_model
from keras.layers import Merge, Input, merge
from keras.layers.core import Dense, Activation, Reshape, Flatten
from keras.layers.embeddings import Embedding
from keras.callbacks import Callback
from keras.preprocessing.text import base_filter
from gensim.models import Word2Vec, Phrases
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords
from nltk.tag import PerceptronTagger
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize
import pickle
import os
import string
import re
import sys

#### PACKAGE REQUIREMENTS ####
##############################
# Gensim, Keras, Numpy, NLTK, 
# MatPlotLib, h5py, Cython
# SciKit Learn
##############################



# This is just a convenience variable
DATASET_PATH = "../datasets/shakespeare/"

#This is a path to a pickled file containing a dictionary from each document in the 
# dataset to two things in this order:
#	1) A list of sentences in that document
#	2) A charnum for each line. Two subsequent lines have the have charnum iff they were
#		spoken by the same person.

CBOW_PATH = DATASET_PATH + "play_dict.p"

# This is a path to a h5py saved version of the Keras Neural Network (topology + weights)
# May not be created if you plan on giving the --train flag
MODEL_PATH = DATASET_PATH + "keras.model"

# How big your context window will be
WINDOW_SIZE = 9

# This is the dimension of the embedded projection
dim_proj = 100

# The dimension of the one-hot-encoding. This is typically less than the vocabulary size (which is naively ~50,000)
# In practice, this tends not to matter since strings are hashed well
max_num_words = 20000

# The starting learning rate of the neural network
lr = 2e-3

# The batch size of the neural network. I highly recommend keeping it at 128
BATCH_SIZE = 128

# The number of batch-epochs to train for. Good results can be seen as early as 20-30
# Between 300-1000 is ideal for a baseline estimate 
NUM_EPOCH = 10

# The number of documents to predict keywords for if using the --predict flag
NUM_DOCS = 1

# The number of keywords to print if using the --predict flag
NUM_KEYWORDS = 25

# If this is true, print the least-predictive words in addition to the most predictive
SHOW_LEAST_PREDICTIVE = False

# If REGENERATE is flagged true, this program will recreate the data stored in CBOW_PATH based on the documents in DATASET_PATH
# 	Otherwise, it will load it from disk.
# If TRAIN is flagged true, this program will recreate and retrain a neural network from scratch
#	Otherwise, it will load it from disk.
# If PREDICT is flagged true, this program will load a neural network from disk and predict keywords
#	For the first NUM_DOCS documents.
REGENERATE, TRAIN, PREDICT, CONTINUE, INTERSECT = False, False, False, False, False


# Code for setting all the above values to be correct. 
# For basic purposes, --generate, --train and --predict are all you need
if __name__ == '__main__':
	for s in sys.argv[1:]:
		if s[:2] != "--":
			print("Invalid command line input.")
			quit()
		else:
			command = s[2:]
			if command == "generate":
				REGENERATE = True 
			if command == "train":
				TRAIN = True 
			if command == "predict":
				PREDICT = True
			if command == "continue":
				CONTINUE = True
			if command == "intersect":
				INTERSECT = True
			if command == "SemEval":
				DATASET_PATH = "../datasets/SemEval2010/"
				OUTFILE = DATASET_PATH + "train.data.p"
				CBOW_PATH = DATASET_PATH + "CBOW.p"
				MODEL_PATH = DATASET_PATH + "keras.model"
			if command == "difficult":
				SHOW_LEAST_PREDICTIVE = True

			# Probably good to note I use ":" as a key-value delimiter rather than "=" cause I think it looks better

			if len(command.split(":")) > 1:
				name, val = command.split(":")
				if name == "outfile":
					OUTFILE = val
				if name == "model":
					MODEL_PATH = val
				if name == "epoch":
					NUM_EPOCH = int(val)
				if name == "cbow":
					CBOW_PATH = val
				if name == "documents":
					NUM_DOCS = int(val) 
				if name == "keywords":
					NUM_KEYWORDS = int(val)

if PREDICT and CONTINUE:
	print("Invalid combination of flags.")
	quit()

#######################################################
# These are functions from keras which needed to be   #
# slightly altered to fit my use. You can basically   #
# ignore them for the most part. They should probably #
# be moved to a different file. 					  #
#######################################################

# A basic punctutation filter with "_" removed because 
# we use "_" to delimit spaces in an n-gram
def base_filter():
    f = '!"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~'
    f = f.replace("'", '')
    f += '\t\n'
    return f


# Processes a document by splitting it up into a list of words
# (or n-grams according to the arg bigram) and filtering out
def bigram_text_to_word_sequence(text, bigram, filters=base_filter(), lower=False, split=" "):
    '''prune: sequence of characters to filter out
    '''
    if lower:
        text = text.lower()
    text = text.translate(string.maketrans(filters, split*len(filters)))
    seq = text.split(split)
    sentences = [_f for _f in seq if _f]
    return bigram(sentences)
 

# Does the same as above, but then returns a hash-based non-unique
# one-hot-encoding of the word sequences
def bigram_one_hot(text, n, bigram, filters=base_filter(), lower=False, split=" "):
    seq = bigram_text_to_word_sequence(text, bigram, filters=filters, lower=lower, split=split)
    return [(abs(hash(w)) % (n - 1) + 1) for w in seq]



# This section triggers if you supply the --generate flag, indicating 
# you want to recreate the training data/labels
if REGENERATE:

	print("Generating data from scratch.")

	texts = pickle.load(open(OUTFILE, 'rb'))[0]

	# This splits your list of texts into a list of sentences
	# At this point (in the training data) document borders
	# are removed.

	sentences = [item for text in texts for item in PunktSentenceTokenizer().tokenize(text.decode("utf8"))]
	sentences = [i.strip(' \n,.;:').replace('\n', ' ').split(' ') for i in sentences]

	# Create and train bigram/trigram converters
	unigram = Phrases(sentences, threshold=float("inf"))
	unigrams = unigram.export_phrases(sentences)

	grams = [unigram]

	sentences_copy = sentences

	threshold = 9.0

	while True:
		bigram = Phrases(sentences_copy, threshold=threshold)
		bigrams = bigram.export_phrases(sentences_copy)
		z = list(set(bigrams) - set(unigrams))
		if len(z) == 0:
			break
		else:
			sentences_copy = bigram[sentences_copy]
			unigrams = bigrams
			grams.append(bigram)
			threshold += 1

	def gram_er(sentences):
		temp = sentences
		for g in grams:
			temp = g[temp]
		return temp

	num_words = len(set([i for k in sentences for i in k]))

	# Convert your texts into one-hot-encoded form
	# Sequences is used for keeping track of the true
	# missing word for every context

	one_hot_sentences = [bigram_one_hot(i, max_num_words, gram_er) for i in texts]
	sequences = [bigram_text_to_word_sequence(i, gram_er) for i in texts]

	missing_words = []

	doc_lens = [len(i) for i in one_hot_sentences]


	# Iterate over all your one-hot-encoded sentences
	# At each iteration, generate a WINDOW_SIZE-width
	# context window. Pop out the center word and store
	# it in labels (and the string-version in missing_words)
	# and store the context in data

	data = []
	labels = []
	i = 0
	for text in one_hot_sentences:
		for start in range(len(text)-WINDOW_SIZE):
			temp = text[start:start+WINDOW_SIZE]
			temp_words = sequences[i][start:start+WINDOW_SIZE]
			missing_words.append(temp_words[WINDOW_SIZE/2])
			label = temp[WINDOW_SIZE/2]
			temp = temp[:WINDOW_SIZE/2] + temp[WINDOW_SIZE/2+1:]
			data.append(temp)
			labels.append(label)
		i += 1

	# Save the results to disk

	data = np.array(data)
	labels = np.array(labels)
	f = open(CBOW_PATH, "wb")
	pickle.dump([data, labels, max_num_words, missing_words, doc_lens], f)
	f.close()


# If you chose not to regenerate, instead you
# want to load your training data from disk
else:

	print("Loading data.")

	f = open(CBOW_PATH, "rb")
	data, labels, num_words, missing_words, doc_lens = pickle.load(f)
	doc_lens = [0] + doc_lens
	f.close()


# This section triggers if you supply the --train flag
# This means you want to recreate and retrain the 
# neural network from scratch
if TRAIN or CONTINUE:

	print("Creating network topology and compiling.")

	# If --train is flagged this section creates the neural network. It is:
	#	1) An Input layer expecting a (WINDOW_SIZE-1)x1 input,
	#		which is the size of a context window.
	#	2) An embedding layer to convert to a dense dim_proj vector
	#		2a) A flatten layer required for technical reasons
	#			unimportant for the comprehension of the network
	#	3) A dense layer to return to a one-hot-encoded predict
	#	4) A softmax activation layer
	if TRAIN:
		i = Input(shape=(WINDOW_SIZE-1,), dtype='int32')
		e = Embedding(output_dim=dim_proj, input_dim=num_words, input_length=WINDOW_SIZE-1)(i)

		flatten = Flatten()(e)
		hidden = Dense(num_words, activation='softmax')(flatten)

		model = Model(input=i, output=hidden)

		# This compiles our network using ADAM with default parameters
		# (except for learning rate: ours is smaller than default)

		adam = Adam(lr=lr)

		model.compile(optimizer=adam, loss='sparse_categorical_crossentropy')

	# Else, if --continue is flagged, reload the problem from disk.
	else:
		model = load_model(MODEL_PATH)

	print("Ready to Fit!")

	# This section splits our data and labels up into batches and trains over them

	data_split, labels_split = np.array_split(data, data.shape[0]/BATCH_SIZE, axis=0), np.array_split(labels, labels.shape[0]/BATCH_SIZE)

	# There are two many samples to place into memory at once, so this is a 
	# generator to step-by-step load them into memory and feed them into the 
	# model for training. This will loop over the data indefinitely in a
	# fixed order (which should be changed)
	def data_generator():
		i = 0
		while True:
			yield (data_split[i], labels_split[i])
			i = i + 1 if i < len(data_split)-1 else 0

	# Keras doesn't come with a way to save a neural network every E epochs 
	# (only every epoch, which here takes too long to be feasible).
	# This is a simple implementation of a subclass of a Keras class, which
	# lets us save our model to disk every E epochs (here E = 100)
	class DelayedCheckpoint(Callback):
		def on_epoch_end(self, batch, logs):
			if (batch % 100) == 0 and batch != 0:
				self.model.save(MODEL_PATH)

	# This trains the network
	model.fit_generator(data_generator(), BATCH_SIZE+1, callbacks=[DelayedCheckpoint()], max_q_size=2, verbose=2, nb_epoch=NUM_EPOCH)

	# This saves the final network to disk
	model.save(MODEL_PATH)


# This section triggers if you supply the --predict flag, meaning
# that you want to predict keywords for some documents
if PREDICT:

	print("Loading model.")

	# Load the neural network from disc
	model = load_model(MODEL_PATH)

	# Keep track of list of keywords
	keywords = []

	# For each of the first NUM_DOCS documents
	for doc in range(2, NUM_DOCS + 2):

		# Load the context vectors, label vectors and label strings for the relevant document
		doc_start, doc_end = sum(doc_lens[:doc]), sum(doc_lens[:doc+1])
		first_data, first_labels, first_words = data[doc_start:doc_end, :], labels[doc_start:doc_end], missing_words[doc_start:doc_end]

		# This keeps track of the total loss for each target word encountered 
		# In the form:
		# losses[(str_form_of_word)] = (sum_of_loss_for_word, num_of_times_word_seen)
		losses = {}

		print("Predicting.")

		# I found it nice to functionize checking for stopwords
		# This would also allow us to easily filter for more advanced
		# stopping conditions like POS-type. This should also probably be
		# moved to another file
		def is_not_stopword(s, tagger):
			if len(s.split('_')) == 1:
				s = str(s)
				if s.lower() in stopwords.words('english'):
					return False 
				if tagger.tag([s])[0][1] in ["NUM", "JJ", "JJR", "JJS", "RB", "RBR", "RBS", "CD", "MD", "VBG", "VBN", "VGP", "VBD", "VBP", "VBZ", "VB", "IN"]:
					return False 
				if len(s) < 2:
					return False
				return True
			else:
				s_s = [is_not_stopword(k, tagger) for k in s.split('_')]
				return False not in s_s

		# For each context vector in the document
		for i in range(first_data.shape[0]-1):

			# You really should be able to do this efficiently in batch form, but
			# the keras API doesn't offer an obvious method. Re-writing some methods
			# might let us speed this section up quite a bit.
			# As-is: we do a separate forward pass for each context vector
			loss =  model.evaluate(first_data[i:i+1, :], first_labels[i:i+1], batch_size=1, verbose=0) 

			# Update our losses dictionary correctly
			losses[first_words[i]] = (loss, 1) if first_words[i] not in losses else (loss + losses[first_words[i]][0], 1 + losses[first_words[i]][1])


		# Instantiate a POS-Tagger
		tagger = PerceptronTagger()

		# Sort the set of words you encountered in the dictionary
		# by their average loss (in increasing order) and filter them
		# for stopwords
		s_l = sorted(losses.keys(), key=lambda x: losses[x][0]/losses[x][1])
		toPrint = []
		i = 0
		while len(toPrint) < NUM_KEYWORDS or len(toPrint) == len(s_l):
			s = s_l[i]
			i += 1
			if is_not_stopword(s, tagger):
				toPrint.append(str(s).replace("_", " "))

		# Print the top 25 keywords.
		print(toPrint)

		keywords.append(toPrint)

		# I really guessed that printing the top 25 when sorting in
		# decreasing error would be more effective, but it seems like
		# the opposite is true. Regardless, for a sanity check, I print
		# out both alternatives. Note the minus sign in the key argument.
		if SHOW_LEAST_PREDICTIVE:
			s_l = sorted(losses.keys(), key=lambda x: -losses[x][0]/losses[x][1])
			toPrint = []
			i = 0
			while len(toPrint) < NUM_KEYWORDS or len(toPrint) == len(s_l):
				s = s_l[i]
				i += 1
				if is_not_stopword(s, tagger):
					toPrint.append(str(s))
			# Print the top 25 keywords.
			print(toPrint)

	if INTERSECT and PREDICT:

		intersect_threshold = .5

		dict_of_words = {}

		for keywordSet in keywords:
			for keyword in keywordSet:
				if keyword not in dict_of_words:
					dict_of_words[keyword] = 1
					for otherSet in keywords:
						if otherSet != keywordSet:
							if keyword in otherSet:
								dict_of_words[keyword] += 1

		soft_intersection = [word for word in dict_of_words.keys() if dict_of_words[word] > intersect_threshold * len(keywords) or len(word.split(" ")) > 1]
		print(soft_intersection)