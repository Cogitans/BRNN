import numpy as np
from keras.callbacks import Callback
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras import backend as K
import matplotlib.pyplot as plt
import time

class DelayedCheckpoint(Callback):
	def on_epoch_end(self, batch, logs):
		if batch == 0:
			sys.stdout.write("Finished Epoch 1.")
			sys.stdout.flush()
		else:
			restart_line()
			sys.stdout.write("Finished Epoch {0}".format(batch+1))
			sys.stdout.flush()
		if (batch % 100) == 0 and batch != 0:
			self.model.save(MODEL_PATH)

class QuanizationUpdate(Callback):
	def on_train_begin(self, logs):
		self.realWeights = [layer.get_weights() for layer in self.model.layers]

	def on_batch_begin(self, batch, logs):
		self.Wold = [layer.get_weights() for layer in self.model.layers]

	def on_batch_end(self, batch, logs):
		pass
		layerWeights = [layer.get_weights() for layer in self.model.layers]
		for index, layer in enumerate(self.model.layers): 
			oldWeights = self.Wold[index]
			newWeights = layerWeights[index]
			gradientUpdate = [oldWeights[i] - newWeights[i] for i in xrange(len(newWeights))]
			self.realWeights[index] = [np.clip(self.realWeights[index][i] - gradientUpdate[i], -1, 1) for i in xrange(len(newWeights))] 
			if isinstance(layer, QuantifiedStem):
				quantifiedWeights = []
				for matrix in self.realWeights[index]:
					quantifiedWeights.append(layer.quantify(matrix))
				layer.set_weights(quantifiedWeights)


class QuantifiedStem():
	def __init__(self):
		self.reasonForExistence = "Be a singleton"



class QuantRNN(SimpleRNN, QuantifiedStem):
    '''Fully-connected RNN where the output is to be fed back to input.
    	With quantization.
    '''
    def __init__(self, output_dim, quant=None, **kwargs):
        self.quantify = quant
        SimpleRNN.__init__(self, output_dim, consume_less="mem", **kwargs)

class QuantGRU(GRU, QuantifiedStem):
    def __init__(self, output_dim, quant=None, **kwargs):
        self.quantify = quant
        GRU.__init__(self, output_dim, consume_less="mem", **kwargs)

class QuantLSTM(LSTM, QuantifiedStem):
    def __init__(self, output_dim, quant=None, **kwargs):
        self.quantify = quant
        LSTM.__init__(self, output_dim, consume_less="mem", **kwargs)	