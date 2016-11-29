import tensorflow as tf
import numpy as np 
from keras import backend as K

def deterministic_ternary(val):
	def to_ret(x):	
		return tf.constant(val, tf.float32) * tf.sign(tf.round(x))
	return to_ret

def stochastic_ternary(val):
	def to_ret(x):
		x_1 = tf.constant(val, tf.float32) * tf.sign(x)
		s = K.hard_sigmoid(tf.abs(x))
		rand = tf.random_uniform(x.get_shape())
		return tf.select(tf.less(rand, s), x_1, tf.zeros_like(x_1))
	return to_ret

def deterministic_binary(x):
	pass

def stochastic_binary(x):
	return tf.sign(tf.select(tf.equal(x, 0), 1, x))
	
def identity(x):
	return x
