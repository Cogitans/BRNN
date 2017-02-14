import tensorflow as tf
import numpy as np 
from keras import backend as K

def deterministic_ternary(val):
	def to_ret(x):	
		return tf.select(tf.less(tf.abs(x), tf.constant(val/2, tf.float32)), tf.zeros_like(x), tf.constant(val, tf.float32)*tf.sign(x))
	return to_ret

def stochastic_ternary(val):
	def to_ret(x):
		x_1 = tf.constant(val, tf.float32) * tf.sign(x)
		s = K.hard_sigmoid(tf.abs(x))
		rand = tf.random_uniform(x.get_shape())
		return tf.select(tf.less(rand, s), x_1, tf.zeros_like(x_1))
	return to_ret

def deterministic_binary(val):
	def to_ret(x):
		return tf.constant(val, tf.float32) * tf.sign(tf.select(tf.equal(x, 0.), tf.ones_like(x), x))
	return to_ret

def stochastic_binary(val):
	def to_ret(x):
		correct = tf.sign(tf.select(tf.equal(x, 0.), tf.ones_like(x), x))
		s = K.hard_sigmoid(tf.abs(x))
		rand = tf.random_uniform(x.get_shape())
		return tf.constant(val, tf.float32) * tf.select(tf.less(rand, s), correct, -1*correct)
	return to_ret

def identity(x):
	def f(x):
		return x
	return f
