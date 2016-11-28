import tensorflow as tf
import numpy as np 

def deterministic_ternary(val):
	def to_ret(x):	
		return tf.constant(val, tf.float32) * tf.sign(tf.round(x))
	return to_ret

def stochastic_ternary(x):
	pass

def deterministic_binary(x):
	pass

def stochastic_binary(x):
	return tf.sign(tf.select(tf.equal(x, 0), 1, x))
	
def identity(x):
	return x
