import tensorflow as tf
import numpy as np 

def deterministic_ternary(val):
	def to_ret(x):	
		return tf.round(tf.clip_by_value(x, -val, val))
	return to_ret

def stochastic_ternary(x):
	pass

def deterministic_binary(x):
	pass

def stochastic_binary(x):
	pass

def identity(x):
	return x
