import sys
import tensorflow as tf
import numpy as np
import six

def DilatedConv_4Mehtods(type, x, k, num_out, factor, name, biased=False):
	"""
	DilatedConv with 4 method: Basic,Decompose,GI,SSC
	Args: 
	type = Basic,Decompose,GI,SSC
	x = input
	k = kernal size
	num_out output num
	factor = dilated factor
	Returns: 
	output layer
	"""
	if (type == 'Basic'): return DilatedConv_Basic(x, k, num_out, factor, name, biased)
	elif (type == 'Decompose'): return DilatedConv_Decompose(x, k, num_out, factor, name, biased)
	elif (type == 'GI'): return DilatedConv_GI(x, k, num_out, factor, name, biased)
	elif (type == 'SSC'): return DilatedConv_SSC(x, k, num_out, factor, name, biased)
	else:
		print('type ERROR! Chose a type from: Basic,Decompose,GI,SSC')
		# exit
		sys.exit(-1)

def DilatedConv_Basic(x, k, num_out, factor, name, biased=False):
	num_input = x.shape[3].value
	with tf.variable_scope(name) as scope:
		weights = tf.get_variable('weights', shape=[k, k, num_input, num_out])
		output = tf.nn.atrous_conv2d(x, weights, factor, padding='SAME')
		if biased:
			bias = tf.get_variable('biases', shape=[num_out])
			output = tf.nn.bias_add(output, bias)
		return output

def DilatedConv_Decompose(x, k, num_out, factor, name, biased=False):
	H = tf.shape(x)[1]
	W = tf.shape(x)[2]
	pad_bottom = (factor - H % factor) if H % factor != 0 else 0
	pad_right = (factor - W % factor) if W % factor != 0 else 0
	pad = [[0, pad_bottom], [0, pad_right]]
	output = tf.space_to_batch(x, paddings=pad, block_size=factor)
	num_input = x.shape[3].value
	with tf.variable_scope(name) as scope:
		w = tf.get_variable('weights', shape=[k, k, num_input, num_out])
		s = [1, 1, 1, 1]
		output = tf.nn.conv2d(output, w, s, padding='SAME')
		if biased:
			bias = tf.get_variable('biases', shape=[num_out])
			output = tf.nn.bias_add(output, bias)
	output = tf.batch_to_space(output, crops=pad, block_size=factor)
	return output

def DilatedConv_GI(x, k, num_out, factor, name, biased=False):
	H = tf.shape(x)[1]
	W = tf.shape(x)[2]
	pad_bottom = (factor - H % factor) if H % factor != 0 else 0
	pad_right = (factor - W % factor) if W % factor != 0 else 0
	pad = [[0, pad_bottom], [0, pad_right]]
	num_input = x.shape[3].value
	with tf.variable_scope(name) as scope:
		w = tf.get_variable('weights', shape=[k, k, num_input, num_out])
		s = [1, 1, 1, 1]
		output = tf.nn.conv2d(output, w, s, padding='SAME')
		fix_w = tf.Variable(tf.eye(factor*factor), name='fix_w')
		l = tf.split(output, factor*factor, axis=0)
		os = []
		for i in six.moves.range(0, factor*factor):
			os.append(fix_w[0, i] * l[i])
			for j in six.moves.range(1, factor*factor):
				os[i] += fix_w[j, i] * l[j]
		output = tf.concat(os, axis=0)
		if biased:
			bias = tf.get_variable('biases', shape=[num_out])
			output = tf.nn.bias_add(output, bias)
	output = tf.batch_to_space(output, crops=pad, block_size=factor)
	return output


def DilatedConv_SSC(x, k, num_out, factor, name, biased=False):
	num_input = x.shape[3].value
	fix_w_size = factor * 2 - 1
	with tf.variable_scope(name) as scope:
		fix_w = tf.get_variable('fix_w', 
			shape=[fix_w_size, fix_w_size, 1, 1, 1], 
			initializer=tf.zeros_initializer)
		mask = np.zeros([fix_w_size, fix_w_size, 1, 1, 1], dtype=np.float32)
		mask[factor - 1, factor - 1, 0, 0, 0] = 1
		fix_w = tf.add(fix_w, tf.constant(mask, dtype=tf.float32))
		output = tf.expand_dims(x, -1)
		output = tf.nn.conv3d(output, fix_w, strides=[1]*4, padding='SAME')
		output = tf.squeeze(output, -1)
		w = tf.get_variable('weights', shape=[k, k, num_input, num_out])
		output = tf.nn.atrous_conv2d(output, w, factor, padding='SAME')
		if biased:
			bias = tf.get_variable('biases', shape=[num_out])
			output = tf.nn.bias_add(output, bias)
		return output