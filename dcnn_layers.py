# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import backend as K

class kMaxPooling(tf.keras.layers.Layer):
	'''
	Implemetation of Dynamic k-Max Pooling layer,
	which was first proposed in "A Convolutional Neural Network for Modelling Sentences (Kalchbrenner et al., 2014)".
	[http://www.aclweb.org/anthology/P14-1062]

	- Reference: https://github.com/bicepjai/Deep-Survey-Text-Classification
	k-Max Pooling layer that extracts the k-highest activations from a sentence (2D Tensor).
	'''

	def __init__(self, k=1, axis=1, **kwargs):
		super(kMaxPooling, self).__init__(**kwargs)
		self.k = k
		self.axis = 1
		self.input_spec = tf.keras.layers.InputSpec(ndim=3) # tf.__version__ == 2.4.1

		assert axis in [1, 2], 'Expected shape is (samples, filters, convolved values), cannot fold along samples dimension or axis not in list [1, 2]'

		# Need to switch the axis with the last element to perform transpose for top_k elements since top_k works in last axis.
		self.transpose_perm = [0, 1, 2] # default.
		self.transpose_perm[self.axis] = 2
		self.transpose_perm[2] = self.axis

	def call(self, inputs):
		# swap sequence dimension to get top_k elements along axis=1.
		transposed_for_top_k = tf.transpose(inputs, perm=self.transpose_perm)

		# extract top_k, returns two tensor [values, indices].
		top_k = tf.nn.top_k(transposed_for_top_k, k=self.k, sorted=True, name=None)[0]

		# return back to normal dimension but now sequence dimension has only k elements
		# performing another transpose will get the tensor back to its original shape but will have k as it's axis_1 size.
		transposed_return = tf.transpose(top_k, perm=self.transpose_perm)
		return transposed_return

	def compute_output_shape(self, input_shape):
		return tuple([input_shape[0], self.k, input_shape[-1]])

class Fold(tf.keras.layers.Layer):
	def __init__(self, **kwargs):
		super(Fold, self).__init__(**kwargs)
		self.input_spec = tf.keras.layers.InputSpec(ndim=3) # tf.__version__ == 2.4.1

	def call(self, inputs):
		input_shape = inputs.get_shape().as_list()

		# split the tensor along dimension 2 into dimension_axis_size/2, which will give us 2 tensors.
		splits = tf.split(inputs, num_or_size_splits=int(input_shape[2]/2), axis=2)

		# reduce_sum of the pair of rows we have split onto.
		reduce_sums = [tf.reduce_sum(s, axis=2) for s in splits]

		# stack them up along the same axis we have reduced.
		row_reduced = tf.stack(reduce_sums, axis=2)
		return row_reduced
