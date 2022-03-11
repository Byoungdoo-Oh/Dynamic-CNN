# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import backend as K

from dcnn_layers import kMaxPooling, Fold

def Dynamic_CNN(X, y, vocab_size, embedding_dim, training=True):
	# Input layer.
	inp = tf.keras.layers.Input(shape=X.shape[1:])
	embeddings = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim,
		mask_zero=True, trainable=True)(inp)

	# Wide convolution (1).
	zero_padding_1 = tf.keras.layers.ZeroPadding1D(padding=(49, 49))(embeddings)
	conv_1 = tf.keras.layers.Conv1D(filters=64, kernel_size=50, padding='same')(zero_padding_1)

	# Dynamic k-Max Pooling (1).
	k_maxpool_1 = kMaxPooling(k=5, axis=1)(conv_1)
	nonlinear_1 = tf.keras.layers.ReLU()(k_maxpool_1) # Non-linear function.

	# Wide convolution (2).
	zero_padding_2 = tf.keras.layers.ZeroPadding1D(padding=(24, 24))(nonlinear_1)
	conv_2 = tf.keras.layers.Conv1D(filters=64, kernel_size=25, padding='same')(zero_padding_2)

	# Folding.
	folding = Fold()(conv_2)

	# Dynamic k-Max Pooling (2).
	k_maxpool_2 = kMaxPooling(k=5, axis=1)(folding)
	nonlinear_2 = tf.keras.layers.ReLU()(k_maxpool_2) # Non-linear function.

	# Flattning.
	flat = tf.keras.layers.Flatten()(nonlinear_2)

	# Output layer.
	out = tf.keras.layers.Dense(units=y.shape[-1], activation='softmax')(flat)

	model = tf.keras.models.Model(inputs=inp, outputs=out)
	return model
