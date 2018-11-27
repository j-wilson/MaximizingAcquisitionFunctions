#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
Authored: 2017-10-16 
Modified: 2018-04-02
'''

# ---- Dependencies
import numpy as np
import tensorflow as tf
from pdb import set_trace as bp

# ---- Constants
from math import pi as _pi
_2pi = 2 * _pi
_sqrt_2pi = _2pi**0.5

# ==============================================
#                                         priors
# ==============================================
class normal(object):
	def __init__(self, loc=0.0, scale=1.0, transform=None):
		self.loc = loc
		self.scale = scale
		self.transform = transform

	def __call__(self, x):
		if self.transform: x = self.transform(x)
		dtype = x.dtype
		loc = tf.cast(self.loc, dtype)
		scale = tf.cast(self.scale, dtype)

		zval = tf.divide(x - loc, tf.square(scale))
		neg_loglik = 0.5*tf.square(zval) + tf.log(_sqrt_2pi * scale)
		return tf.negative(neg_loglik)


class log_normal(object):
	def __init__(self, scale=1.0, transform=None):
		self.scale = scale
		self.transform = transform

	def __call__(self, x):
		if self.transform: x = self.transform(x)
		dtype = x.dtype
		scale = tf.cast(self.scale, dtype)
		loglik = tf.where\
		(
			tf.greater(x, 0),
			-0.5*tf.square(tf.log(x)/scale) - tf.log(_sqrt_2pi*scale*x),
			tf.fill(tf.shape(x), tf.constant(-float('inf'), dtype)),
		)
		return loglik


class horseshoe(object):
	def __init__(self, scale=1.0, transform=None):
		self.scale = scale
		self.transform = transform

	def __call__(self, x):
		if self.transform: x = self.transform(x)
		scale = tf.cast(self.scale, x.dtype)
		return tf.where\
		(
			tf.equal(x, 0),
			tf.fill(tf.shape(x), tf.cast(float('inf'), x.dtype)),
			tf.log(tf.log(1 + 3*tf.square(scale/x))), #inexact
		)


class tophat(object):
	def __init__(self, lower=-float('inf'), upper=float('inf'), transform=None):
		self.lower = lower
		self.upper = upper
		self.transform = transform

	def __call__(self, x):
		if self.transform: x = self.transform(x)
		contraint_upper = tf.greater_equal(x, self.upper)
		contraint_lower = tf.less_equal(x, self.lower)
		violations = tf.logical_or(contraint_upper, contraint_lower)
		penalties = tf.fill(tf.shape(x), tf.constant(-float('inf'), x.dtype))
		return tf.where(violations, penalties, tf.zeros_like(x))
							


