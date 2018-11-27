#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
Hartmann 3-dimensional benchmarking function

Authored: 2016-12-20
Modified: 2018-01-27
'''

# ---- Dependencies
import numpy as np
import tensorflow as tf
from .abstract_task import abstract_task

# ==============================================
#                                      hartmann3
# ==============================================
class hartmann3(abstract_task):
  alpha = [1.00, 1.20, 3.00, 3.20]
  n_dims = 3

  A = np.array([[3.0, 10, 30],
              [0.1, 10, 35],
              [3.0, 10, 30],
              [0.1, 10, 35]])
  P = np.array([[3689, 1170, 2673],
              [4699, 4387, 7470],
              [1091, 8732, 5547],
              [ 381, 5743, 8828]])*1e-4


  def __init__(self, *args, **kwargs):
    super(hartmann3, self).__init__(**kwargs)
    self.input_dim = 3
    self.minimum = -3.86278


  def _tensorflow(self, x, **kwargs):
    A = tf.constant(self.A, dtype=x.dtype)
    P = tf.constant(self.P, dtype=x.dtype)

    y = 0.0
    for i in range(4):
      y_i = 0.0
      for j in range(3):
        y_i += A[i, j] * tf.square(tf.expand_dims(x[...,j],-1) - P[i, j])
      y += self.alpha[i] * tf.exp(-y_i)
    y = tf.negative(y)
    return y


  def _numpy(self, x, **kwargs):
    A, P = self.A, self.P
    y = 0.0
    for i in range(4):
      y += self.alpha[i] * np.exp(-np.dot(np.square(x - P[i]), A[i]))
    return -np.expand_dims(y, -1) #restore axis lost during np.dot
