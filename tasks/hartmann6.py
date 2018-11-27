#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
Hartmann 6-dimensional benchmarking function

Authored: 2016-12-20
Modified: 2018-01-27
'''

# ---- Dependencies
import numpy as np

import tensorflow as tf
from .abstract_task import abstract_task

# ==============================================
#                                      hartmann6
# ==============================================
class hartmann6(abstract_task):
  alpha = [1.00, 1.20, 3.00, 3.20]
  A = np.array([[10.00, 3.00,  17.00, 3.50,  1.70,  8.00],
                [0.05,  10.00, 17.00, 0.10,  8.00,  14.00],
                [3.00,  3.50,  1.70,  10.00, 17.00, 8.00],
                [17.00, 8.00,  0.05,  10.00, 0.10,  14.00]])
  P = np.array([[1312, 1696, 5569, 124,  8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381]])*1e-4

  def __init__(self, *args, **kwargs):
    super(hartmann6, self).__init__(**kwargs)
    self.input_dim = 6
    self.minimum = -3.32237


  def _tensorflow(self, x, **kwargs):
    dtype = x.dtype
    A = tf.constant(self.A, dtype=dtype)
    P = tf.constant(self.P, dtype=dtype)

    y = 0.0
    for i in range(4):
      y_i = 0.0
      for j in range(6):
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
