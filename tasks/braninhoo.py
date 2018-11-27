#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
(Modified) Branin-Hoo benchmarking function

Authored: 2016-12-20
Modified: 2018-01-27
'''

# ---- Dependencies
import numpy as np
import tensorflow as tf
from .abstract_task import abstract_task

# ==============================================
#                                      braninhoo
# ==============================================
class braninhoo(abstract_task):
  c1 = -5.1/(4.0*np.pi**2)
  c2 = 5.0/np.pi
  c3 = 10.0 - 10.0/(8.0*np.pi)

  def __init__(self, *args, **kwargs):
    super(braninhoo, self).__init__(**kwargs)
    self.input_dim = 2
    self.minimum = 0.397887

  def _tensorflow(self, x, task_ids=None, **kwargs):
    c1 = tf.constant(self.c1, dtype=x.dtype)
    c2 = tf.constant(self.c2, dtype=x.dtype)
    c3 = tf.constant(self.c3, dtype=x.dtype)

    # Transfrom X -> Z
    z0 = 15*tf.expand_dims(x[...,0], -1)
    z1 = 15*tf.expand_dims(x[...,1], -1) - 5

    # Branin-Hoo function
    y = tf.square(z1 + c1*tf.square(z0) + c2*z0 - 6.0)\
          + c3*tf.cos(z0) + 10.0
    return y

  def _numpy(self, x, task_ids=None, **kwargs):
    c1, c2, c3 = self.c1, self.c2, self.c3

    # Transfrom X -> Z
    z0 = 15*x[...,[0]]
    z1 = 15*x[...,[1]] - 5

    # Branin-Hoo function
    y = np.square(z1 + c1*np.square(z0) + c2*z0 - 6.0)\
          + c3*np.cos(z0) + 10.0
    return y
