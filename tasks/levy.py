#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
n-dimensional Levy benchmarking function

Authored: 2018-04-25 
Modified: 2018-04-25
'''


# ---- Dependencies
import numpy as np
import tensorflow as tf
from .abstract_task import abstract_task

# ==============================================
#                                           levy
# ==============================================
class levy(abstract_task):
  def __init__(self, input_dim, *args, **kwargs):
    super().__init__(**kwargs)
    self.input_dim = input_dim
    self.minimum = 0.0


  def _numpy(self, x, **kwargs):
    z = 5.0*x - 1.75
    q = np.pi * z
    r = np.square(z - 1)
    sin2 = lambda x: np.square(np.sin(x))
    term1 = sin2(q[...,:1])
    term2 = np.sum(r[...,:-1]*(1+10*sin2(1 + q[...,:-1])), -1, keepdims=True)
    term3 = r[...,-1:] * (1 + sin2(2*q[...,-1:]))
    f = term1 + term2 + term3
    return f


  def _tensorflow(self, x, **kwargs):
    z = 5.0*x - 1.75
    q = np.pi * z
    r = tf.square(z - 1)
    sin2 = lambda x: tf.square(tf.sin(x))
    term1 = sin2(q[...,:1])
    term2 = tf.reduce_sum(r[...,:-1]*(1+10*sin2(1 + q[...,:-1])), -1, keepdims=True)
    term3 = r[...,-1:] * (1 + sin2(2*q[...,-1:]))
    f = term1 + term2 + term3
    return f