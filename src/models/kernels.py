#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
Kernel covariance functions.

[!] For some kernels, a numerical constant 'eps'
is used to address TensorFlow issue #4919.


Authored: 2017-02-25
Modified: 2018-01-06
'''

# -------- Dependencies
import numpy as np
import tensorflow as tf

__all__ =\
[
  'squared_exponential',
  'matern52',
  'matern32',
]

# ==============================================
#                                        kernels
# ==============================================
def squared_exponential(r2, eps=None):
  with tf.control_dependencies([tf.assert_non_negative(r2)]):
    return tf.exp(-0.5 * r2)


def matern52(r2, eps=None):
  if (eps is None): eps = np.finfo(r2.dtype.as_numpy_dtype).eps
  with tf.control_dependencies([tf.assert_non_negative(r2)]):
    r2 = 5.0 * tf.clip_by_value(r2, eps, float('inf'))
    r = tf.sqrt(r2)
    return (1.0 + r + (1/3)*r2) * tf.exp(-r)


def matern32(r2, eps=None):
  if (eps is None): eps = np.finfo(r2.dtype.as_numpy_dtype).eps
  with tf.control_dependencies([tf.assert_non_negative(r2)]):
    r = tf.sqrt(3.0 * tf.clip_by_value(r2, eps, float('inf')))
    return (1.0 + r) * tf.exp(-r)


# ==============================================
#                            Developers' Section
# ==============================================
# ------------ Scrap work goes here ------------
'''
import numpy as np
from src.utils import gradient_wrap
def matern52_grad(op, grad):
  r2 = op.inputs[0]; r = tf.sqrt(5.0 * r2)
  return -(5.0/6.0) * tf.exp(-r) * (1 + r) * grad

@gradient_wrap(matern52_grad, name='matern52')
def matern52(r2):
  r2 = 5.0 * r2; r = np.sqrt(r2)
  return (1.0 + r + r2/3.0) * np.exp(-r)

def matern32_grad(op, grad):
  r2 = op.inputs[0]; r = tf.sqrt(3.0 * r2)
  return -1.5 * tf.exp(-r) * grad

@gradient_wrap(matern32_grad, name='matern32')
def matern32(r2):
  r = np.sqrt(3.0 * r2)
  return (1.0 + r) * np.exp(-r)
'''