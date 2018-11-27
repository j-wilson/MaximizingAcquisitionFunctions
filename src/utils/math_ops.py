#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
Utility methods

Authored: 2016-12-10
Modified: 2018-05-30
'''

# -------- Dependencies
import numpy as np
import tensorflow as tf
from src.utils.linalg_ops import tril

__all__ =\
[
  'soft_less',
  'heaviside',
  'cumulative_min',
  'diff',
  'safe_div',
  'safe_log',
  'pnorm',
  'cma',
  'rank_vals',
]


# -------- Constants
_inv_sqrt2 = 1.0 / np.sqrt(2.0)
_inv_sqrt2pi = 1.0 / np.sqrt(2.0 * np.pi)


# ==============================================
#                                       math_ops
# ==============================================
def heaviside(X, continuity=None, eps=None):
  '''
  Heaviside step function.
  '''
  if (eps is None): eps = np.finfo(X.dtype.as_numpy_dtype).eps
  with tf.name_scope('heaviside') as scope:
    if continuity.lower() in ['l', 'left']:
      # Left-continuous Heaviside: H(0) = 0
      X = X - eps
    elif continuity.lower() in ['r', 'right']:
      # Right-continuous Heaviside: H(0) = 1
      X = X + eps 
    return 0.5 + 0.5*tf.sign(X)


def soft_less(A, B, temperature=1e-3, axis=-1, rank=None, eps=None):
  '''
  Computes a differentiable approximation to: A < B. As 'temperature'
  goes to zero, the approximation becomes exact (up to machine precision).
  '''
  with tf.name_scope('soft_less') as scope:
    diffs = B - A
    if (rank is None): rank = len(list(diffs.get_shape()))
    if (eps is None): eps = np.finfo(diffs.dtype.as_numpy_dtype).eps

    diffs = diffs - eps #break ties
    inv_temp = tf.reciprocal(tf.cast(temperature, diffs.dtype))
    soft_lts = tf.sigmoid(inv_temp * diffs)
    return soft_lts


def cumulative_min(tensor, maxval=None):
  '''
  Cumulative minimum along the last axis.
  '''
  if (maxval is None): maxval = tf.reduce_max(tensor)
  shape = tf.shape(tensor)
  pad = tf.zeros(tf.concat([shape, [1]], 0), dtype=tensor.dtype)
  X = tf.expand_dims(tensor, -2) + pad
  return tf.reduce_min(tril(X - maxval), axis=-1) + maxval


def diff(tensor, axis=-1, rank=None):
  '''
  Computes element-wise differences along a given axis.
  '''
  if (rank is None): rank = tensor.get_shape().ndims
  if (axis < 0): axis = rank + axis
  A = (rank - 1) * [slice(None)]; A.insert(axis, slice(1, None))
  B = (rank - 1) * [slice(None)]; B.insert(axis, slice(0, -1))
  return tensor[A] - tensor[B]


def safe_div(numerator, denominator):
  '''
  Returns zero-divide-zero as zero
  '''
  with tf.name_scope('safe_div') as scope:
    return tf.where\
    (
      tf.logical_and\
      (
        tf.equal(numerator, 0),
        tf.equal(denominator, 0),
      ),
      tf.zeros_like(numerator),
      tf.div(numerator, denominator),
      name='safe_div',
    )


def safe_log(src, lower=None, upper=float('inf')):
  '''
  Returns log(src \in [lower, upper])
  '''
  with tf.name_scope('safe_log') as scope:
    dtype = src.dtype
    if (lower is None):
      lower = tf.constant(1e-16 if dtype==tf.float32 else 1e-32, dtype=dtype)
    else:
      lower = tf.constant(lower, dtype=dtype)
    return tf.log(tf.clip_by_value(src, lower, upper))


def pnorm(p, x, root=True, axis=None):
  '''
  Compute the p-norm of tensor x.
  '''
  with tf.name_scope('pnorm') as scope:
    x = tf.abs(x) ** p
    if (axis is not None):
      norm = tf.reduce_sum(x, axis)
    else:
      norm = tf.reduce_sum(x)
    if root and p != 1:
      norm = norm ** (1.0/p)
    return norm


def cma(count, old, new, mask=None):
  '''
  Cumulative moving average. If a mask is provided,
  only updates unmasked elements
  '''
  tf.Assert(tf.greater_equal(count, 0), [count])
  if count.dtype != old.dtype:
    count = tf.cast(count, old.dtype)
  if (mask is not None):
    new = (1.0 - mask)*old + mask*new
  return (count*old + new)/(count + 1.0)


def rank_vals(tensor, prev_ranks=None, axis=-1, dtype=None): 
  def numpy_subroutine(ndarray, prev_ranks):
    '''
    Numpy subroutine for computing/updating rankings.
    '''
    n_dims = ndarray.ndim
    n_vals = ndarray.shape[axis]
    dtype = prev_ranks.dtype
    if (prev_ranks.size <= 1):
      # Compute ranks from scratch
      ranks = np.argsort(np.argsort(ndarray,
                    axis=axis), axis=axis).astype(dtype)
    else:
      # Update ranks to account to new element along the specified axis
      slices = [slice(None)]*(axis if axis >= 0 else n_dims+axis)
      mask = np.greater(ndarray[slices + [slice(0, -1)]],
                ndarray[slices + [slice(-1, n_vals)]]).astype(dtype=dtype)
      ranks = np.concatenate([(prev_ranks + mask), n_vals - 1\
                     - np.sum(mask, axis=axis, keepdims=True)], axis=axis)
    return ranks

  checks = []
  if (dtype is None):
    dtype = prev_ranks.dtype if prev_ranks is not None else tensor.dtype
  if (prev_ranks is None):
    prev_ranks = tf.zeros([], dtype=dtype)
  else:
    # Rank one update
    prev_ranks = tf.cast(prev_ranks, dtype=dtype)
    checks += [tf.assert_equal(tf.shape(prev_ranks)[axis],
                               tf.shape(tensor)[axis] - 1)]

  with tf.control_dependencies(checks):
    ranks = tf.py_func(numpy_subroutine, [tensor, prev_ranks], [dtype])[0]
    ranks.set_shape(tensor.shape[axis])
    return ranks
