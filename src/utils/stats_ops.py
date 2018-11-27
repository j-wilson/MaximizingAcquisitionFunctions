#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
Additional statistics and probability operations

[!] To Do:
  - Test slice_sampler for usage and correctness

Authored: 2016-12-10
Modified: 2018-02-01
'''

# -------- Dependencies
import numpy as np
import tensorflow as tf
from src.utils import tensor_ops

__all__ =\
[
  'normal_pdf',
  'normal_cdf',
  'hypergeometric_sampler',
  'inv_transform_sampler',
  'random_multivariate_t',
]

# ---- Constants
_isqrt2pi = 1.0/np.sqrt(2*np.pi)
_isqrt2 = 1.0/np.sqrt(2)


# ==============================================
#                                      stats_ops
# ==============================================
def normal_pdf(x, loc=None, scale=None):
  if (loc is not None): x = tf.subtract(x, loc)
  if (scale is not None): x = tf.divide(x, scale)

  prob = _isqrt2pi * tf.exp(-0.5*x**2)
  if (scale is not None): prob /= scale
  return prob


def normal_cdf(x, loc=None, scale=None):
  if (loc is not None): x = tf.subtract(x, loc)
  if (scale is not None): x = tf.divide(x, scale)
  return 0.5 * tf.erfc(-_isqrt2 * x)


def hypergeometric_sampler(num_samples, energies, transform_fn=None, 
  pad_val=0.0, dtype='int32'):
  '''
  This isn't actually a hypergeometric distribution...just not sure what
  its formal name is...
  '''
  samples, probabilities = [], []

  dist_shape = tf.shape(energies)
  num_values = tf.size(energies)
  padding = tf.fill([num_values], tf.cast(pad_val, dtype=energies.dtype))

  def to_probs(energies):
    if callable(transform_fn):
      energies = transform_fn(energies)
    partition = tf.reduce_sum(energies, axis=-1, keep_dims=True)
    with tf.control_dependencies([tf.assert_positive(partition)]):
      return tf.divide(energies, partition)

  with tf.control_dependencies([tf.assert_rank(energies, 2)]): 
    offsets = tf.reduce_prod(dist_shape[1:]) * tf.range(dist_shape[0])

  for k in range(num_samples):
    probs = to_probs(energies)
    bins = tf.cumsum(probs, axis=-1)
    
    # Draw new samples (w/o replacement)
    rvs = tf.expand_dims(tf.random_uniform(dist_shape[:-1]), axis=-1)
    samples += [tf.reduce_sum(tf.cast(rvs > bins, dtype), axis=-1)]

    # Store (conditional) probabilities of sampled items
    indices = offsets + samples[-1]
    probabilities += [tf.gather(tf.reshape(probs, [num_values]), indices)]
      
    # Update energy values
    if k < (num_samples - 1):
      # Note: 'indices' must be promoted to N-by-1 for scatter_nd to work!
      exclude = tf.scatter_nd(indices[:,None], 
                  tf.ones([dist_shape[0]], tf.int8), [num_values])
      energies = tf.reshape(
                  tf.where(tf.cast(exclude, tf.bool), padding,
                    tf.reshape(energies, [num_values])),
                  dist_shape)

  samples = tf.stack(samples, axis=-1)
  probabilities = tf.stack(probabilities, axis=-1)
  
  return samples, probabilities


def inv_transform_sampler(num_samples, weights, replace=True, 
  transform_fn=None, pad_val=0.0, dtype='int32'):
  ftype = weights.dtype

  # Preallocate local variables
  if not replace:
    n_bins = tf.size(weights)
    pads = tf.fill([n_bins], tf.cast(pad_val, weights.dtype))

  def to_energy(weights, exclude=None):
    #Convert sample weights to unnormalized probabilites (energies)
    if (exclude is not None):
      if exclude.dtype != tf.bool:
        exclude = tf.cast(exclude, tf.bool)
      weights = tf.where(exclude, pads, weights)
    if callable(transform_fn):
      return transform_fn(weights)
    return weights

  def cond(count, samples, probs):
    return tf.less(count, num_samples)

  def body(count, samples, probs):
    # Note: samples must be promoted to N-by-1 for scatter_nd to work!
    exclude = tf.scatter_nd(samples[:,None], tf.ones([count], tf.int8), [n_bins])
    energies = to_energy(weights, exclude)
    bins = tf.cumsum(energies)

    rvs = bins[-1] * tf.random_uniform([num_samples - count, 1], dtype=ftype)
    new_samples = tf.unique(tf.reduce_sum(tf.cast(rvs > bins[None, :], dtype), -1))[0]

    samples = tf.concat([samples, new_samples], 0)
    probs = tf.concat([probs, tf.gather(energies, new_samples)/bins[-1]], 0)
    return tf.size(samples), samples, probs

  checks = []
  if not callable(transform_fn):
    checks += [tf.assert_non_negative(weights, name='neg_test')]
  if replace:
    checks += [tf.assert_greater_equal(
                tf.reduce_sum(weights > pads[0]), 
                num_samples, name='sufficiency_test')]

  with tf.control_dependencies(checks):
    energies = to_energy(weights)
    bins = tf.cumsum(energies)
    rvs = bins[-1] * tf.random_uniform([num_samples, 1], dtype=ftype)
    samples = tf.reduce_sum(tf.cast(rvs > bins[None, :], dtype), -1)

    if not replace:
      samples = tf.unique(samples)[0]
      probs = tf.gather(energies, samples)/bins[-1]
      count = tf.size(samples)
      _, samples, probs = tf.while_loop(cond, body, [count, samples, probs])
    else:
      probs = tf.gather(bins, energies)/bins[-1]

    samples.set_shape([num_samples])
    return samples, probs


def random_multivariate_t(shape, deg_freedom, dtype=None, name=None):
  '''
  Draw samples from a multivariate Student-t distribution.
  [!] Hack: assumes last term in `shape` := distribution dimensionality
  '''
  with tf.name_scope('random_multivariate_t'):
    normal_rvs = tf.random_normal(shape, dtype=dtype)
    gamma_rvs = tf.random_gamma(shape[:-1], alpha=0.5*deg_freedom,
                                beta=0.5*deg_freedom, dtype=normal_rvs.dtype)
    inv_sqrt_gamma_rvs = tf.reciprocal(gamma_rvs**0.5)[...,None]
    return tf.multiply(normal_rvs, inv_sqrt_gamma_rvs, name=name)
