#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
Abstract base class for AMIGO tasks

Authored: 2017-01-22
Modified: 2018-01-28
'''

# --- Dependencies
import numpy as np
import numpy.random as npr
import tensorflow as tf
from abc import abstractmethod

# ==============================================
#                                  abstract_task
# ==============================================
class abstract_task(object):
  def __init__(self, input_dim=1, output_dim=1, dtype=None, 
    noise=None, stop_gradient=True, seed=None, **kwargs):
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.dtype = dtype
    self.noise = noise
    self.stop_gradient = stop_gradient
    
    # Pseudo-random number generator
    self.seed = seed
    if (self.seed is None):
      self.seed = npr.choice(int(1e6))
    self.rng = npr.RandomState(self.seed)


  @abstractmethod
  def _numpy(self, *args, **kwargs):
    pass

  @abstractmethod
  def _tensorflow(self, *args, **kwargs):
    pass


  def tensorflow(self, *args, noisy=True, stop_gradient=None, **kwargs):
    if (stop_gradient is None): stop_gradient = self.stop_gradient
    y = self._tensorflow(*args, **kwargs)
 
    if (noisy and self.noise is not None):
      rvs = tf.random_normal(tf.shape(y), dtype=y.dtype)
      y += (self.noise**0.5) * rvs

    if stop_gradient:
      y = tf.stop_gradient(y)

    return y


  def numpy(self, *args, noisy=True, **kwargs):
    y = self._numpy(*args, **kwargs)
    if (noisy and self.noise is not None):
      rvs = self.rng.randn(*y.shape).astype(y.dtype)
      y += (self.noise**0.5) * rvs
    return y



