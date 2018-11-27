#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
Negative probability of improvement.

Authored: 2017-10-19 
Modified: 2018-04-13
'''

# ---- Dependencies
import logging
import tensorflow as tf
from src import utils
from src.losses import myopic_loss
from pdb import set_trace as bp

# ==============================================
#                                    negative_pi
# ==============================================
class negative_pi(myopic_loss):
  def __init__(self, level_fn=None, temperature=None, **kwargs):
    '''
    Arguments
      level_fn : function returning the level w.r.t. improvement is measured
      temperature : temperature parameter for soft_less
    '''
    super().__init__(**kwargs)
    if (level_fn is None):
      level_fn = lambda outputs_old, **kwargs: tf.reduce_min(outputs_old)
    self.level_fn = level_fn
    self.temperature = temperature


  def _closed_form(self, means, var, levels=None, level_fn=None, **kwargs):
    '''
    Closed-form expression for negative, marginal PI.
    '''
    if (level_fn is None): level_fn = self.level_fn
    with tf.name_scope('closed_form') as scope:
      if (levels is None): levels = level_fn(means=means, **kwargs)
      resid = levels - means
      stddv = utils.tile_as(tf.sqrt(var), resid)
      upper_bounds = utils.safe_div(resid, stddv)
      probabilities = utils.normal_cdf(upper_bounds)
      return tf.negative(probabilities)


  def _monte_carlo(self, means, cov, levels=None, samples=None, weights=None,
    num_fantasies=None, temperature=None, incumbents=None, level_fn=None, **kwargs):
    '''
    Monte Carlo estimate of negative q-EI.
    '''
    if (level_fn is None): level_fn = self.level_fn
    if (temperature is None): temperature = self.temperature
    if (num_fantasies is None): num_fantasies = self.num_fantasies
    with tf.name_scope('monte_carlo') as scope:
      if (samples is None): #generate samples for Monte Carlo integral
        samples = self.draw_samples(num_fantasies, means, cov, **kwargs)
        
      if (levels is None):
        levels = level_fn(means=means, **kwargs)

      minima = tf.reduce_min(samples, axis=-1)
      if (incumbents is not None):
        minima = tf.minimum(minima, incumbents)

      if (temperature is None):
        # Vanilla Monte Carlo PI (non-differentiable)
        improved = tf.cast(tf.less(minima, levels), minima.dtype)
      else:
        # Relaxed Monte Carlo PI (differentiable)
        improved = utils.soft_less(minima, levels, temperature)

      # (Weighted) sample average
      estimate = self.reduce_samples(improved, weights, axis=-1)
      return tf.negative(estimate)
