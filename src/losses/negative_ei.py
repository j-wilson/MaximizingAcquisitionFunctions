#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
Negative expected improvement.

Authored: 2017-10-19 
Modified: 2018-10-17
'''

# ---- Dependencies
import logging
import tensorflow as tf
from src import utils
from src.losses import myopic_loss
from pdb import set_trace as bp

# ==============================================
#                                    negative_ei
# ==============================================
class negative_ei(myopic_loss):
  def __init__(self, level_fn=None, **kwargs):
    super().__init__(**kwargs)
    '''
    Arguments
      level_fn : function returning the level w.r.t. improvement is measured
    '''
    if (level_fn is None):
      def level_fn(outputs_old, **kwargs):
        return tf.reduce_min(outputs_old, -2, keepdims=True)
    self.level_fn = level_fn


  def _closed_form(self, means, var, levels=None, level_fn=None, **kwargs):
    '''
    Closed-form expression for negative, marginal EI.
    '''
    if (level_fn is None): level_fn = self.level_fn
    with tf.name_scope('closed_form') as scope:
      if (levels is None): levels = level_fn(means=means, **kwargs)
      resid = levels - means
      stddv = utils.tile_as(tf.sqrt(var), resid)
      zvals = utils.safe_div(resid, stddv)
      EI = tf.add(stddv*utils.normal_pdf(zvals), 
                  resid*utils.normal_cdf(zvals))
      return tf.negative(EI)


  def _monte_carlo(self, means, cov, levels=None, samples=None, weights=None,
    num_fantasies=None, incumbents=None, level_fn=None, **kwargs):
    '''
    Monte Carlo estimate of negative q-EI.
    '''
    if (level_fn is None): level_fn = self.level_fn
    if (num_fantasies is None): num_fantasies = self.num_fantasies
    with tf.name_scope('monte_carlo') as scope:
      if (samples is None): #generate samples for Monte Carlo integral
        samples = self.draw_samples(num_fantasies, means, cov, **kwargs)

      if (levels is None): #determine exceedance level(s)
        levels = level_fn(means=means, **kwargs)

      minima = tf.reduce_min(samples, axis=-1)
      if (incumbents is not None):
        minima = tf.minimum(minima, incumbents)

      improvement = tf.nn.relu(levels - minima)

      # (Weighted) sample average
      estimate = self.reduce_samples(improvement, weights, axis=-1)
      return tf.negative(estimate)

