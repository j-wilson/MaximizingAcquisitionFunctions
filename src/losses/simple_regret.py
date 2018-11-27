#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
Simple regret.

Authored: 2017-10-19 
Modified: 2018-04-13
'''

# ---- Dependencies
import logging
import tensorflow as tf
from src.losses import myopic_loss
from pdb import set_trace as bp

# ==============================================
#                                  simple_regret
# ==============================================
class simple_regret(myopic_loss):
  def _closed_form(self, means, var, **kwargs):
    '''
    Closed-form expression for marginal, simple regret.
    '''
    with tf.name_scope('closed_form') as scope:
    	return means


  def _monte_carlo(self, means, cov, samples=None, weights=None,
    num_fantasies=None, incumbents=None, **kwargs):
    '''
    Monte Carlo estimate of parallel, simple regret.
    '''
    if (num_fantasies is None): num_samples = self.num_fantasies
    with tf.name_scope('monte_carlo') as scope:
      if (samples is None):
        samples = self.draw_samples(num_fantasies, means, cov, **kwargs)

      # Calculate sample-wise minima
      minima = tf.reduce_min(samples, axis=-1)
      if (incumbents is not None):
        minima = tf.minimum(minima, incumbents)

      # (Weighted) sample average
      estimate = self.reduce_samples(minima, weights, axis=-1)
      return estimate