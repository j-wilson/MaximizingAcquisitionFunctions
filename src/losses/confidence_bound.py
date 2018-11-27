#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
Upper/Lower confidence bound.

Authored: 2017-10-19
Modified: 2018-04-25
'''

# ---- Dependencies
import logging
import tensorflow as tf
from src import utils
from src.losses import myopic_loss

# ---- Constants
from math import pi as _pi
_half_pi = 0.5 * _pi

# ==============================================
#                               confidence_bound
# ==============================================
class confidence_bound(myopic_loss):
  def __init__(self, beta=2.0, lower=True, transform_samples=True,
    **kwargs):
    super().__init__(**kwargs)
    self.beta = beta
    self.lower = lower
    self.transform_samples = transform_samples


  def _closed_form(self, means, var, beta=None, lower=None, **kwargs):
    '''
    Closed-form expression for upper/lower, marginal confidence bound.
    '''
    if (beta is None): beta = self.beta
    if (lower is None): lower = self.lower
    with tf.name_scope('closed_form') as scope:
      if lower: bounds = means - tf.sqrt(beta * var)
      else: bounds = means + tf.sqrt(beta * var)
      return bounds


  def _monte_carlo(self, means, cov, samples=None, weights=None, beta=None,
    lower=None, num_fantasies=None, transform_samples=None, parallelism=None,
    incumbents=None, **kwargs):
    '''
    Monte Carlo estimate to q-LCB/q-UCB.
    '''
    if (lower is None): lower = self.lower
    if (num_fantasies is None): num_fantasies = self.num_fantasies
    if (transform_samples is None): transform_samples = self.transform_samples
    with tf.name_scope('monte_carlo') as scope:
      if (samples is None):
        samples = self.draw_samples(num_fantasies, means, cov, 
                                beta=beta, lower=lower, **kwargs)

      elif transform_samples: #transform samples drawn from N(means, cov)
        # [!] Improve me, assumed shapes: 
        #   `means`   = [..., parallelism, 1]
        #   `samples` = [..., num_samples, parallelism]
        mu = utils.swap_axes(means, axis1=-1, axis2=-2)
        residuals = self.transform_residuals(samples - mu, beta, lower)
        samples = mu + residuals

      # Get sample extrema
      extrema_fn = tf.reduce_min if lower else tf.reduce_max
      extrema = extrema_fn(samples, axis=-1)
      if (incumbents is not None):
        if lower: extrema = tf.minimum(extrema, incumbents)
        else: extrema = tf.reduce_max(extrema, incumbents)

      # (Weighted) sample average
      estimate = self.reduce_samples(extrema, weights, axis=-1)
      return estimate


  def draw_samples(self, num_samples, means, cov, lower=None, beta=None,
    name=None, **kwargs):
    '''
    Draw samples from the positive orthant of a multivariate normal
    distribution with rescaled covariance 
        \tilde{\Sigma} := (0.5 * \pi * \beta) * \Sigma.
    '''
    if (lower is None): lower = self.lower
    if (beta is None): beta = self.beta
    with tf.name_scope('draw_samples') as scope:
      # Sample and transform Gaussian residuals
      raw_residuals = super().draw_samples(num_samples, None, cov, **kwargs)
      residuals = self.transform_residuals(raw_residuals, beta, lower)

      # Re-center samples
      if (means is not None):
        samples = tf.add(residuals, utils.swap_axes(means, -1, -2), name=name)
      else:
        samples = tf.identity(residuals, name=name)
      return samples


  def transform_residuals(self, residuals, beta=None, lower=None):
    '''
    Map Gaussian residuals $\gamma := Y - \mu$ as either
      UCB : $\gamma -> \sqrt(0.5*\pi*\beta) * \abs(\gamma)$
      LCB : $\gamma -> -\sqrt(0.5*\pi*\beta) * \abs(\gamma)$
    '''
    if (beta is None): beta = self.beta
    if (lower is None): lower = self.lower
    with tf.name_scope('transform_residuals') as scope:
      scale = (_half_pi * beta) ** 0.5
      new_residuals = tf.abs(scale * residuals)
      if lower: return tf.negative(new_residuals) #flip sign for LCB
      return new_residuals



# ==============================================
#                            Developers' Section
# ==============================================
# ------------ Scrap work goes here ------------
'''
'''