#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
Monte Carlo-based Entropy Search.

To Do:
  - Double check computed entropy terms

Authored: 2016-07-19
Modified: 2018-06-04
'''

# ---- Dependencies
import logging
import numpy as np
import tensorflow as tf
from src import utils
from src.losses import nonmyopic_loss, negative_ei
from pdb import set_trace as bp

logger = logging.getLogger(__name__)
# ==============================================
#                                 entropy_search
# ==============================================
class entropy_search(nonmyopic_loss):
  def __init__(self, *args, temperature=None, proposal_fn=None, **kwargs):
    '''
    Arguments
      temperature : relaxation temperature for Monte Carlo estimated p_min
    '''
    super().__init__(*args, **kwargs)
    if (proposal_fn is None): proposal_fn = negative_ei()
    self.proposal_fn = proposal_fn
    self.temperature = temperature

    # [!] Ugly...
    self.log_prior = tf.get_variable('entropy_search/log_prior',
                                      initializer=tf.zeros_initializer,
                                      shape=[self.num_discrete],
                                      dtype=self.dtype)


  def integrand(self, means_future, cov_future, samples=None, log_prior=None,
    temperature=None, num_samples=None, **kwargs):
    if (log_prior is None): log_prior = self.log_prior
    if (temperature is None): temperature = self.temperature
    if (num_samples is None): num_samples = self.num_fantasies  
    with tf.name_scope('integrand') as scope:
      if (samples is None):
        samples = self.draw_samples(num_samples, means_future, cov_future, **kwargs)

      if (temperature is None):
        p_min = self.calc_pmin_hard(samples)
      else:
        p_min = self.calc_pmin_soft(samples, temperature)

    H = self.calc_entropy(p_min, log_prior, axis=-1)
    return tf.reduce_mean(H, axis=-1)


  def calc_pmin_hard(self, samples, dtype=None):
    '''
    Estimate p_min via right-continuous Heaviside.
    Functionally equivalent to using, e.g., tf.argmin,
    but differentiable (gradients are zero a.e. however).
    '''
    if (dtype is None): dtype = samples.dtype
    with tf.name_scope('calc_pmin_hard') as scope:
      minima = tf.reduce_min(samples, keepdims=True, axis=-1)
      one_hots = utils.heaviside(minima - samples, 'right')
      return tf.reduce_mean(one_hots, axis=-2)


  def calc_pmin_soft(self, samples, temperature=None, dtype=None):
    '''
    Approximate p_min via softmax.
    '''
    if (temperature is None): temperature = self.temperature
    if (dtype is None): dtype = samples.dtype
    with tf.name_scope('calc_pmin_soft') as scope:
      test_temp = tf.assert_positive(temperature, name='test_temperature')
      with tf.control_dependencies([test_temp]):
        # samples = tf.expand_dims(samples, 1) #hack for plotting mult. temperatures
        soft_mins = tf.nn.softmax((-1.0/temperature) * samples, axis=-1)
      return tf.reduce_mean(soft_mins, axis=-2)


  def calc_entropy(self, p_min, log_prior=None, axis=0, **kwargs):
    log_term = utils.safe_log(p_min, lower=1e-6)
    if (log_prior is not None): log_term += log_prior
    H = -tf.reduce_sum(tf.multiply(p_min, log_term), axis=axis, **kwargs)
    return H


  def update(self, sess, inputs_old, outputs_old, model=None,
    proposal_fn=None, num_options=2**16, dtype=None, **kwargs):
    '''
    Update the underlying discretization and associated log prior
    '''
    if (proposal_fn is None): proposal_fn = self.proposal_fn
    if (model is None): model = self.model
    if (dtype is None): dtype = self.dtype

    inputs_ref = self.get_or_create_ref('inputs/old', 2, dtype=dtype)
    outputs_ref = self.get_or_create_ref('outputs/old', 2, dtype=dtype)
    options_ref = self.get_or_create_ref('inputs/new', 3, dtype=dtype)

    input_dim = inputs_old.shape[-1]
    options_src = self.rng.rand(num_options, 1, input_dim)

    losses_op = proposal_fn\
    (
      options_ref,
      inputs_ref,
      outputs_ref,
      model=model,
      parallelism=1,
    )

    feed_dict =\
    {
      inputs_ref:inputs_old,
      outputs_ref:outputs_old,
      options_ref:options_src
    }

    prior = np.ravel(-sess.run(losses_op, feed_dict))
    eps = min(1e-6, 1e-6*np.mean(prior))
    log_prior = np.log(np.clip(prior, eps, np.inf))
    weights = log_prior - np.min(log_prior) #sampling measure
    weights /= np.sum(weights)

    indices = self.rng.choice(num_options, self.num_discrete,
                                    p=weights, replace=False)

    assign_ref = self.get_or_create_ref('assign', dtype=dtype)
    assign_disc_op = self.get_or_create_node\
    (
      'assign',
      tf.assign,
      (self.discretization, assign_ref),
    )
    sess.run(assign_disc_op, {assign_ref : options_src[indices, 0, :]})

    assign_logp_op = self.get_or_create_node\
    (
      'assign',
      tf.assign,
      (self.log_prior, assign_ref),
    )
    sess.run(assign_logp_op, {assign_ref : log_prior[indices]})






# ==============================================
#                            Developers' Section
# ==============================================
# ------------ Scrap work goes here ------------
'''
'''