#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
Authored: 2018-01-04 
Modified: 2018-04-05
'''

# ---- Dependencies
import logging
import numpy as np
import tensorflow as tf
from src import utils
from src.losses import nonmyopic_loss, negative_ei
from functools import reduce
from pdb import set_trace as bp

logger = logging.getLogger(__name__)
# ==============================================
#                             knowledge_gradient
# ==============================================
class knowledge_gradient(nonmyopic_loss):
  def __init__(self, *args, proposal_fn=None, **kwargs):
    super().__init__(*args, **kwargs)
    if (proposal_fn is None): proposal_fn = negative_ei()
    self.proposal_fn = proposal_fn


  def integrand(self, means_future, cov_future, samples=None, num_samples=None,
    futures=None, outputs_old=None, means_pool=None, means_disc=None, **kwargs):
    if (num_samples is None): num_samples = self.num_fantasies
    with tf.name_scope('integrand') as scope:
      future_min = tf.reduce_min(tf.squeeze(means_future, -1), -1)

      if (futures is not None):
        future_min = tf.minimum(future_min, tf.reduce_min(futures, -1))

      present_min = []
      if (outputs_old is not None):#only correct in the noise-free setting
        present_min.append(tf.reduce_min(outputs_old))

      if (means_disc is not None):
        present_min.append(tf.reduce_min(means_disc))

      if (means_pool is not None):
        present_min.append(tf.reduce_min(means_pool, -2))

      present_min = reduce(tf.minimum, present_min)

      improvement = tf.nn.relu(present_min - future_min)
      return tf.negative(tf.reduce_mean(improvement, -1))


  def update(self, sess, inputs_old, outputs_old, model=None,
    proposal_fn=None, num_options=2**16, dtype=None, **kwargs):
    '''
    Update the underlying discretization.
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



# ==============================================
#                            Developers' Section
# ==============================================
# ------------ Scrap work goes here ------------
'''
'''