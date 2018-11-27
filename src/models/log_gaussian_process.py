#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
Authored: 2018-02-20 
Modified: 2018-06-04
'''

# ---- Dependencies
import numpy as np
import tensorflow as tf
from src import utils
from src.models import gaussian_process
from pdb import set_trace as bp

# ==============================================
#                           log_gaussian_process
# ==============================================
class log_gaussian_process(gaussian_process):
  def __init__(self, *args, apply_log=True, **kwargs):
    super().__init__(*args, **kwargs)
    self.apply_log = apply_log #take the log of Y internally?

  def predict(self, new_inputs, old_inputs, old_outputs, 
    full_cov=False, apply_log=None, **kwargs):
    if (apply_log is None): apply_log = self.apply_log
    with tf.name_scope('predict') as scope:
      if apply_log: old_outputs = tf.log(old_outputs)
      log_means, log_cov = super().predict(new_inputs, old_inputs, old_outputs,
                                          full_cov=full_cov, **kwargs)

      if full_cov:
        log_var = tf.expand_dims(tf.matrix_diag_part(log_cov), -1)
        log_mv = log_means + 0.5*log_var
        means = tf.exp(log_mv)
        cov = tf.multiply\
        (
          tf.exp(log_mv + utils.swap_axis(log_mv, -1, -2)),
          tf.exp(log_cov) - 1.0,
        )
        return means, cov

      else:
        log_var = log_cov
        log_mv = log_means + 0.5*log_var
        means = tf.exp(log_mv)
        var = tf.exp(2*log_mv) * (tf.exp(log_var) - 1)
        return means, var


  def log_likelihood(self, X, Y, apply_log=None, **kwargs):
    if (apply_log is None): apply_log = self.apply_log
    if apply_log: Y = tf.log(Y)
    return super().log_likelihood(X, Y, **kwargs)


  def draw_samples(self, num_samples, X1, X0=None, Y0=None, base_rvs=None,
    means=None, cov=None, name=None, apply_log=None, **kwargs):
    if (X0 is None or Y0 is None):
      raise NotImplementedError("Sampling from the prior not yet supported")

    if (apply_log is None): apply_log = self.apply_log
    if apply_log: Y0 = tf.log(Y0)

    if (cov is None):
      log_means, log_cov = super().predict(X1, X0, Y0, full_cov=True, **kwargs)
    log_samples = super().draw_samples(num_samples, X1, 
                                        means=log_means,
                                        cov=log_cov, **kwargs)

    return tf.exp(log_samples)


