#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
Unit tests for models classes.

Authored: 2016-12-28
Modified: 2018-04-16
'''

# ---- Dependencies
import os, sys, unittest
import numpy as np
import numpy.random as npr
import scipy.linalg as spla
import scipy.stats as sps
from scipy.spatial.distance import cdist
import tensorflow as tf

# Relative imports
sys.path.insert(0, os.path.join(sys.path[0], '..'))
from src import models

# ==============================================
#                                    test_models
# ==============================================
def squared_exponential(X0, X1, log_lenscales=0.0, log_amplitude=0.0,
  log_noise=None):
  # Transform kernel hyperparameters
  lenscales = np.exp(log_lenscales)
  amplitude = np.exp(log_amplitude)
  noise = None if (log_noise is None) else np.exp(log_noise)

  # Compute square Euclidean distances
  r2 = cdist(X0/lenscales, X1/lenscales, metric='sqeuclidean')

  # Calculate covariance
  cov = (amplitude**2) * np.exp(-0.5 * r2)
  if (noise is None):
    return cov
  else:
    return cov + noise*np.eye(len(cov))


def jitter_cholesky(matrix, jitter=0.0):
  return spla.cholesky(matrix + jitter*np.eye(matrix.shape[0]), lower=True)


def _gp_posterior(X1, X0, Y0, log_noise=0.0, jitter=1e-6,
  kernel=squared_exponential, mean=None, **kwargs):
  '''
  Vanilla Gaussian process posterior
  '''
  prior_mean = np.mean(Y0) if (mean is None) else mean
  resid = Y0 - prior_mean
  k_00 = kernel(X0, X0, log_noise=log_noise, **kwargs)
  k_01 = kernel(X0, X1, **kwargs)
  chol = jitter_cholesky(k_00, jitter=jitter)
  beta = spla.cho_solve((chol, True), k_01)
  mean = prior_mean + np.dot(beta.T, resid)
  cov = kernel(X1, X1, **kwargs) - np.dot(beta.T, k_01)
  return mean, cov


def gp_posterior(X1, X0, Y0, **kwargs):
  '''
  Wrapper for computing (broadcasted) n-dimensional posteriors
  '''
  rank1, rank0 = X1.ndim, X0.ndim
  if (rank1 == rank0 == 2):
    # Vanilla GP posterior
    return _gp_posterior(X1, X0, Y0, **kwargs)
  elif rank1 == rank0:
    # Prediction and observation sets are 1-to-1
    posteriors = [gp_posterior(*xxy, **kwargs) for xxy in zip(X1, X0, Y0)]
  elif rank1 > rank0:
    # Multiple predictions sets given same observation sets
    posteriors = [gp_posterior(x1, X0, Y0, **kwargs) for x1 in X1]
  else:
    # Same predictions sets given multiple observation sets
    posteriors = [gp_posterior(X1, *xy, **kwargs) for xy in zip(X0, Y0)]

  Means, Covs = map(lambda term: np.asarray(term), zip(*posteriors))
  return Means, Covs


class test_gp(unittest.TestCase):
  '''
  Unit tests for Gaussian process model.
  '''
  def __init__(self, *args, dtype='float64', error_tol=None, seed=None, **kwargs):
    super().__init__(*args, **kwargs)
    self.dtype = dtype
    if (error_tol is None): #100x should usually pass
      error_tol = 100 * np.finfo(dtype).eps
    self.error_tol = error_tol

    if (seed is None):
      seed = npr.randint(2**31 - 1)
    self.seed = seed


  def build_gp(self, kernel='squared_exponential', dtype=None,
    initialize=True, sess=None, **kwargs):
    '''
    Build and initialize a Guassian process model instance.
    '''
    if (dtype is None): dtype = self.dtype

    gp = models.gaussian_process\
    (
      kernel_id=kernel,
      dtype=dtype,
      **kwargs,
    )

    if initialize:
      if (sess is None): sess = tf.get_default_session()
      assert sess is not None, 'No TensorFlow session was found'
      params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, gp.name)
      sess.run(tf.variables_initializer(params))

    return gp


  def test_gp_init(self, num_states=2, graph=None, dtype=None,
    seed=None, error_tol=1e-16):
    '''
    Test initialization of GP model instance.
    '''
    if (error_tol is None): error_tol = self.error_tol
    if (graph is None): graph = tf.Graph()
    if (dtype is None): dtype = self.dtype
    if (seed is None): seed = self.seed
    rng = npr.RandomState(seed)

    with tf.Session(graph=graph) as sess:
      # Initialize GP w/ multiple model states
      def create_state_src():
        cast = getattr(np, dtype)
        return\
        {
          'mean' : cast(rng.rand()),
          'log_lenscales' : cast(rng.rand()),
          'log_amplitude' : cast(rng.rand()),
          'log_noise' : cast(rng.rand()),
        }
      states_src = [create_state_src() for k in range(num_states)]
      gp = self.build_gp(states=states_src, dtype=dtype)

      for state_id, state_src in enumerate(states_src):
        # Compare with direct accessing via model's state
        param_ids, param_vals = zip(*gp.states[state_id].items())
        state = {k:v for k,v in zip(param_ids, sess.run(tuple(param_vals)))}
        self.assertTrue(state == state_src)

        # Compare with property-style accessing of active state
        gp.active_state = state_id #set ID of active state
        for param_id, param_val in state_src.items():
          if hasattr(gp, param_id):
            abs_diff = np.abs(param_val - sess.run(getattr(gp, param_id)))
            self.assertTrue(abs_diff < error_tol)

          if 'log_' in param_id and hasattr(gp, param_id.replace('log_', '')):
            exp_val = getattr(gp, param_id.replace('log_', '')) #exp(param)
            abs_diff = np.abs(np.exp(param_val) - sess.run(exp_val))
            self.assertTrue(abs_diff < error_tol)


  def _test_gp_predict(self, sess, gp, num_new, num_old, input_dim=1,
    output_dim=1, rank_old=2, rank_new=2, set_size=4, dtype=None,
    seed=None, error_tol=None, scope='test_gp_predict'):
    '''
    Compare TensorFlow- and numpy-based GP predictions.
    '''
    if (error_tol is None): error_tol = self.error_tol
    if (dtype is None): dtype = self.dtype
    if (seed is None): seed = self.seed

    with tf.name_scope(scope) as scope:
      rng = npr.RandomState(seed)

      # Shape of observation/predictions sets (excl. input/output dim)
      shape_old = (rank_old - 2)*[set_size] + [num_old]
      shape_new = (rank_new - 2)*[set_size] + [num_new]

      # Generate synthetic data
      X1_src = rng.rand(*shape_new, input_dim).astype(dtype)
      X0_src = rng.rand(*shape_old, input_dim).astype(dtype)
      Y0_src = rng.rand(*shape_old, output_dim).astype(dtype)

      # Create TensorFlow placeholders
      X1_ref = tf.placeholder(dtype, name='X1', shape=X1_src.shape)
      X0_ref = tf.placeholder(dtype, name='X0', shape=X0_src.shape)
      Y0_ref = tf.placeholder(dtype, name='Y0', shape=Y0_src.shape)
      feed_dict = {X1_ref:X1_src, X0_ref:X0_src, Y0_ref:Y0_src}

      # Get source values for GP hyperparameters
      param_vals = sess.run(tuple(gp.state.values()))
      params = {k:v for k,v in zip(gp.state.keys(), param_vals)}

      # Compute numpy-based ground truth values
      mean_np, cov_np = gp_posterior\
      (
        X1_src,
        X0_src,
        Y0_src,
        jitter=gp.jitter,
        **params, #use same hyperparameter values
      )
      var_np = np.expand_dims(np.diagonal(cov_np, axis1=-2, axis2=-1), -1)

      # Compare joint posteriors & compare with groud truth
      joint_op = gp.predict(X1_ref, X0_ref, Y0_ref, full_cov=True)
      mean_tf, cov_tf = sess.run(joint_op, feed_dict=feed_dict)
      self.assertTrue(np.abs(mean_np - mean_tf).max() < error_tol)
      self.assertTrue(np.abs(cov_np - cov_tf).max() < error_tol)

      # Compare marginal posteriors with groud truth
      marginal_op = gp.predict(X1_ref, X0_ref, Y0_ref, full_cov=False)
      mean_tf2, var_tf = sess.run(marginal_op, feed_dict=feed_dict)
      self.assertTrue(np.abs(mean_np - mean_tf2).max() < error_tol)
      self.assertTrue(np.abs(var_np - var_tf).max() < error_tol)


  def test_gp_predict(self, graph=None, seed=None):
    '''
    Test GP model predictions across different usage scenarios.
    '''
    if (graph is None): graph = tf.Graph()
    if (seed is None): seed = self.seed
    rng = npr.RandomState(seed)

    with tf.Session(graph=graph) as sess:
      # Build/initialize GP
      gp = self.build_gp\
      (
        log_lenscales=np.log(0.1),
        log_amplitude=np.log(1.2),
        log_noise=np.log(1e-3),
      )
      for rank_new in range(2, 4): #rank of observation sets
        for rank_old in range(2, 4): #rank of prediction sets
          self._test_gp_predict\
          (
            sess,
            gp,
            num_new=64,
            num_old=128,
            input_dim=4,
            rank_old=rank_old,
            rank_new=rank_new
          )


if __name__ == '__main__':
  unittest.main()
