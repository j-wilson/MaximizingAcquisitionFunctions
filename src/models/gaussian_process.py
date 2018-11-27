#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
TensorFlow implementation of Gaussian stochastic
process model.

Authored: 2017-02-25
Modified: 2018-10-17
'''

# ---- Dependencies
import numpy as np
import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface
from src import utils
from src.models import\
(
  abstract_model,
  priors as _priors,
  kernels as _kernels,
)

# Ensure dictionaries are ordered
from sys import version_info
if (version_info.major + 0.1 * version_info.minor < 3.6):
  from collections import OrderedDict as dict

from pdb import set_trace as bp
# ==============================================
#                               gaussian_process
# ==============================================
class gaussian_process(abstract_model):
  def __init__(self, log_lenscales=np.log(1.0), log_amplitude=np.log(1.0),
    log_noise=np.log(0.01), mean=0.0, kernel_id='matern52', priors=None,
    num_states=1, spo_config=None, jitter=None, **kwargs):
    super().__init__(**kwargs)

    if (jitter is None):
      jitter = 1e-6 if tf.as_dtype(self.dtype) == tf.float64 else 1e-4
    self.jitter = jitter
    self.kernel_id = kernel_id

    # Establish initial set of model states
    while (self.num_states < num_states):
      self.create_state(mean=mean,
                        log_lenscales=log_lenscales,
                        log_amplitude=log_amplitude,
                        log_noise=log_noise)

    # Hyperparameter priors
    self.priors =\
    {
      'log_amplitude' : _priors.log_normal(transform=tf.exp), 
      'log_noise' : _priors.horseshoe(scale=0.1, transform=tf.exp),
      # [!] Lengthscale prior directly incorporated into L-BFGS-B's bounds
      # 'log_lenscales' : _priors.tophat(upper=2, transform=tf.exp),
      **(priors or {}), #user-specified parameter priors take precedence
    }

    # Configuration for ScipyOptimize
    self.spo_config = self.update_dict\
    ({
      'method' : 'L-BFGS-B',
      'var_to_bounds' :
      {# <string> keys gets swapped out for state-specific <tf.Variable>
        'log_lenscales' : (np.log(1e-2), np.log(2.0)),
        'log_amplitude' : (np.log(1e-3), np.log(1e4)),
        'log_noise' : (-np.inf, np.log(1.0)),
      },
      'options' : {'maxiter' : 1000},
    }, spo_config)


  @abstract_model.stateful
  def log_likelihood(self, X, Y, priors=None, as_negative=False, chol=None,
    resid=None, gamma=None, noisy=True, name='log_likelihood', **kwargs):
    if (priors is None): priors = self.priors
    with tf.name_scope('log_likelihood') as scope:
      if (chol is None): chol = self.compute_cholesky(X, noisy=noisy, **kwargs)
      if (resid is None): resid = Y - self.mean_fn(X, Y)
      if (gamma is None): gamma = tf.cholesky_solve(chol, resid)

      # Compute data log likelihood
      count = tf.cast(tf.shape(X)[0], self.dtype) #num. data points
      logdet = 2.0*tf.reduce_sum(tf.log(tf.diag_part(chol)))
      data_llh = -0.5 * tf.squeeze\
      (
        tf.matmul(resid, gamma, transpose_a=True) + logdet\
        + tf.cast(tf.log(2*np.pi), self.dtype) * count
      )

      # Compute prior log likelihood
      prior_llh = list()
      for param_id, prior in priors.items():
        # Parameter either passed in or belongs to active state
        param = kwargs.get(param_id, self.state[param_id])
        prior_llh.append(prior(param))

      # Combine data-driven and prior-based terms
      log_likelihood = tf.add_n([data_llh] + prior_llh, name=name)
      if (as_negative): return tf.negative(log_likelihood)
      return log_likelihood


  @abstract_model.stateful
  def predict(self, inputs_new, inputs_old, outputs_old, use_rank2=False,
    **kwargs):
    with tf.name_scope('predict') as scope:
      if use_rank2: #use vanilla matrix algebra subroutine
        pred = self._predict_2d(inputs_new, inputs_old, outputs_old, **kwargs)
      else: #use broadcasted, n-dimensional tensor algebra subroutine
        pred = self._predict_nd(inputs_new, inputs_old, outputs_old, **kwargs)
    return pred


  def _predict_2d(self, X1, X0, Y0, full_cov=False, chol=None, precis=None,
    K_10=None, K_01=None, **kwargs):
    '''
    Subroutine for computing vanilla rank-2 GP posteriors.
    '''
    with tf.name_scope('_predict_2d') as scope:
      if (K_10 is None):
        if (K_01 is None): K_10 = self.covar_fn(X1, X0, **kwargs)
        else: K_10 = K_10 = tf.transpose(K_01)

      # Solve for $\beta := K_10 (K_00 + nz*I)^{-1}$
      if (precis is not None): #efficient but unstable...
        beta = tf.matmul(K_10, precis)
      else:
        if (chol is None): 
          chol = self.compute_cholesky(X0, noisy=True, **kwargs)
        if (K_01 is None): K_01 = tf.transpose(K_10)
        beta = tf.transpose(tf.cholesky_solve(chol, K_01))

      resid = Y0 - self.mean_fn(X0, Y0)
      mean = self.mean_fn(X1, Y0) + tf.matmul(beta, resid)
      if full_cov:
        if (K_01 is None): K_01 = tf.transpose(K_10)
        cov = tf.subtract\
        (
          self.covar_fn(X1, X1, **kwargs),
          tf.matmul(beta, K_01),
        )
        return mean, cov
      else:
        var = tf.subtract\
        (
          self.covar_fn(X1, **kwargs),
          tf.reduce_sum(beta*K_10, axis=-1, keepdims=True)
        )
        return mean, var


  def _auto_tile(self, A, B, num_ranks):
    # Rank of slices to tile (relative to the first axis)
    rank_a = A.shape.ndims
    num_ranks = num_ranks if (num_ranks > 0) else rank_a + num_ranks

    # Tile size-one axis of A
    shape_a = tf.shape(A)[:num_ranks]
    shape_b = tf.shape(B)[:num_ranks]

    tiles_a = tf.where(tf.equal(shape_a, 1), shape_b, tf.ones_like(shape_a))
    A_tiled = tf.tile(A, tf.concat([tiles_a, (rank_a - num_ranks)*[1]], 0))
    return A_tiled


  def _predict_nd(self, X1, X0, Y0, full_cov=False, chol=None,
    precis=None, K_10=None, K_01=None, **kwargs):
    '''
    Subroutine for computing broadcasted rank-n GP posteriors.

    [!] Bug: Hacks been added to handle the case where
              'X0' and 'Y0' have been padded to rank 4.
    '''
    with tf.name_scope('_predict_nd') as scope:
      if (K_10 is None):
        if (K_01 is None): K_10 = self.covar_fn(X1, X0, **kwargs)
        else: K_10 = K_10 = utils.swap_axes(K_01, -1, -2) 

      # Solve for $\beta := K_10 (K_00 + nz*I)^{-1}$
      if (precis is not None): #efficient but unstable...
        Beta = utils.broadcast_matmul(K_10, precis)
      else:
        if (chol is None):
          chol = self.compute_cholesky(X0, noisy=True, **kwargs)
        if (K_01 is None):
          K_01 = utils.swap_axes(K_10, -1, -2)

        # Solve for all systems of equations
        if (Y0.shape.ndims == 4):
          # Hack to deal with padding
          L = chol[0, 0]
          B = tf.transpose(K_01, [2, 0, 1, 3])
          X = tf.cholesky_solve(L, tf.reshape(B, [tf.shape(B)[0], -1]))
          Beta = tf.transpose(tf.reshape(X, tf.shape(B)), [1, 2, 3, 0])
        else:
          Beta = utils.swap_axes(utils.cholesky_solve(chol, K_01), -1, -2)

      resid = Y0 - self.mean_fn(X0, Y0)
      if (Y0.shape.ndims == 4):
        resid = self._auto_tile(resid, Beta, num_ranks=-2)

      Means = self.mean_fn(X1, Y0) + utils.broadcast_matmul(Beta, resid)
      if full_cov:
        if (K_01 is None): K_01 = utils.swap_axes(K_10, -1, -2)
        Covs = tf.subtract\
        (
          self.covar_fn(X1, X1, **kwargs),
          utils.broadcast_matmul(Beta, K_01)
        )
        return Means, Covs
      else:
        Vars = tf.subtract\
        (
          self.covar_fn(X1, **kwargs),
          tf.reduce_sum(Beta*K_10, axis=-1, keepdims=True),
        )
        return Means, Vars


  def mean_fn(self, X1, Y0=None, prior_mean=None):
    if (prior_mean is None): prior_mean = self.mean
    with tf.name_scope('mean_fn') as scope:
      if (prior_mean is None):
        assert Y0 is not None, 'Unable to infer GP prior mean'
        prior_mean = tf.reduce_mean(Y0, axis=-2, keepdims=True)

      M = utils.expand_as(prior_mean, X1, axis=0)
      shape_M = tf.unstack(tf.shape(M))[:-1]
      shape_X = tf.unstack(tf.shape(X1))[:-1]
      shape_X = shape_M[:max(0, len(shape_M) - len(shape_X))] + shape_X
      # ^ if X has lower rank, broadcast the mean; hence, tile by 1

      means = tf.tile(M, [tf.div(*s) for s in zip(shape_X, shape_M)] + [1])
      means.set_shape(X1.get_shape()[:-1].concatenate([1]))
      return means


  def covar_fn(self, X0, X1=None, lenscales=None, amplitude=None,
    noise=None, noisy=False, dist2=None, kernel_id=None, **kwargs):
    with tf.name_scope('covar_fn') as scope:
      if (kernel_id is None): kernel_id = self.kernel_id
      if (lenscales is None): lenscales = self.lenscales
      if (amplitude is None): amplitude = self.amplitude

      dtype = self.dtype
      try:
        kernel_fn = getattr(_kernels, kernel_id)
      except:
        raise Exception("Unrecognized kernel function '{:s}'".format(kernel_id))

      # Canonical metric for GP prior
      if (dist2 is None):
        if (X1 is not None):
          inv_scales = 1.0/lenscales
          dist2 = utils.pdist2(inv_scales*X0, inv_scales*X1)
        else:
          dist2 = tf.zeros_like(X0[...,:1])

      # Apply kernel covariance function
      corr = kernel_fn(dist2)
      cov = (amplitude**2) * corr
      if not noisy: return cov        
      return self.noise_fn(cov, noise=noise)


  def noise_fn(self, covar, noise=None, **kwargs):
    if (noise is None): noise = self.noise
    if (noise is None): return covar #model is noiseless
    def lt_rank2():
      return covar + tf.cast(noise, dtype=covar.dtype)
    def ge_rank2():
      I_noisy = noise * tf.eye(tf.shape(covar)[-1], dtype=covar.dtype)
      return covar + I_noisy
    return tf.cond(tf.less(tf.rank(covar), 2), lt_rank2, ge_rank2)


  @abstract_model.stateful
  def fit(self, sess, inputs, outputs, var_list=None, spo_config=None, 
    feed_dict=None, **kwargs):
    '''
    Fit a given model state via MAP estimation.
    '''
    assert inputs.ndim == outputs.ndim == 2, 'Tensor rank should be 2'
    if (feed_dict is None): feed_dict = dict()

    # Which <tf.Variable> should we optimize?
    if (var_list is None):
      filter_fn = lambda v: isinstance(v, tf.Variable)
      var_list = filter(filter_fn, self.state.values())
    var_list = tuple(var_list)

    # Build/retrieve negative log likelihood
    input_dim, output_dim = inputs.shape[-1], outputs.shape[-1]
    inputs_ref = self.get_or_create_ref('inputs/old', [None, input_dim])
    outputs_ref = self.get_or_create_ref('outputs/old', [None, output_dim])
    nll = self.get_or_create_node\
    (
      group='log_likelihood',
      fn=self.log_likelihood,
      args=(inputs_ref, outputs_ref),
      kwargs={**kwargs, 'state_id':self.active_state, 'as_negative':True},
    )

    # Get (updated) copy of configuration for Scipy Optimize
    spo_config = self.update_dict(self.spo_config, spo_config, as_copy=True)

    # For active parameters, replace <string> keys with <tf.Variable>
    var_to_bounds = dict()
    for key, bounds in spo_config.get('var_to_bounds', {}).items():
      for var in var_list:
        if key in var.name: #[!] too permissive, improve me...
          var_to_bounds[var] = bounds; break
    spo_config['var_to_bounds'] = var_to_bounds

    # Initialize/run optimizer
    feed_dict = {**feed_dict, inputs_ref:inputs, outputs_ref:outputs}
    optimizer = ScipyOptimizerInterface(nll, var_list, **spo_config)
    optimizer.minimize(sess, feed_dict)
    return var_list

  
  @abstract_model.stateful
  def draw_samples(self, num_samples, X1, X0=None, Y0=None, base_rvs=None,
    means=None, cov=None, name=None, **kwargs):
    if (X0 is None or Y0 is None):
      raise NotImplementedError("Sampling from the prior not yet supported")

    if (cov is None):
      means, cov = self.predict(X1, X0, Y0, full_cov=True, **kwargs)
    chol = self.compute_cholesky(X1, cov=cov)

    if (base_rvs is None):
      rvs_shape = [num_samples, tf.shape(chol)[-1]]
      base_rvs = tf.random_normal(rvs_shape, dtype=chol.dtype)

    # Produce MVN samples with shape [..., num_samples, mvn_dim]
    residuals = tf.tensordot(chol, base_rvs, [[-1], [-1]])
    if (means is None):
      # If cov is provided but means are not, samples will be zero-mean
      samples = utils.swap_axes(residuals, -1, -2, name=name)
    else:
      samples = utils.swap_axes(means + residuals, -1, -2, name=name)
    return samples


  @abstract_model.stateful
  def compute_cholesky(self, inputs, cov=None, noisy=False, **kwargs):
    with tf.name_scope('compute_cholesky') as scope:
      if (cov is None):
        cov = self.covar_fn(inputs, inputs, noisy=noisy, **kwargs)
      elif noisy:
        cov = self.noise_fn(cov, **kwargs)
      chol = utils.jitter_cholesky(cov, jitter=self.jitter)
      return chol


  @abstract_model.stateful
  def compute_precision(self, inputs, cov=None, noisy=False, **kwargs):
    with tf.name_scope('compute_precision') as scope:
      if (cov is None):
        cov = self.covar_fn(inputs, inputs, noisy=noisy, **kwargs)
      elif noisy:
        cov = self.noise_fn(cov, **kwargs)

      precis = utils.jitter_inverse(cov, jitter=self.jitter)
      return precis


  @property
  def mean(self):
    return self.state.get('mean', None)

  @property
  def log_lenscales(self):
    return self.state['log_lenscales']

  @property
  def lenscales(self):
    return tf.exp(self.log_lenscales)

  @property
  def log_amplitude(self):
    return self.state['log_amplitude']

  @property
  def amplitude(self): 
    return tf.exp(self.log_amplitude)

  @property
  def log_noise(self):
    return self.state.get('log_noise', None)

  @property
  def noise(self): 
    log_noise = self.log_noise
    if (log_noise is None): return log_noise
    return tf.exp(log_noise)



# ==============================================
#                            Developers' Section
# ==============================================
# ------------ Scrap work goes here ------------
'''
'''