#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
Base class for myopic loss functions defined
w.r.t. a model-based posterior distribution.

Authored: 2017-12-27 
Modified: 2018-06-04
'''

# ---- Dependencies
import logging
import numpy as np
import tensorflow as tf
from src import utils
from src.utils import cache_ops
from src.misc.sharded_fn import sharded_fn
from src.misc import random_feature_mapping as rf_ops
from src.losses import abstract_loss
from pdb import set_trace as bp

logger = logging.getLogger(__name__)
# ==============================================
#                                    myopic_loss
# ==============================================
class myopic_loss(abstract_loss):
  def __init__(self, model=None, num_fantasies=2**10, num_rand_features=2**10,
    sharding_config=None, use_rand_features=False, cache_rvs=False, **kwargs):
    super().__init__(**kwargs)
    self.model = model
    self.num_fantasies = num_fantasies
    self.num_rand_features = num_rand_features
    self.use_rand_features = use_rand_features
    self.cache_rvs = cache_rvs

    self.sharding_config = self.update_dict\
    ({# Configuration for sharded loss subroutines
      'closed_form' : {'shard_limit':2**16, 'exact_sharding':True},
      'monte_carlo' : {'shard_limit':2**12, 'exact_sharding':True},
    }, sharding_config)


  def _closed_form(self, *args, **kwargs):
    raise NotImplementedError\
    (
      '{} does not implement a sharded, closed-form subroutine.'\
      .format(self.__class__)
    )


  def _monte_carlo(self, samples, *args, **kwargs):
    raise NotImplementedError\
    (
      '{} does not implement a sharded, Monte Carlo subroutine.'\
      .format(self.__class__)
    )


  def evaluate(self, pools, inputs_old, outputs_old, parallelism=None,
    fantasy_rvs=None, subroutine=None, model=None, cache=None, scope=None,
    use_rand_features=None, **kwargs):
    if (model is None): model = self.model
    if (scope is None): scope = self.name
    if (cache is None): cache = self.cache
    if (parallelism is None): parallelism = pools.get_shape()[-2].value
    if (use_rand_features is None): use_rand_features = self.use_rand_features

    # Positions/keyword arguments passed to subroutine
    eval_args = (pools, inputs_old, outputs_old)
    eval_kwargs =\
    {
      **kwargs,
      'parallelism' : parallelism,
      'model' : model,
      'fantasy_rvs' : fantasy_rvs,
      'use_rand_features' : use_rand_features,
    }

    # Identify appropriate subroutine
    if (subroutine is None):
      if parallelism == 1 and not use_rand_features:
        subroutine = self.closed_form
        eval_id = None
      else:
        subroutine = self.monte_carlo
        eval_id = self.get_eval_id(subroutine, eval_args, **eval_kwargs)
    else:
      eval_id = None

    # Build/retieve operation
    losses_op = self.get_or_create_node\
    ( 
      cache=cache,
      group=scope,
      fn=subroutine,
      args=eval_args,
      kwargs=eval_kwargs,
      id=eval_id,
    )

    return losses_op


  def sharded_subroutine(self, fn, pools, inputs_old, outputs_old,
    sharding_config, model=None, map_kwargs=None, retval_shapes=None,
    posteriors=None, predict_kwargs=None, use_rand_features=None,
    rf_kwargs=None, **kwargs):
    '''
    Wrapper for sharded computation of subroutines.
    '''
    if (use_rand_features is None): use_rand_features = self.use_rand_features
    if (predict_kwargs is None): predict_kwargs = dict()
    if (rf_kwargs is None): rf_kwargs = dict()
    if (model is None): model = self.model

    # What sharding strategy are we using etc.?
    sharding_kwargs =\
    {
      **sharding_config,
      'retval_shapes' : retval_shapes,
      'map_kwargs' : (map_kwargs or {}),
    }

    # Provide default dtypes for mapped function retvals
    if sharding_kwargs['map_kwargs'].get('dtype') is None:
      if use_rand_features: dtypes = pools.dtype
      else: dtypes = (pools.dtype, 3*(pools.dtype,))
      sharding_kwargs['map_kwargs']['dtype'] = dtypes

    # Determine which positional arguments to shard
    rank_new = len(pools.get_shape().as_list())
    rank_old = len(inputs_old.get_shape().as_list())
    if rank_new == rank_old > 2:
      sharded_args = (pools, inputs_old, outputs_old)
      constant_args = tuple()
    else:
      sharded_args = (pools,)
      constant_args = (inputs_old, outputs_old)

    if (posteriors is not None):
      sharded_args = sharded_args + (posteriors,)

    # Wrapped call to subroutine
    def mapped_fn(*sharded_args, kwargs=kwargs):
      pools = sharded_args[0]
      if use_rand_features:
        theta_mu = tf.transpose(self.mapping.mu)
        means = self.mapping(pools, **{**rf_kwargs, 'theta':theta_mu})

        samplesT = self.mapping(pools, **rf_kwargs)
        samples = utils.swap_axes(samplesT, -1 , -2)
        return fn(means, None, samples=samples, pools=pools,
                  inputs_old=inputs_old, outputs_old=outputs_old,
                  **kwargs)

      if (posteriors is not None):
        # Reuse precomputed posteriors
        means, cov, chol = sharded_args[-1]
        sharded_args = sharded_args[:-1]
      else:
        # Compute marginal/joint posterior individual pools
        means, cov = model.get_or_create_node\
        (
          group='predict',
          fn=model.predict,
          args=sharded_args + constant_args,
          kwargs=predict_kwargs,
          stateful=True,
        )
        chol = utils.jitter_cholesky(cov, jitter=model.jitter)

      losses = fn(means, cov, pools=pools, inputs_old=inputs_old,
                  outputs_old=outputs_old, chol=chol, **kwargs)

      return losses, (means, cov, chol)

    # Build sharded op
    return sharded_fn(mapped_fn, *sharded_args, **sharding_kwargs)


  def closed_form(self, pools, inputs_old, outputs_old, predict_kwargs=None,
    sharding_config=None, **kwargs):
    '''
    Wrapper for sharded computation of closed-form losses.
    '''
    if (predict_kwargs is None): predict_kwargs = dict()
    if (sharding_config is None): sharding_config = dict()

    # Update defaults with user-specified settings
    predict_kwargs = {'full_cov':False, **predict_kwargs}
    sharding_config =\
    {
      **self.sharding_config['closed_form'],
      **sharding_config
    }

    return self.sharded_subroutine\
    (
      self._closed_form, pools, inputs_old, outputs_old,
      predict_kwargs=predict_kwargs,
      sharding_config=sharding_config,
      **kwargs
    )


  def monte_carlo(self, pools, inputs_old, outputs_old, fantasy_rvs=None,
    parallelism=None, num_samples=None, predict_kwargs=None, 
    sharding_config=None, **kwargs):
    '''
    Wrapper for sharded computation of Monte Carlo estimated losses.
    '''
    if (parallelism is None): parallelism = pools.get_shape()[-2].value
    if (predict_kwargs is None): predict_kwargs = dict()
    if (sharding_config is None): sharding_config = dict()

    # Ensure that normal r.v.'s are consistent across shards
    if (fantasy_rvs is None):
      if hasattr(self, 'fantasy_rvs'):
        fantasy_rvs = self.fantasy_rvs
      else:
        if (num_samples is None): num_samples = self.num_fantasies
        rvs_shape = tf.TensorShape((num_samples, parallelism))
        fantasy_rvs = tf.random_normal(rvs_shape, dtype=pools.dtype.base_dtype)

    # Update defaults with user-specified settings
    predict_kwargs = {'full_cov':True, **predict_kwargs}
    sharding_config=\
    {
      **self.sharding_config['monte_carlo'],
      **sharding_config
    }

    return self.sharded_subroutine\
    (
      self._monte_carlo, pools, inputs_old, outputs_old,
      base_rvs=fantasy_rvs,
      predict_kwargs=predict_kwargs,
      sharding_config=sharding_config,
      **kwargs
    )


  def draw_samples(self, num_samples, means=None, cov=None,
    base_rvs=None, parallelism=None, chol=None, model=None,
    name=None, dtype=None, **kwargs):
    if (model is None): model = self.model
    with tf.name_scope('draw_samples') as scope:
      # Compute Cholesky factor LL^{T} = K
      if (chol is None):
        jitter = getattr(model, 'jitter', None)
        chol = utils.jitter_cholesky(cov, jitter=jitter)
      elif (dtype is not None):
        chol = tf.cast(chol, dtype)

      # Generate i.i.d. standard normal random variables
      if (base_rvs is None):
        if (parallelism is None): parallelism = chol.get_shape()[-1]
        rvs_shape = tf.TensorShape((num_samples, parallelism))
        base_rvs = tf.random_normal(rvs_shape, dtype=chol.dtype)
      elif (dtype is not None):
        base_rvs = tf.cast(base_rvs, dtype)

      # Produce MVN samples
      samples = tf.tensordot(chol, base_rvs, [[-1], [-1]])
      if (means is not None):
        samples += tf.cast(means, samples.dtype)

      return utils.swap_axes(samples, -1, -2, name=name)


  def reduce_samples(self, samples, weights=None, axis=-1, keepdims=False):
    '''
    Computes the (weighted) sample average.
    '''
    if (weights is None):
      average = tf.reduce_mean(samples, axis=axis, keepdims=keepdims)
    else:
      average = tf.tensordot(weights, samples, [[-1], [axis]])
      if keepdims: average = tf.expand_dims(average, axis)
    return average


  def update(self, sess, inputs_old, outputs_old, model=None, **kwargs):
    if (model is None): model = self.model
    if self.use_rand_features:
      logger.info('Updating random feature-based approximate sample paths.')
      if getattr(self, 'mapping', None) is None:
        self.mapping = self.build_mapping(sess, inputs_old,
                              outputs_old, model, **kwargs)
      self.mapping.resample(inputs_old, outputs_old, sess=sess)


  def prepare(self, sess, inputs_old, outputs_old, parallelism, model=None,
    num_fantasies=None, dtype=None, **kwargs):
    '''
    Generate terms needed by relevant subroutine for computing losses.
    '''
    if (model is None): model = self.model
    if (dtype is None): dtype = self.dtype
    if (num_fantasies is None): num_fantasies = self.num_fantasies
    placeholders, feed_dict = dict(), dict()

    if parallelism > 1:
      # Draw normal r.v.'s used to fantasize at simulation locations
      fantasy_shape = [num_fantasies, parallelism]
      fantasy_rvs_ref = self.get_or_create_ref('fantasy_rvs', 2)
      fantasy_rvs_src = self.rng.randn(*fantasy_shape).astype(dtype)
      placeholders['fantasy_rvs'] = fantasy_rvs_ref
      feed_dict[fantasy_rvs_ref] = fantasy_rvs_src

      if self.cache_rvs: # mainly used for plotting
        self.fantasy_rvs = self.get_or_create_var('fantasy_rvs', fantasy_shape, dtype)
        sess.run(tf.assign(self.fantasy_rvs, fantasy_rvs_src))

    return placeholders, feed_dict


  def build_embedding(self, inputs_old, outputs_old, model=None,
    output_dim=None, trainable=False, dtype=None):
    '''
    Create random feature embedding used to form approximate
    sample paths.
    '''
    if (model is None): model = self.model
    if (dtype is None): dtype = model.dtype
    if (output_dim is None): output_dim = self.num_rand_features

    # Generate random feature embeddings
    embedding = rf_ops.sample_embedding\
    (
      kernel_id=model.kernel_id,
      input_dim=inputs_old.shape[-1],
      output_dim=output_dim,
      log_lenscales=model.log_lenscales,
      log_amplitude=model.log_amplitude,
      trainable=trainable,
      dtype=dtype,
    )
    embedding_params = [embedding.weights, embedding.phases]
    embedding.init_op = tf.variables_initializer(embedding_params)
    return embedding


  def build_mapping(self, sess, inputs_old, outputs_old, model=None,
    num_fantasies=None, embedding=None, trainable=False, scope='rf',
    reuse=None, dtype='float32', **kwargs):
    if (model is None): model = self.model
    if (dtype is None): dtype = model.dtype
    if (num_fantasies is None): num_fantasies = self.num_fantasies

    inputs_ref = self.get_or_create_ref('inputs/old', 2, dtype=model.dtype)
    outputs_ref = self.get_or_create_ref('outputs/old', 2, dtype=model.dtype)
    with tf.variable_scope(scope, reuse=reuse) as vs:
      if (embedding is None):
        embedding = self.build_embedding(inputs_old, outputs_old, model,
                                      trainable=trainable, dtype=dtype)
        _ = sess.run(embedding.init_op)

      # Constant mean function for approx. sample paths
      prior_mean = model.mean
      assert prior_mean is not None, 'Not setup for use of empirical mean'

      num_features = embedding.output_dim
      theta_shape = [num_fantasies, num_features]
      theta_var = tf.get_variable('theta', theta_shape, dtype,
                                  initializer=tf.random_normal_initializer,
                                  trainable=trainable)
      _ = sess.run(tf.variables_initializer([theta_var]))

      # Build random feature-based sample paths (mapping)
      mean_fn = lambda Z, *args, **kwargs: tf.cast(prior_mean, Z.dtype)
      mapping = rf_ops.random_feature_mapping(embedding, theta_var, mean_fn)

      # Build operations for (re)sampling theta parameters
      sample_op, theta_posterior = rf_ops.sample_theta\
      (
        embedding,
        num_fantasies,
        inputs_ref,
        outputs_ref - prior_mean,
        noise=model.noise,
        jitter=model.jitter,
        dtype=model.dtype,
      )
      theta_assign_ref = tf.placeholder(dtype)
      theta_assign_op = tf.assign(theta_var, theta_assign_ref)

      # Store the mean theta
      mu_op = theta_posterior[0]
      mapping.mu = tf.get_variable('theta_mu', [num_features, 1], dtype)

      mu_assign_ref = tf.placeholder(dtype)
      mu_assign_op = tf.assign(mapping.mu, mu_assign_ref)

      def resample(X, Y, sess=sess):
        _ = sess.run(embedding.init_op)
        feed_dict = {inputs_ref:X, outputs_ref:Y}
        theta_src, mu_src = sess.run([sample_op, mu_op], feed_dict)

        _ = sess.run([theta_assign_op, mu_assign_op],
                     {theta_assign_ref: theta_src,
                      mu_assign_ref: np.sqrt(2/num_features) * mu_src})
      mapping.resample = resample

      return mapping


  def get_eval_id(self, fn, args, fantasy_rvs=None, **kwargs):
    '''
    Helper function for manually specifying ID to enable
    sharing of evaluation op across pool-sizes.
    '''
    if (fantasy_rvs is not None):
      kwargs = {**kwargs, 'parallelism':fantasy_rvs.get_shape()[-1].value}
      # ^reused iff fantasy_rvs has shape [..., None]

    return cache_ops.get_id\
    (
      fn=fn,
      args=args,
      kwargs=kwargs
    )



# ==============================================
#                            Developers' Section
# ==============================================
# ------------ Scrap work goes here ------------
'''
def direct_subroutine(self, fn, pools, inputs_old, outputs_old,
  sharding_config, model=None, map_kwargs=None, retval_shapes=None,
  posteriors=None, predict_kwargs=None, use_rand_features=None, **kwargs):
  # Wrapper for direct computation of subroutines.
  if (use_rand_features is None): use_rand_features = self.use_rand_features
  if (predict_kwargs is None): predict_kwargs = dict()
  if (model is None): model = self.model

  if use_rand_features:
    samples = utils.swap_axes(self.mapping(sharded_args[0]), -1 , -2)
    return fn(None, None, samples=samples, pools=pools,
              inputs_old=inputs_old, outputs_old=outputs_old,
              **kwargs)

  if (posteriors is None):
    # Marginal/joint posterior individual pools
    means, cov = model.get_or_create_node\
    (
      group='predict',
      fn=model.predict,
      args=(pools, inputs_old, outputs_old),
      kwargs=predict_kwargs,
      stateful=True,
    )
    chol = utils.jitter_cholesky(cov)
  else:
    means, cov, chol = posteriors

  kwargs = {**kwargs, 'chol' : chol}
  return fn(means, cov, pools=pools, inputs_old=inputs_old,
            outputs_old=outputs_old, **kwargs), (means, cov, chol)
'''