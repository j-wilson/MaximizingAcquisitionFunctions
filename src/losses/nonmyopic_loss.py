#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
Base class for nonmyopic loss functions defined
w.r.t. a model-based posterior distribution and
a discretized input space.

Authored: 2018-01-03
Modified: 2018-06-04
'''

# ---- Dependencies
import numpy as np
import tensorflow as tf
from src import utils
from src.utils import cache_ops
from src.misc.sharded_fn import sharded_fn
from src.losses import abstract_loss
from pdb import set_trace as bp

# ==============================================
#                                 nonmyopic_loss
# ==============================================
class nonmyopic_loss(abstract_loss):
  def __init__(self, discretization=None, model=None, num_futures=32,
    num_fantasies=256, sharding_config=None, cache_rvs=False,
    run_unit_test=False, **kwargs):
    super().__init__(**kwargs)
    self.model = model
    self.discretization = discretization
    self.num_fantasies = num_fantasies
    self.num_futures = num_futures
    self.cache_rvs = cache_rvs #useful for plotting purposes
    self.run_unit_test = run_unit_test

    self.sharding_config = self.update_dict\
    ({# Configuration for sharded loss subroutines
      'monte_carlo' : {'shard_limit':2**8, 'exact_sharding':True},
    }, sharding_config)



  def intergand(self, *args, **kwargs):
    raise NotImplementedError\
    (
      '{} does not implement an integrand.'.format(self.__class__)
    )


  def evaluate(self, pools, inputs_old, outputs_old, parallelism=None,
    fantasy_rvs=None, future_rvs=None, predict_kwargs=None,
    model=None, cache=None, subroutine=None, **kwargs):
    '''
    Outermost Monte Carlo integral.
    '''
    if (model is None): model = self.model
    if (cache is None): cache = self.cache
    if (subroutine is None): subroutine = self.monte_carlo
    if (predict_kwargs is None): predict_kwargs = dict()
    if (parallelism is None): parallelism = pools.get_shape()[-2]

    # Precompute terms required downstream
    eval_args = self.get_requirements(pools, inputs_old, outputs_old,
                            cache=cache, model=model, **predict_kwargs)

    # Keyword arguments passed to subroutine
    eval_kwargs =\
    {
      **kwargs,
      'pools' : pools,
      'inputs_old' : inputs_old,
      'outputs_old' : outputs_old,
      'parallelism' : parallelism,
      'model' : model,
      'fantasy_rvs' : fantasy_rvs,
      'future_rvs' : future_rvs,
    }

    # Manually specify ID to allow sharing across pool-sizes
    eval_id = self.get_eval_id(subroutine, eval_args, **eval_kwargs)

    # [!] temp hack: store stuff for use in unit test
    if self.run_unit_test:
      self.model = model
      self.inputs_old = inputs_old
      self.outputs_old = outputs_old
      self.pools = pools

    # Build/retieve operation
    losses_op = utils.get_or_create_node\
    (
      cache=cache,
      group='loss',
      fn=subroutine,
      args=eval_args,
      kwargs=eval_kwargs,
      id=eval_id,
    )

    return losses_op


  def _monte_carlo(self, means_pool, cov_pool, xov_pd, means_disc, cov_disc,
    chol_pool=None, future_rvs=None, fantasy_rvs=None, parallelism=None,
    num_futures=None, num_fantasies=None, **kwargs):
    '''
    Generic Monte Carlo subroutine for 1-step lookahead integrands.
    Computes updated discretization posteriors conditioned on each
    sample future.
    '''
    if (num_futures is None): num_futures = self.num_futures
    if (num_fantasies is None): num_fantasies = self.num_fantasies
    with tf.name_scope('monte_carlo'.format(self.name)) as scope:
      if (chol_pool is None): chol_pool = utils.jitter_cholesky(cov_pool)

      # Fantasize residuals (Y - \mu) at pool locations
      resid_pool = self.draw_samples\
      (
        num_futures,
        chol=chol_pool,
        base_rvs=future_rvs,
        parallelism=parallelism,
      )

      # Sample pool outcomes to condition upon
      futures = resid_pool + utils.swap_axes(means_pool, -1, -2)

      # Solve for beta := K(disc, pool) K_nz(pool, pool)^{-1}
      beta = utils.swap_axes(tf.cholesky_solve(chol_pool, xov_pd), -1, -2)

      # Compute changes in discretization posterior
      d_cov = -tf.einsum('ijk,ikl->ijl', beta, xov_pd)
      ds_means = tf.einsum('ijk,ilk->ilj', beta, resid_pool)

      # Posteriors conditional on each sampled future
      means_future = means_disc + tf.expand_dims(ds_means, -1)
      covs_future = cov_disc[None, None] + d_cov[:, None]

      if self.run_unit_test:
        means_future, covs_future = self.test_conditional_posterior\
        (
          means_future,
          covs_future,
          futures
        )

      # Monte Carlo integration subroutine
      estimate = self.integrand\
      (
        means_future,
        covs_future,
        num_samples=num_fantasies,
        base_rvs=fantasy_rvs,
        futures=futures,
        means_pool=means_pool,
        means_disc=means_disc,
        cov_disc=cov_disc,
        **kwargs
      )
      return estimate


  def get_requirements(self, pools, inputs_old, outputs_old, 
    discretization=None, chol=None, model=None, cache=None,
    scope=None):
    '''
    Precompute required terms. By default, we compute posteriors for
    both the pools and discretization (given the old i/o) along with
    their posterior cross-covariance.

    [!] Wow; I'm ugly. Please improve me...
    '''
    if (discretization is None): discretization = self.discretization
    if (scope is None): scope = self.name
    if (model is None): model = self.model

    # -- Prior cross-covariances
    K_pd = self.get_or_create_node\
    (
      cache=cache,
      group=scope,
      fn=model.covar_fn,
      args=(pools, discretization),
    )

    K_po = self.get_or_create_node\
    (
      cache=cache,
      group=scope,
      fn=model.covar_fn,
      args=(pools, inputs_old),
    )

    K_od = self.get_or_create_node\
    (
      cache=cache,
      group=scope,
      fn=model.covar_fn,
      args=(inputs_old, discretization),
    )
 
    # -- Posterior cross-covariance
    # K(p, d) - K(p, o)K_nz(o,o)^{-1}K(o, d)
    if (chol is None):
      chol = model.get_or_create_node\
      (
        group='cholesky',
        fn=model.compute_cholesky,
        args=(inputs_old,),
        kwargs={'noisy':True},
        stateful=True,
      )

    beta = self.get_or_create_node\
    (
      cache=cache,
      group=scope,
      fn=tf.cholesky_solve,
      args=(chol, K_od),
    )

    d_xovs = self.get_or_create_node\
    (
      cache=cache,
      group=scope,
      fn=tf.tensordot,
      args=(K_po, beta, ((-1,), (0,))),
    )

    xov_pd = self.get_or_create_node\
    (
      cache=cache,
      group=scope,
      fn=tf.subtract,
      args=(K_pd, d_xovs),
    ) 

    # Joint posteriors over individual pools
    means_pool, cov_pool = model.get_or_create_node\
    (
      group='predict',
      fn=model.predict,
      args=(pools, inputs_old, outputs_old),
      kwargs={'full_cov':True, 'chol':chol, 'K_10':K_po},
      stateful=True,
    )

    # Joint posterior over discretization
    means_disc, cov_disc = model.get_or_create_node\
    (
      group='predict',
      fn=model.predict,
      args=(discretization, inputs_old, outputs_old),
      kwargs={'full_cov':True, 'chol':chol, 'K_01':K_od},
      stateful=True,
    )

    return means_pool, cov_pool, xov_pd, means_disc, cov_disc


  def draw_samples(self, num_samples, means=None, cov=None,
    base_rvs=None, parallelism=None, chol=None, model=None,
    name=None, **kwargs):
    if (model is None): model = self.model
    with tf.name_scope('draw_samples') as scope:
      # Compute Cholesky factor LL^{T} = K
      if (chol is None):
        jitter = getattr(model, 'jitter', None)
        chol = utils.jitter_cholesky(cov, jitter=jitter)

      # Generate i.i.d. standard normal random variables
      if (base_rvs is None):
        if (parallelism is None): parallelism = chol.get_shape()[-1]
        rvs_shape = tf.TensorShape((num_samples, parallelism))
        base_rvs = tf.random_normal(rvs_shape, dtype=chol.dtype)

      # Produce MVN samples
      residuals = tf.tensordot(chol, base_rvs, [[-1], [-1]])
      if (means is None):
        samples = utils.swap_axes(residuals, -1, -2, name=name)
      else:
        samples = utils.swap_axes(means + residuals, -1, -2, name=name)  

      return samples


  def prepare(self, sess, inputs_old, outputs_old, parallelism, model=None,
    num_fantasies=None, num_futures=None, dtype=None, **kwargs):
    '''
    Generate terms needed by relevant subroutine for computing losses.
    '''
    if (model is None): model = self.model
    if (dtype is None): dtype = self.dtype
    if (num_futures is None): num_futures = self.num_futures
    if (num_fantasies is None): num_fantasies = self.num_fantasies
    placeholders, feed_dict = dict(), dict()

    # Draw normal r.v.'s used to generate outcomes to futures
    future_shape = [num_futures, parallelism]
    future_rvs_ref = self.get_or_create_ref('future_rvs', 2)
    future_rvs_src = self.rng.randn(*future_shape).astype(dtype)
    placeholders['future_rvs'] = future_rvs_ref
    feed_dict[future_rvs_ref] = future_rvs_src

    # Draw normal r.v.'s used to fantasize at simulation locations
    fantasy_shape = [num_fantasies, self.num_discrete]
    fantasy_rvs_ref = self.get_or_create_ref('fantasy_rvs', 2)
    fantasy_rvs_src = self.rng.randn(*fantasy_shape).astype(dtype)
    placeholders['fantasy_rvs'] = fantasy_rvs_ref
    feed_dict[fantasy_rvs_ref] = fantasy_rvs_src

    if self.cache_rvs: # mainly used for plotting
      self.future_rvs = self.get_or_create_var('future_rvs', future_shape, dtype)
      sess.run(tf.assign(self.future_rvs, future_rvs_src))

      self.fantasy_rvs = self.get_or_create_var('fantasy_rvs', fantasy_shape, dtype)
      sess.run(tf.assign(self.fantasy_rvs, fantasy_rvs_src))

    return placeholders, feed_dict


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


  def sharded_subroutine(self, fn, means_pool, cov_pool, xov_pd,
    means_disc, cov_disc, sharding_config, model=None, map_kwargs=None,
    retval_shapes=None, **kwargs):
    '''
    Wrapper for sharded computation of subroutines.
    '''
    if (model is None): model = self.model

    # What sharding strategy are we using etc.?
    sharding_kwargs =\
    {
      **sharding_config,
      'retval_shapes' : retval_shapes,
      'map_kwargs' : {'dtype':model.dtype, **(map_kwargs or {})},
      # ^default to assuming subroutine only returns a tensor of losses
    }

    # Which positional arguments get sharded?
    sharded_args = (means_pool, cov_pool, xov_pd)
    constant_args = (means_disc, cov_disc)

    # Wrapped call to subroutine
    def mapped_fn(*sharded_args):
      return fn(*sharded_args, *constant_args, **kwargs)

    # Build sharded op
    return sharded_fn(mapped_fn, *sharded_args, **sharding_kwargs)


  def monte_carlo(self, means_pool, cov_pool, xov_pd, means_disc, cov_disc,
    future_rvs=None, fantasy_rvs=None, parallelism=None, num_futures=None,
    num_fantasies=None, sharding_config=None, **kwargs):
    '''
    Wrapper for sharded computation of Monte Carlo estimated losses.
    '''
    if (parallelism is None): parallelism = cov_pool.get_shape()[-2].value
    if (sharding_config is None): sharding_config = dict()

    # Ensure that normal r.v.'s are consistent across shards
    if (future_rvs is None):
      if hasattr(self, 'future_rvs'):
        future_rvs = self.future_rvs
      else:
        if (num_futures is None): num_futures = self.num_futures
        rvs_shape = tf.TensorShape((num_futures, parallelism))
        future_rvs = tf.random_normal(rvs_shape, dtype=cov_pool.dtype)

    if (fantasy_rvs is None):
      if hasattr(self, 'fantasy_rvs'):
        fantasy_rvs = self.fantasy_rvs
      else:
        if (num_fantasies is None): num_fantasies  = self.num_fantasies
        rvs_shape = tf.TensorShape((num_fantasies, self.num_discrete))
        fantasy_rvs = tf.random_normal(rvs_shape, dtype=cov_disc.dtype)

    # Update defaults with user-specified settings
    sharding_config=\
    {
      **self.sharding_config['monte_carlo'],
      **sharding_config
    }

    return self.sharded_subroutine\
    (
      self._monte_carlo, means_pool, cov_pool, xov_pd, means_disc, cov_disc,
      future_rvs=future_rvs, fantasy_rvs=fantasy_rvs,
      sharding_config=sharding_config,
      **kwargs
    )


  @property
  def num_discrete(self):
    return self.discretization.get_shape()[0].value


  def test_conditional_posterior(self, means_cond, covs_cond, pools_samples,
    discretization=None):
    if (discretization is None): discretization = self.discretization

    # # Temporarly set constant prior mean for model
    has_mean = 'mean' in self.model.state
    original_mean = self.model.mean #store reference to original
    self.model.state['mean'] = tf.reduce_mean(self.outputs_old)

    # Preallocate diagonal observation noise matrix
    noise = self.model.noise
    if (noise is None):
      I_nz = 0.0
    else:
      # Only apply noise to observed outcomes (not fantasized ones)
      I_nz = tf.diag(noise * tf.concat\
      ([
        tf.ones(tf.shape(self.inputs_old)[:1], dtype=noise.dtype),
        tf.zeros(tf.shape(self.pools)[1:2], dtype=noise.dtype),
      ], 0))

    append_inputs = lambda X: tf.concat([self.inputs_old, X], 0)
    append_outputs = lambda Y: tf.concat([self.outputs_old, Y], 0)
    def outer_mapper(mapped_args): #iterate over pools
      # Precompute precision matrix
      pool, pool_samples = mapped_args
      inputs_old_ext = append_inputs(pool)
      K_ext = self.model.covar_fn(inputs_old_ext, inputs_old_ext)
      chol_ext = utils.jitter_cholesky(K_ext + I_nz, jitter=self.model.jitter)

      def inner_mapper(pool_sample): #iterate over fantasized pool outcomes
        outputs_old_ext = append_outputs(pool_sample[..., None])
        return self.model.predict\
        (
          discretization,
          inputs_old_ext,
          outputs_old_ext,
          chol=chol_ext,
          full_cov=True,
        )

      means, covs = tf.map_fn\
      (
        inner_mapper,
        pool_samples,
        (tf.float64, tf.float64),
      )
      return means, covs[:1] #identical slices along first axis

    test_means, test_covs = tf.map_fn\
    (
      outer_mapper,
      (self.pools, pools_samples),
      (tf.float64, tf.float64),
    )

    # Restore original prior mean
    if has_mean: self.model.state['mean'] = original_mean
    else: self.model.state.pop('mean')

    # Test absolute differences
    abs_diff_covs = tf.abs(test_covs - covs_cond)
    abs_diff_means = tf.abs(test_means - means_cond)
    err_tol_covs = tf.cast(1e-8, dtype=abs_diff_covs.dtype)
    err_tol_means = tf.cast(1e-8, dtype=abs_diff_means.dtype)
    with tf.control_dependencies\
    ([
      tf.assert_less(tf.reduce_max(abs_diff_covs), err_tol_covs, name='test_cond_cov'),
      tf.assert_less(tf.reduce_max(abs_diff_means), err_tol_means, name='test_cond_mean'),
    ]):
      return tf.identity(means_cond), tf.identity(covs_cond)


# ==============================================
#                            Developers' Section
# ==============================================
# ------------ Scrap work goes here ------------
'''
'''