#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
Bayesian optimization search agent

To Do:
  - Input warping
  - Hyperparameter marginalization
  - Reincorporate multi-task / task_ids

Authored: 2017-02-25
Modified: 2018-10-17
'''

# ---- Dependencies
import time, logging
import numpy as np
import tensorflow as tf
from functools import reduce
from tensorflow.contrib.opt import ScipyOptimizerInterface
from src import utils
from src.agents import search_agent

# Ensure dictionaries are ordered
from sys import version_info
if (version_info.major + 0.1 * version_info.minor < 3.6):
  from collections import OrderedDict as dict

from pdb import set_trace as bp
logger = logging.getLogger(__name__)
# ==============================================
#                          bayesian_optimization
# ==============================================
class bayesian_optimization(search_agent):
  def __init__(self, loss_fn, model, num_options=2**16, risk=0,
    configs=None, **kwargs):
    '''
    [!] Needs a doc string...
    '''
    super().__init__(**kwargs)
    self.loss_fn = loss_fn
    self.model = model
    self.num_options = num_options
    self.risk = risk
    self.modules = dict() #ugly...

    # Default settings for optimizers used internally
    self.configs = self.update_dict\
    ({# Random search
      'random':
      {
        'eval_limit' : 2**16,
      },

      # Stochastic gradient descent
      'sgd':
      {
        'method' : 'train.AdamOptimizer',
        'learning_rate' : 25e-3, #initial learning rate
        'use_averaging' : False,
        'step_limit' : 1000,
        'batch_size' : 128,
      },

      # Scipy Optimize
      'scipy':
      {
        'method' : 'L-BFGS-B',
        'maxiter' : 1000,
        'ftol' : 1e6*np.finfo('float64').eps,
      },

      # NLopt
      'nlopt':
      {
        'method' : 'GN_DIRECT',
        'maxfun' : 10000,
      },

      # CMA-ES
      'cmaes':
      {
        'method' : 'CMAEvolutionStrategy',
        'step_limit' : 10000,
        'stop_early' : False,
        'sigma0' : 0.5,
        'bounds' : [0, 1],
      },

      # Successive halving
      'halving':
      {
        'eval_limit' : 2**20,
        'expon_step' : 1,
        'min_samples' : 128,
      },

    }, configs)


  def suggest_inputs(self, num_suggest, inputs_old, outputs_old, sess,
    inputs_pend=None, num_options=None, num_to_optimize=32, subroutine=None,
    options=None, use_sobol=False, allow_duplicates=True, time_limit=np.inf,
    dtype=None, **kwargs):
    if (num_options is None): num_options = self.num_options
    if (dtype is None): dtype = self.dtype

    # Prepare a collection of initial input sets
    start_time = self.timer()
    if subroutine not in [self._optimize_inputs_random, self._optimize_inputs_halving]:
      inputs_new = self.sample_starts(num_to_optimize, num_suggest, sess,
                                      inputs_old, outputs_old, inputs_pend,
                                      time_limit=time_limit/10)
    else:
      input_dim = inputs_old.shape[-1]
      set_shape = [num_options, num_suggest, input_dim]
      inputs_new = self.sample_inputs(set_shape, options=options,
                                use_sobol=use_sobol, dtype=dtype)

    # Run an optimization subroutine
    nodes, loss_seq = self.optimize_inputs\
    (
      sess=sess,
      inputs_new=inputs_new,
      inputs_old=inputs_old,
      outputs_old=outputs_old,
      inputs_pend=inputs_pend,
      time_limit=time_limit - (self.timer() - start_time),
      subroutine=subroutine,
      **kwargs,
    )

    # Re-evaluate losses for optimized new inputs
    inputs_new = sess.run(nodes['inputs_new'])
    losses_op, feed_dict, nodes = self.appraise_inputs\
    (
      sess=sess,
      inputs_new=inputs_new,
      inputs_old=inputs_old,
      outputs_old=outputs_old,
      inputs_pend=inputs_pend,
      nodes=nodes,
      **{**kwargs, 'inputs_as_var':False},
    )[:3]
    losses = np.atleast_1d(np.squeeze(sess.run(losses_op, feed_dict)))

    # Suggest loss minimizer as next choice
    argmin = np.argmin(losses)
    suggestions = inputs_new[argmin]
    if not allow_duplicates: #[!] temp. hack: don't allow duplicate inputs
      exclude = np.vstack(filter(None.__ne__, (inputs_old, inputs_pend)))
      was_duplicate = np.full([num_suggest], False)
      for k, suggestion in enumerate(suggestions):
        while np.equal(exclude, suggestion).all(axis=1).any():
          #[!] Can inf. loop if options are exhausted...
          suggestion = self.sample_inputs([input_dim], options=options)
          was_duplicate[k] = True
        suggestions[k] = suggestion
        exclude = np.vstack([exclude, suggestion[None]])
      num_duplicates = np.sum(was_duplicate)
      if num_duplicates > 0:
        logger.info('Replaced {:d} duplicate input(s)'.format(num_duplicates))

    return suggestions, losses[argmin]


  def appraise_inputs(self, sess, inputs_new, inputs_old, outputs_old,
    inputs_pend=None, feed_dict=None, model=None, loss_fn=None, nodes=None,
    loss_kwargs=None, predict_kwargs=None, inputs_as_var=False, **kwargs):
    '''
    Assess the quality of different sets of new inputs given observed i/o
    pairs (inputs_old, outputs_old) along with any pending inputs.
    '''
    if (model is None): model = self.model
    if (loss_fn is None): loss_fn = self.loss_fn
    if (feed_dict is None): feed_dict = dict()
    if (predict_kwargs is None): predict_kwargs = dict()
    loss_kwargs = dict() if (loss_kwargs is None) else loss_kwargs.copy()

    # Prepare TensorFlow counterparts and feed_dict
    nodes, feed_dict = self.prepare(sess, inputs_new, inputs_old, outputs_old,
                                inputs_pend, feed_dict, nodes, inputs_as_var)

    # Precompute Cholesky factor of K_nz(inputs_old, inputs_old)
    cholesky_ref = self.get_or_create_ref('cholesky', inputs_old.ndim)
    if feed_dict.get(cholesky_ref, None) is None: #check existance
      cholesky_op = model.get_or_create_node\
      (
        group='predict',
        fn=model.compute_cholesky,
        args=(nodes['inputs_old'],),
        kwargs={'noisy':True},
      )
      feed_dict[cholesky_ref] = sess.run(cholesky_op, feed_dict)

    # Get placeholders/sources values needed by loss function
    num_pending = 0 if (inputs_pend is None) else inputs_pend.shape[-2]
    parallelism = num_pending + inputs_new.shape[1]
    prep_terms = loss_fn.prepare(sess, inputs_old, outputs_old,
                                  parallelism, self.model,
                                  **loss_kwargs)

    # User-specified terms take precedence
    loss_kwargs = {**prep_terms[0], **loss_kwargs}
    feed_dict = {**prep_terms[1], **feed_dict}

    # Call TensorFlow subroutine for pool-wise loss computation
    losses_op = loss_fn\
    (
      pools=nodes['pools'],
      inputs_old=nodes['inputs_old'],
      outputs_old=nodes['outputs_old'],
      parallelism=parallelism,
      model=model,
      predict_kwargs={'chol':cholesky_ref, **predict_kwargs},
      **loss_kwargs,
    )

    # Allow losses to return additional terms
    if isinstance(losses_op, tuple): losses_op, *extras = losses_op
    else: extras = None

    # [!] Hack to deal with padding
    if losses_op.shape.ndims == 4:
      losses_op = self.get_or_create_node('reduce', tf.reduce_mean, (losses_op, 0))

    return losses_op, feed_dict, nodes, extras


  def update(self, sess, inputs_old, outputs_old, model=None, var_list=None,
    loss_fn=None, reset=True, fit_kwargs=None, loss_kwargs=None, verbose=2,
    **kwargs):
    '''
    Update modules such as a surrogate model and loss function.
    '''
    if (model is None): model = self.model
    if (loss_fn is None): loss_fn = self.loss_fn
    if (fit_kwargs is None): fit_kwargs = dict()
    if (loss_kwargs is None): loss_kwargs = dict()
    if (var_list is None): var_list = tuple(model.state.values())
    else: var_list = tuple(var_list)

    tic_fit = self.timer()
    if reset: #reset model params first
      reset_op = model.get_or_create_node\
      (
        group='initial',
        fn=tf.variables_initializer,
        args=(var_list,),
      )
      _ = sess.run(reset_op)

    # Fit model to provided data
    _ = model.fit(sess, inputs_old, outputs_old, **fit_kwargs)
    var_values = sess.run(var_list)
    if (verbose > 0):
      logger.info('Fit model in {:.2e}s'.format(self.timer() - tic_fit))

    if (verbose > 1):
      print_vals = [np.around(v, 3).tolist() for v in var_values]
      logger.info('Updated model vars: {}'.format(print_vals))

    # Update loss function
    tic_loss = self.timer()
    _ = loss_fn.update(sess, inputs_old, outputs_old,
                    **{'model': model, **loss_kwargs})
    if (verbose > 0):
      logger.info('Updated loss function in {:.2e}s'.format(self.timer() - tic_loss))

    return var_values


  def optimize_inputs(self, sess, inputs_new, inputs_old, outputs_old,
    inputs_pend=None, var_list=None, loss_fn=None, batch_generator=None,
    config=None, subroutine=None, loss_kwargs=None, time_limit=np.inf, **kwargs):
    '''
    Wrapper for input optimization subroutines.
    '''
    start_time = self.timer()
    if (loss_fn is None): loss_fn = self.loss_fn
    if (loss_kwargs is None): loss_kwargs = dict()

    # Determine degree of parallelism
    num_pending = 0 if (inputs_pend is None) else inputs_pend.shape[-2]
    parallelism = num_pending + inputs_new.shape[1]

    # Build/retrieve pools variable and associated loss op
    losses_op, feed_dict, nodes, extras = self.appraise_inputs\
    (
      sess, inputs_new, inputs_old, outputs_old,
      inputs_pend=inputs_pend,
      inputs_as_var=True,
      loss_fn=loss_fn,
      loss_kwargs=loss_kwargs,
      **kwargs,
    )

    # Create mini-batch generator
    if (batch_generator is None):
      def batch_generator(batch_size, parallelism=parallelism):
        batch_kwargs = {**loss_kwargs, 'num_fantasies':batch_size}
        _, feed_dict = loss_fn.prepare(sess, inputs_old, outputs_old,
                            parallelism, self.model, **batch_kwargs)
        return feed_dict #new source values for placeholders

    # Branch according to parallelism
    if (subroutine is None):
      if parallelism == 1:
        subroutine = self._optimize_inputs_scipy
      else:
        subroutine = self._optimize_inputs_sgd

    # Run optimization subroutine
    loss_seq = subroutine\
    (
      sess=sess,
      inputs_new=nodes['inputs_new'],
      losses_op=losses_op,
      feed_dict=feed_dict,
      var_list=var_list,
      batch_generator=batch_generator,
      time_limit=time_limit - (self.timer() - start_time),
      parallelism=parallelism,
      nodes=nodes,
      extras=extras,
      config=config,
      loss_fn=loss_fn,
    )

    return nodes, loss_seq


  def _optimize_inputs_random(self, sess, inputs_new, losses_op, feed_dict=None,
    config=None, time_limit=np.inf, options=None, **kwargs):
    '''
    Optimize a loss function w.r.t. a set of inputs using Random Search.
    Batched evaluations are used in order to allow for a runtime limit.
    '''
    feed_dict = dict() if (feed_dict is None) else feed_dict.copy()
    config = self.update_dict(self.configs['random'], config, as_copy=True)

    # Settings for Random Search
    eval_limit = config['eval_limit']
    shard_shape = inputs_new.get_shape().as_list()
    shard_size = shard_shape[0]
    start_time = self.timer()

    # Iterate Random Search until evaluation/runtime budget is exhausted
    inputs_seq, loss_seq, counter = [], [], 0
    while counter + shard_size <= eval_limit:
      if (self.timer() - start_time > time_limit): break
      inputs_seq.append(self.rng.rand(*shard_shape))
      feed_per_step = {**feed_dict, inputs_new:inputs_seq[-1]}
      losses_per_step = sess.run(losses_op, feed_per_step)
      loss_seq.append(np.ravel(losses_per_step))
      counter += shard_size

    # Choose top-k loss minimizers
    loss_seq = np.hstack(loss_seq)
    argmins = np.argpartition(loss_seq, shard_size-1)[:shard_size]

    # Assign optimized values to <tf.Variable>
    ref = self.get_or_create_ref('assignment', dtype=inputs_new.dtype)
    assign_op = self.get_or_create_node('assign', tf.assign, (inputs_new, ref))
    sess.run(assign_op, {ref:np.vstack(inputs_seq)[argmins]})

    logger.info('Random Search evaluated {:d} losses in {:.3e}s'\
          .format(len(loss_seq), self.timer() - start_time))

    return loss_seq


  def _optimize_inputs_sgd(self, sess, inputs_new, losses_op, optimizer=None,
    feed_dict=None, var_list=None, batch_generator=None, config=None,
    time_limit=np.inf, global_step=None, scope='sgd', reuse=None,
    **kwargs):
    '''
    Optimize a loss function w.r.t. a set of inputs using
    stochastic gradient descent methods such as ADAM
    '''
    feed_dict = dict() if (feed_dict is None) else feed_dict.copy()

    # Cast var_list to <tuple> and add in new inputs variable
    if (var_list is None): var_list = tuple()
    else: var_list = tuple(var_list)
    if not isinstance(inputs_new, tf.Variable):
      raise TypeError('Candidates must be a <tf.Variable>')
    var_list = var_list + (inputs_new,)

    with tf.variable_scope(scope, reuse=reuse) as vs:
      if (global_step is None):
        var_name = '{}/global_step'.format(vs.name)
        global_step = self.get_or_create_var(var_name, shape=[], dtype='int64')

      # Get or create optimizer module
      config = self.update_dict(self.configs['sgd'], config, as_copy=True)
      if (optimizer is None): #[!] improve me
        if self.modules.get(config['method'], None) is None:
          reducer = lambda pkg, attr: getattr(pkg, attr)
          constructor = reduce(reducer, [tf]+config['method'].split('.'))
          if constructor in [tf.train.GradientDescentOptimizer]:
            decay_rate = tf.pow(tf.cast(global_step + 1, 'float64'), -0.7)
            learning_rate = decay_rate * config.get('learning_rate', 1.0)
            self.modules[config['method']] = constructor(learning_rate)
          elif constructor in [tf.train.MomentumOptimizer]:
            decay_rate = tf.pow(tf.cast(global_step + 1, 'float64'), -0.7)
            learning_rate = decay_rate * config.get('learning_rate', 1.0)
            self.modules[config['method']] = constructor(learning_rate, 0.9)
          else:
            learning_rate = config.get('learning_rate', 25e-3)
            self.modules[config['method']] = constructor(learning_rate)

        optimizer = self.modules[config['method']]

      # Build/retrieve update operations
      _update_op = self.get_or_create_node\
      (# Compute/apply raw updates
        group=scope,
        fn=optimizer.minimize,
        args=(losses_op,), 
        kwargs={'var_list':tuple(var_list), 'global_step':global_step},
      )

      clip_op = self.get_or_create_node\
      (# Post-update, clip pools to [0, 1]^{d}
        group=scope,
        fn=tf.clip_by_value,
        args=(inputs_new, 0, 1),
        control_dependencies=(_update_op,),
      )

      update_op = self.get_or_create_node\
      (# Assign clipped values to new inputs variable
        group=scope,
        fn=tf.assign,
        args=(inputs_new, clip_op),
      )

      # Initialize/reset local variables
      local_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, vs.name)
      reset_op = self.get_or_create_node\
      (
        group=vs.name,
        fn=tf.variables_initializer,
        args=(tuple(local_vars),)
      )
      _ = sess.run(reset_op)

      # Execution settings
      step_limit = config.get('step_limit', 1000)
      batch_size = config.get('batch_size', 128)

      # Perform stochastic gradient descent
      start_time = self.timer()
      fetches = [inputs_new, losses_op, update_op]
      inputs_seq, loss_seq = [], []
      # inputs_seq.append(sess.run(inputs_new))
      for step in range(step_limit):
        if self.timer() - start_time >= time_limit: break
        if callable(batch_generator):
          minibatch = batch_generator(batch_size)
          feed_dict.update(minibatch)
        inputs_per_step, losses_per_step, _ = sess.run(fetches, feed_dict)
        inputs_seq.append(inputs_per_step)
        loss_seq.append(losses_per_step)

      # Assign Polyak average to <tf.Variable>
      if config.get('use_averaging', False):
        ref = self.get_or_create_ref('assignment', dtype=inputs_new.dtype)
        op = self.get_or_create_node('assign', tf.assign, (inputs_new, ref))
        _ = sess.run(op, {ref : np.mean(inputs_seq, 0)})

      loss_seq = np.ravel(loss_seq)
      logger.info('SGD evaluated {:d} losses in {:.3e}s'\
            .format(len(loss_seq), self.timer() - start_time))

      # self.inputs_seq = inputs_seq #[!] temp. hack: for plotting
      return loss_seq


  def _optimize_inputs_scipy(self, sess, inputs_new, losses_op, feed_dict=None,
    var_list=None, config=None, time_limit=np.inf, **kwargs):
    '''
    Optimize a loss function w.r.t. a set of inputs using
    Scipy Optimize method such as L-BFGS-B.
    '''
    feed_dict = dict() if (feed_dict is None) else feed_dict.copy()

    # Cast var_list to <tuple> and append new inputs variable
    if (var_list is None): var_list = tuple()
    else: var_list = tuple(var_list)
    if not isinstance(inputs_new, tf.Variable):
      raise TypeError('Candidates must be a <tf.Variable>')
    var_list = tuple(set(var_list + (inputs_new,))) #singularize

    # ScipyOptimize configuration
    config = self.update_dict(self.configs['scipy'], config, as_copy=True)

    # Create callback functions
    inputs_seq, loss_seq, start_time = [], [], self.timer()
    def loss_callback(inputs_src, losses_per_step):
      loss_seq.append(losses_per_step)
      inputs_seq.append(inputs_src.copy())
      if self.timer() - start_time >= time_limit:
        raise StopIteration('Runtime limit exhausted.')

    # Get/build optimizer
    loss_op = self.get_or_create_node('scipy', tf.reduce_mean, (losses_op,))
    optim_id = (loss_op,) + var_list  #[!] temp. hack: generalize this!
    if not 'scipy' in self.modules: self.modules['scipy'] = dict()
    if not optim_id in self.modules['scipy']:
      method = config.pop('method', 'L-BFGS-B')
      var_to_bounds = config.pop('var_to_bounds', dict())
      var_to_bounds[inputs_new] = (0, 1)
      self.modules['scipy'][optim_id] = ScipyOptimizerInterface\
      (
        loss_op,
        var_list, method=method,
        var_to_bounds=var_to_bounds,
        options=config
      )
    optimizer = self.modules['scipy'][optim_id]

    # Run ScipyOptimize instance
    try:
      optimizer.minimize\
      (
        session=sess,
        feed_dict=feed_dict,
        fetches=[inputs_new, losses_op],
        loss_callback=loss_callback
      )
    except StopIteration as e:
      pass #allows us to stop early

    # Get the best input sets found during optimization
    loss_seq = np.ravel(loss_seq)
    var_shape = inputs_new.get_shape().as_list()
    seq_shape = [-1] + var_shape[1:]
    inputs_seq = np.reshape(inputs_seq, seq_shape)
    argmins = np.argpartition(loss_seq, var_shape[0] - 1)[:var_shape[0]]

    # Assign to <tf.Variable>
    ref = self.get_or_create_ref('assignment', dtype=inputs_new.dtype)
    assign_op = self.get_or_create_node('assign', tf.assign, (inputs_new, ref))
    _ = sess.run(assign_op, {ref:inputs_seq[argmins]})

    loss_seq = np.ravel(loss_seq)
    logger.info('ScipyOptimize evaluated {:d} losses in {:.3e}s'\
            .format(len(loss_seq), self.timer() - start_time))
    return loss_seq


  def _optimize_inputs_nlopt(self, sess, inputs_new, losses_op, feed_dict=None,
    var_list=None, config=None, time_limit=np.inf, **kwargs):
    '''
    Optimize a loss function w.r.t. a set of inputs using
    NLopt methods such as Dividing Rectangles (DIRECT).
    '''
    feed_dict = dict() if (feed_dict is None) else feed_dict.copy()

    # Prepare requisite graph elements
    inputs_new_shape = inputs_new.get_shape().as_list()

    # Create TensorFlow wrapper
    inputs_seq, loss_seq, start_time = [], [], self.timer()
    def wrapper(inputs_new_vect, user_data):
      if self.timer() - start_time > time_limit:
        raise StopIteration('Runtime limit exhausted.')
      feed_dict[inputs_new] = np.reshape(inputs_new_vect, inputs_new_shape)
      losses_per_step = sess.run(losses_op, feed_dict)
      inputs_seq.append(feed_dict[inputs_new].copy())
      loss_seq.append(losses_per_step)
      return np.mean(losses_per_step)

    # Initialize and configure NLopt optimizer
    import nlopt # pylint: disable=import-not-at-top
    config = self.update_dict(self.configs['nlopt'], config, as_copy=True)
    num_params = int(np.prod(inputs_new_shape))
    optimizer = nlopt.opt(getattr(nlopt, config['method']), num_params)
    optimizer.set_min_objective(wrapper)
    optimizer.set_maxeval(config.get('maxfun', 1024))
    optimizer.set_lower_bounds(config.get('lower', 0.0))
    optimizer.set_upper_bounds(config.get('upper', 1.0))

    use_initial = config.pop('use_initial', True)
    if use_initial:
      starts = np.ravel(sess.run(inputs_new))
    else:
      self.rng.rand(num_params)

    # Run optimizer until convergence or time limit is exhausted
    try:
      _ = optimizer.optimize(starts)
    except StopIteration as e:
      pass #allows us to stop early

    inputs_seq, loss_seq = np.stack(inputs_seq), np.ravel(loss_seq)
    argmins = np.where(loss_seq == np.min(loss_seq))[0]
    inputs_new_src = inputs_seq[self.rng.choice(argmins)]

    # Assign optimized values to <tf.Variable>
    ref = self.get_or_create_ref('assignment', dtype=inputs_new.dtype)
    assign_op = self.get_or_create_node('assign', tf.assign, (inputs_new, ref))
    sess.run(assign_op, {ref:inputs_new_src})

    logger.info('NLopt evaluated {:d} losses in {:.3e}s'\
          .format(len(loss_seq), self.timer() - start_time))
    return loss_seq


  def _optimize_inputs_cmaes(self, sess, inputs_new, losses_op,
    feed_dict=None, var_list=None, batch_generator=None, config=None,
    time_limit=np.inf, **kwargs):
    '''
    Optimize a loss function w.r.t. a set of inputs using
    Covariance Matrix Adaptation Evolution Strategy (CMA-ES).
    '''
    feed_dict = dict() if (feed_dict is None) else feed_dict.copy()

    # Prepare requisite graph elements
    loss_op = self.get_or_create_node('cmaes', tf.reduce_mean, (losses_op,))
    inputs_new_shape = inputs_new.get_shape().as_list()
    popsize, num_params = inputs_new_shape[0], int(np.prod(inputs_new_shape[1:]))

    # Initialize and configure CMA-ES optimizer
    import cma # pylint: disable=import-not-at-top
    config = self.update_dict(self.configs['cmaes'], config, as_copy=True)
    config.setdefault('seed', self.rng.randint(2**31 - 1))
    constructor = getattr(cma, config.pop('method', 'CMAEvolutionStrategy'))

    # Extract options
    sigma0 = config.pop('sigma0', 0.5)
    step_limit = config.pop('step_limit', 1000)
    batch_size = config.pop('batch_size', 128)

    stop_early = config.pop('stop_early', True)
    use_initial = config.pop('use_initial', True)
    optimizer = constructor(num_params*[0.5], sigma0, config)

    # Run CMA-ES w/ parallel function evaluations
    def run_cmaes(step_limit, popsize, min_cov=np.finfo('float').eps):
      start_time, counter, loss_seq = self.timer(), 0, []
      while counter < step_limit:
        # Check additional stopping condition(s)
        if self.timer() - start_time >= time_limit: break
        if stop_early and optimizer.stop(): break

        if callable(batch_generator) and batch_size > 0:
          minibatch = batch_generator(batch_size)
          feed_dict.update(minibatch)

        # Request next generation
        try:
          inputs_new_src = optimizer.ask(popsize)
        except:
          break #CMA-ES can crash for various reasons...

        # Use externally chosen initial points as first generation?
        if use_initial and counter == 0:
          inputs_new_src = np.reshape(sess.run(inputs_new), [popsize, -1])

        # Evaluate generation
        feed_dict[inputs_new] = np.reshape(inputs_new_src, inputs_new_shape)
        losses_per_step = np.ravel(sess.run(losses_op, feed_dict))
        optimizer.tell(inputs_new_src, losses_per_step)
        loss_seq.append(losses_per_step)
        counter += 1

      # CMA-ES only returns a single "best"; we augment it to conform w/ API.
      best = optimizer.result[0]
      if (best is None): #optimizer wasn't run (e.g. time_limit < 0)
        inputs_new_src = self.rng.rand(*inputs_new_shape)
      else:
        best = np.reshape(best, [1]+inputs_new_shape[1:])
        rvs = self.rng.rand(popsize - 1, *inputs_new_shape[1:])
        inputs_new_src = np.vstack([best, rvs])

      loss_seq = np.ravel(loss_seq)
      logger.info('CMA-ES evaluated {:d} losses in {:.3e}s'\
            .format(len(loss_seq), self.timer() - start_time))

      return inputs_new_src, loss_seq
    inputs_new_src, loss_seq = run_cmaes(step_limit, popsize)

    # Assign optimized values to <tf.Variable>
    ref = self.get_or_create_ref('assignment', dtype=inputs_new.dtype)
    assign_op = self.get_or_create_node('assign', tf.assign, (inputs_new, ref))
    sess.run(assign_op, {ref:inputs_new_src})
    return loss_seq


  def _optimize_inputs_halving(self, sess, inputs_new, losses_op,
    feed_dict=None, config=None, time_limit=np.inf, options=None,
    batch_generator=None, parallelism=None, nodes=None, extras=None,
    loss_fn=None, cost_history=None, **kwargs):
    '''
    Optimize a loss function w.r.t. a set of inputs using Successive Halving.
    Batched evaluations are used in order to allow for a runtime limit.
    '''
    if (loss_fn is None): loss_fn = self.loss_fn
    if (cost_history is None): #historical cost per iteration
      if hasattr(self, 'halving_costs'): cost_history = self.halving_costs
      else: cost_history = self.halving_costs = dict()

    # Local aliases to keep things legible
    get_node = self.get_or_create_node
    get_var = self.get_or_create_var
    get_ref = self.get_or_create_ref

    feed_dict = dict() if (feed_dict is None) else feed_dict.copy()
    config = self.update_dict(self.configs['halving'], config, as_copy=True)
    shard_shape = inputs_new.get_shape().as_list()
    shard_size = shard_shape[0]

    # Placeholders for cached posteriors (non-RF only)
    posteriors_refs =\
    (
      get_ref('mean/new', 3),
      get_ref('cov/new', 3),
      get_ref('cholesky/new', 3)
    )

    # Determine degree of parallelism
    num_pending = len(feed_dict[nodes['inputs_pend']])
    num_suggest = shard_shape[-2]
    if (parallelism is None):
      parallelism = num_pending + num_suggest

    # Precompute RF path minima at pending input locations
    if (parallelism > 1 and loss_fn.use_rand_features):
      inputs_pend_ref = nodes.pop('inputs_pend')
      inputs_pend = feed_dict.pop(inputs_pend_ref, [])
      if len(inputs_pend):
        incumbents_ref = get_ref('halving/incumbents', [None])
        paths_op = get_node\
        (
          group='halving/rf',
          fn=loss_fn.mapping,
          args=(inputs_pend_ref,),
        )
        fantasies = sess.run(paths_op, {inputs_pend_ref : inputs_pend})
        incumbents_src = np.min(fantasies, axis=0)

    # Determine sample counts (batch_sizes) to iterate over
    max_samples = config.get('max_samples', self.loss_fn.num_fantasies)
    if (parallelism is None or parallelism > 1):
      min_samples = config.get('min_samples', 128)
      lower = int(np.log2(min_samples))
      upper = int(np.log2(max_samples))
      expon_step = config.get('expon_step', 1) #power of 2 to iterate by
      if expon_step < 0: expon_step = upper - lower
      log_sizes = list(range(lower, upper+1, expon_step))
      log_sizes[-1] = upper #[!] temp. hack
      batch_sizes = [int(2 ** log_size) for log_size in log_sizes]
    else:
      # For parallelism==1, we use closed-form solutions
      batch_sizes = [max_samples]

    def allocate_time(time_limit, batch_sizes, cost_history):
      '''
      Allocate time to each iteration of successive halving.
      If costs are known, allocate s.t. the number of losses
      evaluated per step is inv. proportional to the Monte
      Carlo batch-size.
      '''
      try:
        allocs = []
        for batch_size in batch_sizes:
          cost_id = (num_pending, num_suggest, batch_size)
          cost = cost_history[cost_id]
          allocs.append(cost/batch_size)
      except KeyError as e:
        allocs = np.ones(len(batch_sizes))

      # Return renormalized time allocations
      return np.multiply(time_limit/np.sum(allocs), allocs)
    time_allocs = allocate_time(time_limit, batch_sizes, cost_history)

    def run_random_search(time_limit, eval_limit, options=None,
      posteriors=None, losses_op=losses_op, feed_dict=feed_dict,
      extras=extras):
      '''
      Iterate Random Search until evaluation/runtime budget is exhausted
      '''
      if (extras is None): extras = list()
      inputs_seq, loss_seq, _extras = [], [], []
      counter, run_start = 0, self.timer()

      shard_costs = []
      while counter + shard_size <= eval_limit:
        tic = self.timer()
        if (self.timer() - run_start > time_limit): break
        if (options is None):
          inputs_seq.append(self.rng.rand(*shard_shape))
        else:
          inputs_seq.append(options[counter : counter + shard_size])
          for (ref, src) in zip(posteriors_refs, posteriors):
            feed_dict[ref] = src[counter : counter + shard_size]

        feed_per_step = {**feed_dict, inputs_new:inputs_seq[-1]}
        losses_per_step, *extras_per_step =\
          sess.run([losses_op] + extras, feed_per_step)

        loss_seq.append(np.ravel(losses_per_step))
        if len(extras_per_step):
          _extras.append(extras_per_step[0]) #[!] temp hack
        counter += shard_size
        shard_costs.append(self.timer() - tic)

      if len(_extras):
        _extras = list(map(np.vstack, zip(*_extras)))
      return np.vstack(inputs_seq), np.hstack(loss_seq), shard_costs, _extras

    # Minimize loss function via successive halving
    losses = None
    options = None
    counters = []
    num_iters = len(batch_sizes)
    bsize_prev = 0
    posteriors = None
    start_time = self.timer()
    for k, (batch_size, time_alloc) in enumerate(zip(batch_sizes, time_allocs)):
      # Update r.v.'s used for Monte Carlo integral
      feed_dict.update(batch_generator(batch_size - bsize_prev))

      # Get/build loss function
      # if (parallelism > 1 and loss_fn.use_rand_features):
      if loss_fn.use_rand_features:
        # Extract active subset of theta (i.e. subset of sample paths)
        starts = (bsize_prev, 0)
        sizes = (batch_size - bsize_prev, -1)
        theta = loss_fn.mapping.theta

        theta_active = get_node('halving/theta', tf.slice, (theta, starts, sizes))
        loss_kwargs =\
        {
          'use_rand_features' : parallelism > 1,
          'rf_kwargs' : {'theta' : theta_active}
        }

        # Pass on incumbent sample mins
        if (num_pending > 0):
          loss_kwargs['incumbents'] = incumbents_ref
          feed_dict[incumbents_ref] = incumbents_src[bsize_prev : batch_size]

        losses_op, feed_dict, nodes, extras = self.appraise_inputs\
        (
          sess=sess,
          inputs_new=self.rng.rand(*inputs_new.get_shape()),
          inputs_old=feed_dict[nodes['inputs_old']],
          outputs_old=feed_dict[nodes['outputs_old']],
          feed_dict=feed_dict.copy(),
          nodes=nodes.copy(),
          inputs_as_var=True,
          loss_kwargs=loss_kwargs,
        )

      elif (k == 1):
        # Build loss reusing cached posteriors
        losses_op, feed_dict, nodes, extras = self.appraise_inputs\
        (
          sess=sess,
          inputs_new=options[:1],
          inputs_old=feed_dict[nodes['inputs_old']],
          outputs_old=feed_dict[nodes['outputs_old']],
          inputs_pend=feed_dict[nodes['inputs_pend']],
          feed_dict=feed_dict.copy(),
          nodes=nodes.copy(),
          inputs_as_var=True,
          loss_kwargs={'posteriors':posteriors_refs},
        )

      # Evaluate candidate options
      eval_limit = config['eval_limit'] if options is None else len(options)
      options, losses_new, costs, posteriors = run_random_search\
      (
        time_limit=time_alloc,
        eval_limit=eval_limit,
        options=options,
        posteriors=posteriors,
        losses_op=losses_op,
        feed_dict=feed_dict,
        extras=extras,
      )
      cost_id = (num_pending, num_suggest, batch_size)
      cost_history[cost_id] = np.mean(costs)
      #print(batch_size, time_alloc, '\t', len(options), '\t', np.mean(costs))

      # Combine old and new sample estimates
      if (losses is None):
        losses = losses_new
      else:
        num_losses = len(losses_new)
        losses_old = losses[:num_losses]
        weight_old = bsize_prev
        weight_new = batch_size - bsize_prev
        losses = np.divide\
        (
          weight_old * losses_old + weight_new*losses_new,
          weight_old + weight_new
        )

      # If limited by num. evals, prune and sort; otherwise, sort.
      bsize_prev = batch_size
      num_options = len(options)
      counters.append(num_options)
      if (k + 1 < num_iters):
        if (time_limit == np.inf) and (num_options > shard_size):
          denom = 2 ** int(expon_step)
          top_k = max(shard_size, num_options//denom)
          leaders = np.argpartition(losses, top_k - 1)[:top_k]
          indices = leaders[np.argsort(losses[leaders])]
        else:
          indices = np.argsort(losses)
        options = options[indices]
        losses = losses[indices]
        posteriors = [term[indices] for term in posteriors]

    # Choose top-k loss minimizers
    argmins = np.argpartition(losses, shard_size - 1)[:shard_size]

    # Assign optimized values to <tf.Variable>
    ref = get_ref('assignment', dtype=inputs_new.dtype)
    assign_op = get_node('assign', tf.assign, (inputs_new, ref))
    sess.run(assign_op, {ref:options[argmins]})

    logger.info('Successive Halving evaluated {:d} options in {:.3e}s'\
          .format(counters[0], self.timer() - start_time))

    return losses


  def suggest_answers(self, num_suggest, inputs_old, outputs_old,
    sess, risk=None, num_starts=2**10, num_options=2**16,
    config=None, time_limit=np.inf, in_order=False, model=None):
    '''
    Recommend a set of inputs as final answers. By default,
    suggests the top-k minimizers of posterior mean.
    '''
    if (risk is None): risk = self.risk

    start_time = self.timer()
    if (model is None): model = self.model
    config = self.update_dict(self.configs['scipy'], config, as_copy=True)

    # Local aliases to keep things legible
    get_node = self.get_or_create_node
    get_var = self.get_or_create_var
    get_ref = self.get_or_create_ref

    # Build observed i/o nodes
    rank_o = inputs_old.ndim
    input_dim = inputs_old.shape[-1]
    inputs_ref = get_ref('inputs/old', rank_o)
    outputs_ref = get_ref('outputs/old', rank_o)
    feed_dict = {inputs_ref:inputs_old, outputs_ref:outputs_old}

    # Precompute Cholesky factor of K_nz(inputs_old, inputs_old)
    cholesky_ref = get_ref('cholesky/old', inputs_old.ndim)
    if feed_dict.get(cholesky_ref, None) is None: #check existance
      cholesky_op = get_node\
      (
        group='predict',
        fn=model.compute_cholesky,
        args=(inputs_ref,),
        kwargs={'noisy':True},
      )
      feed_dict[cholesky_ref] = sess.run(cholesky_op, feed_dict)

    def build_loss(inputs, risk=risk, feed_dict=feed_dict):
      '''
      Get or create a loss functions as: $\mu + sqrt(\lambda*\sigma^{2})$
      '''
      mean_op, var_op = model.get_or_create_node\
      (
        group='predict',
        fn=model.predict,
        args=(inputs, inputs_ref, outputs_ref),
        kwargs={'chol':cholesky_ref},
        stateful=True,
      )
      if (risk != 0):
        risk_ref = get_ref('answers/risk', 0)
        stddevs = get_node('answers', tf.sqrt, (var_op,))
        d_means = get_node('answers', tf.multiply, (stddevs, risk_ref))
        loss_op = get_node('answers', tf.subtract, (mean_op, d_means))
        feed_dict[risk_ref] = risk
      else:
        loss_op = mean_op
      return loss_op, feed_dict, (mean_op, var_op)

    # Sample starting positions
    def sample_starts(num_options, use_sobol=True,**kwargs):
      num_old = len(inputs_old)
      num_samples = num_options - num_old
      src = np.vstack\
      ([
        inputs_old,
        self.sample_inputs([num_samples, input_dim], use_sobol=use_sobol)
      ])
      ref = get_ref('inputs/new', src.ndim)
      loss_op, feed_dict, _ = build_loss(ref)
      losses = np.squeeze(sess.run(loss_op, {**feed_dict, ref:src}), -1)
      return self.sample_propto_loss(src, losses, num_starts, **kwargs)[0]

    # Deterministically choose the top_k and sample the rest
    starts_src = sample_starts(num_options, top_k=min(32, num_options))
    starts_ref = self.get_or_create_ref('assign')

    # Initialize new inputs variable and assign starting values
    answers_var = get_var('inputs/new', starts_src.shape)
    init_op = get_node\
    (
      group='initialize',
      fn=tf.variables_initializer,
      args=((answers_var,),),
    )

    assign_op = get_node\
    (
      group='assign',
      fn=tf.assign,
      args=(answers_var, starts_ref),
      control_dependencies=(init_op,), #initialize first
    )
    _ = sess.run(assign_op, {starts_ref : starts_src})

    loss_op, feed_dict, (mean_op, var_op) = build_loss(answers_var)
    time_remaining = time_limit - (self.timer() - start_time)
    _ = self._optimize_inputs_scipy(sess, answers_var, loss_op, feed_dict,
                                  config=config, time_limit=time_remaining)

    # Return top-k minimizers of the (penalized) posterior mean
    fetches = (answers_var, mean_op, var_op)
    answers_src, means, sigma2s = sess.run(fetches, feed_dict)
    assert means.shape[-1] == 1, "Multiple objectives not yet supported"

    means = np.squeeze(means, axis=-1)
    indices = np.argpartition(means, num_suggest - 1)[:num_suggest]
    if in_order: #return suggestions in sorted order (descending)
      indices = np.argsort(outputs_old[indices])

    return answers_src[indices], (means[indices], sigma2s[indices])


  def prepare(self, sess, inputs_new, inputs_old, outputs_old,
    inputs_pend=None, feed_dict=None, nodes=None, inputs_as_var=False):
    '''
    Helper method for preparing TensorFlow nodes corresponding
    with Numpy source values and associated feed_dict.
    '''
    if (nodes is None): nodes = dict()
    if (feed_dict is None): feed_dict = dict()

    # Local aliases to keep things legible
    get_node = self.get_or_create_node
    get_var = self.get_or_create_var
    get_ref = self.get_or_create_ref

    # Check basic shape info
    rank_o = inputs_old.ndim
    rank_c, shape_c = inputs_new.ndim, inputs_new.shape
    assert rank_c > 1, 'Candidates must be at least 2-dimensional'
    assert rank_o > 1, 'Observations must be at least 2-dimensional'
    assert rank_o == outputs_old.ndim, 'Observations have mismatched rank'
    assert outputs_old.shape[-1] == 1, 'Multiple objectives not yet supported'

    # Build node for observed inputs
    if ('inputs_old' not in nodes):
      nodes['inputs_old'] = get_ref('inputs/old', rank_o)
    feed_dict[nodes['inputs_old']] = inputs_old

    # Build node for observed outputs
    if ('outputs_old' not in nodes):
      nodes['outputs_old'] = get_ref('outputs/old', rank_o)
    feed_dict[nodes['outputs_old']] = outputs_old

    # Handle pending inputs
    if (inputs_pend is None): #avoid branching by with/without pending
      inputs_pend = np.empty((0, shape_c[-1]), dtype=inputs_new.dtype)

    if ('inputs_pend' not in nodes):
      nodes['inputs_pend'] = get_ref('inputs/pend', 2)
    feed_dict[nodes['inputs_pend']] = inputs_pend

    # Build node for new inputs
    node = nodes.get('inputs_new', None)
    if inputs_as_var and not isinstance(node, tf.Variable):
      # Create <tf.Variable> and assign initial values
      node = get_var('inputs/new', shape=inputs_new.shape)
      ref = get_ref('assignment', dtype=node.dtype)
      assign_op = get_node('assign', tf.assign, (node, ref))
      _ = sess.run(assign_op, {ref : inputs_new})
    elif not inputs_as_var:
      if not isinstance(node, tf.Tensor):
        # Create <tf.Placeholder> and add to feed_dict
        node = get_ref('inputs/new', rank_c)
      feed_dict[node] = inputs_new
    nodes['inputs_new'] = node

    # Expand and tile pending jobs
    shape_n = get_node('misc', tf.shape, (nodes['inputs_new'],))
    _inputs_pend = get_node\
    ('misc', tf.tile,
      (
        get_node('misc', utils.expand_to, (nodes['inputs_pend'], rank_c)),
        (get_node('misc', shape_n.__getitem__, (0,)), 1, 1),
      )
    )

    # Combine pending and new inputs to form pools
    nodes['pools'] = get_node\
    (
      group='inputs/new',
      fn=tf.concat,
      args=((_inputs_pend, nodes['inputs_new']),),
      kwargs={'axis':-2},
    )

    return nodes, feed_dict


  def sample_starts(self, num_starts, num_suggest, sess, inputs_old,
    outputs_old, inputs_pend=None, feed_dict=None, losses_op=None,
    eval_limit=2**20, time_limit=np.inf, shard_size=256, **kwargs):
    '''
    Sample starting positions proportional to marginal utility
    (negative losses) for multi-start query optimization subroutines.
    '''
    # Fantasize pending outputs at corresponding means
    if (inputs_pend is not None):
      inputs_old_ref = self.get_or_create_ref('inputs/old', 2)
      outputs_old_ref = self.get_or_create_ref('outputs/old', 2)
      inputs_pend_ref = self.get_or_create_ref('inputs/pend', 2)
      means_op, _ = self.get_or_create_node\
      (
        group='predict',
        fn=self.model.predict,
        args=(inputs_pend_ref, inputs_old_ref, outputs_old_ref,),
      )
      means_pend = sess.run(means_op, {
                            inputs_old_ref:inputs_old,
                            outputs_old_ref:outputs_old,
                            inputs_pend_ref:inputs_pend})
      inputs_old = np.vstack([inputs_old, inputs_pend])
      outputs_old = np.vstack([outputs_old, means_pend])

    input_dim = inputs_old.shape[-1]
    if (losses_op is None):
      losses_op, feed_dict, nodes, extras = self.appraise_inputs\
      (
        sess=sess,
        inputs_new=np.empty([0, 1, input_dim]),
        inputs_old=inputs_old,
        outputs_old=outputs_old,
        feed_dict=feed_dict,
        **kwargs,
      )

    options, losses = [], []
    start_time = self.timer()
    for k in range(eval_limit // shard_size):
      if (self.timer() - start_time > time_limit): break
      options.append(self.rng.rand(shard_size, input_dim))
      feed_k = {**feed_dict, nodes['inputs_new'] : np.expand_dims(options[-1], 1)}
      losses_k = sess.run(losses_op, feed_k)
      losses.append(np.atleast_1d(np.squeeze(losses_k)))

    losses = np.hstack(losses)
    options = np.vstack(options)
    num_options = len(options)

    logger.info('Evaluated {:d} marginal losses in {:.3e}s'\
          .format(num_options, self.timer() - start_time))

    weights = np.clip(np.max(losses) - losses,
                      np.finfo(losses.dtype).eps,
                      np.inf)
    weights /= np.sum(weights)
    indices, fillers = [], []
    while len(indices) + len(fillers) < num_starts:
      is_unique = True
      indices_new = self.rng.choice(num_options, num_suggest,
                                    p=weights, replace=False)

      for indices_old in indices:
        if np.array_equal(indices_new, indices_old):
          is_unique = False
          break

      if is_unique:
        indices.append(indices_new)
      else:
        random_start = self.rng.rand(num_suggest, input_dim)
        fillers.append(random_start)

    starts = options[np.stack(indices)]
    if len(fillers): starts = np.vstack([starts, np.stack(fillers)])
    return starts



# ==============================================
#                            Developers' Section
# ==============================================
# ------------ Scrap work goes here ------------
'''
'''