#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
Authored: 2018-04-11 
Modified: 2018-04-30
'''

# ---- Dependencies
import os, sys, time
import logging
import numpy as np
import numpy.random as npr
import tensorflow as tf

from pdb import set_trace as bp
from timeit import default_timer
from tensorflow.contrib.opt import ScipyOptimizerInterface

sys.path.insert(0, os.path.join(sys.path[0], '..'))
from src.misc import random_feature_mapping as rf_ops

logger = logging.getLogger(__name__)
# ==============================================
#                                     task_utils
# ==============================================
def minimize_task(config, sess, task, num_starts=None, num_options=None,
  scope='minimize_task', reuse=None, spo_config=None, dtype=None, rng=None):
  '''
  Estimate task minimum using multi-start L-BFGS-B.
  '''
  if (rng is None): rng = npr.RandomState(config.seed)
  if (dtype is None): dtype = task.dtype
  if (spo_config is None): spo_config = dict()
  if (num_starts is None): num_starts = config.num_starts_fmin
  if (num_options is None): num_options = config.num_options_fmin
  with tf.variable_scope(scope, reuse=reuse) as vs:
    # Build minimization target
    shape = [num_starts, task.input_dim]
    inputs_var = tf.get_variable('inputs', shape=shape, dtype=dtype,
                        initializer=tf.random_uniform_initializer())

    task_op = task.tensorflow(inputs_var, noisy=False, stop_gradient=False)
    loss_op = tf.reduce_mean(task_op)

    # Find starting positions via initial random sweep
    counter, x_mins, f_mins, = 0, None, None
    while counter + num_starts <= num_options:
      inputs = rng.rand(*shape)
      outputs = np.squeeze(sess.run(task_op, {inputs_var : inputs}))
      if (counter == 0):
        x_mins, f_mins = inputs, outputs
      else:
        inputs = np.vstack([x_mins, inputs])
        outputs = np.hstack([f_mins, outputs])
        argmins = np.argpartition(outputs, num_starts - 1)[:num_starts]
        x_mins, f_mins = inputs[argmins], outputs[argmins]
      counter += num_starts
    _ = sess.run(tf.assign(inputs_var, x_mins))

    # Initialize task optimizer
    spo_config =\
    {
      'method' : 'L-BFGS-B',
      'var_list' : [inputs_var],
      'var_to_bounds' : {inputs_var : (0, 1)},
      'options' : {'maxiter' : 1024},
      **spo_config, #user-specified settings take precedence
    }
    optimizer = ScipyOptimizerInterface(loss_op, **spo_config)

    # Run task optimizer
    _ = sess.run(tf.variables_initializer([inputs_var]))
    _ = optimizer.minimize(sess)

    # Evaluate task at optimized input locations
    inputs, outputs = sess.run([inputs_var, task_op])
    argmin = np.argmin(outputs)
    x_min = inputs[argmin]
    f_min = outputs[argmin, 0]
    return x_min, f_min


def estimate_task_mean(config, task, num_samples=2**16, shard_size=2**12,
  rng=None, noisy=False, timer=None, **kwargs):
  '''
  Estimate mean of task a given task.
  '''
  if (rng is None): rng = npr.RandomState(config.seed)
  logger.info('Forming Monte Carlo estimate to prior mean')
  timer = time.process_time if config.use_cpu_time else default_timer
  tic = timer()

  num_shards, mean = int(num_samples/shard_size), 0
  for k in range(num_shards):
    X = rng.rand(shard_size, task.input_dim)
    Y = task.numpy(X, noisy=noisy, **kwargs)
    mean = np.divide(k*mean + np.mean(Y), k + 1)

  logger.info('Estimated prior mean as {:.3e} in {:.3e}s'\
        .format(mean, timer() - tic))
  return mean


def build_rfgp_task(config, sess, input_dim, kernel=None, num_features=2**14,
  lenscales=None, amplitude=1.0, noise=None, dtype=None, trainable=False,
  scope='task', reuse=None, rng=None, **kwargs):
  '''
  Build a <random_feature_mapping> instance to be used as a task.
  '''
  if (rng is None): rng = npr.RandomState(config.seed)
  if (dtype is None): dtype = config.float
  if (kernel is None): kernel = config.kernel
  if (noise is None): noise = config.noise
  if (lenscales is None): lenscales = config.lenscale
  if (amplitude is None): amplitude = config.amplitude
  with tf.variable_scope(scope, reuse=reuse) as vs:

    # Generate random feature embeddings
    embedding = rf_ops.sample_embedding\
    (
      kernel_id=kernel,
      input_dim=input_dim,
      output_dim=num_features,
      log_lenscales=np.log(lenscales).astype(dtype),
      log_amplitude=np.log(amplitude).astype(dtype),
      trainable=trainable,
      dtype=dtype,
    )

    embedding_params =\
    [
      embedding.weights,
      embedding.phases,
      embedding.log_amplitude,
      embedding.log_lenscales
    ]
    embedding_init_op = tf.variables_initializer(embedding_params)
    _ = sess.run(embedding_init_op)

    # Sample \theta and initialize as <tf.Variable>
    theta_op, _ = rf_ops.sample_theta(embedding, 1)
    theta_src = sess.run(theta_op)
    theta_var = tf.get_variable('theta', initializer=theta_src, trainable=trainable)
    theta_init_op = tf.variables_initializer([theta_var])
    _ = sess.run(theta_init_op)

    # Build random feature-based sample path
    task = rf_ops.random_feature_mapping(embedding, theta_var)

    # Add task.tensorflow() method to mimic task API
    def _tensorflow(*args, noise=noise, noisy=True,
      stop_gradient=True, **kwargs):
      Y = task(*args, **kwargs)
      if (noisy and noise is not None):
        rvs = tf.random_normal(tf.shape(Y), dtype=Y.dtype)
        Y += (noise**0.5) * rvs

      if stop_gradient: #gradient-free blackbox
        Y = tf.stop_gradient(Y)
      return Y
    task.tensorflow = _tensorflow

    input_ref = tf.placeholder(task.dtype, [None, task.input_dim])
    noisy_task_op = task.tensorflow(input_ref, noisy=True)
    hidden_task_op = task.tensorflow(input_ref, noisy=False)
    def _numpy(x, noisy=True):
      if noisy: return sess.run(noisy_task_op, {input_ref:x})
      return sess.run(hidden_task_op, {input_ref:x})
    task.numpy = _numpy

    return task

