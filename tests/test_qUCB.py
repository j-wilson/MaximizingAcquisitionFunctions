#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
Test for q-UCB. Compares a direct Monte Carlo 
estimate with sequential computation via 
fantasized outcomes.

Authored: 2017-10-25 
Modified: 2018-11-27
'''

# ---- Dependencies
import numpy as np
import numpy.random as npr
import scipy.linalg as spla
import tensorflow as tf
from timeit import default_timer

import os, sys
sys.path.insert(0, os.path.join(sys.path[0], '..'))
from src import utils
from src.losses import confidence_bound

from pdb import set_trace as bp

# ---- Constants
_half_pi = 0.5 * np.pi

# ==============================================
#                                      test_qUCB
# ==============================================
def epmgp_cov_prior(ndim, scale=10.0, rng=None):
  '''
  Prior over covariance matrices from Cunningham's EPMGP paper.
  '''
  if (rng is None): rng = npr
  eigen_vals = np.diag(rng.exponential(scale, size=[ndim]))
  orthog_mat = spla.svd(rng.randn(ndim, ndim))[0]
  covariance = np.dot(orthog_mat, np.inner(eigen_vals, orthog_mat))
  return covariance


def test_1ucb(mu, var, beta=1.0, num_samples=int(1e6), rvs=None, rng=None):
  if (rng is None): rng = npr
  if (rvs is None): rvs = rng.randn(num_samples)
  samples = mu + np.abs(np.sqrt(_half_pi * beta * var)*rvs)
  ucb_mc = np.mean(samples, 0)
  ucb_cf = mu + np.sqrt(beta * var)
  return ucb_mc, ucb_cf


def test_2ucb(sess, mu, K, beta=2.0, num_samples=int(1e5),
  shard_size=int(1e7), rng=None):
  '''
  Compares a joint sampling procedure for estimating 2-UCB with
  the following sequential alternative.
    For k = 1, ..., n:
      1. Sample y0^{k} from prior distribution of 1-UCB integral at x0
      2. Condition x1 on y0^{k} and sample from (conditional) 1-UCB prior
      3. Estimate 2-UCB(x0, x1 | y0^{k}) as: E[max(y0_{k}, y1|y0_{k})]
    Estimate 2-UCB by averaging over (3)
  '''
  if (rng is None): rng = npr

  q = len(mu)
  assert q == 2, "parallelism 'q' should equal 2"

  # Rescale covariance
  K_tilde = _half_pi * beta * K

  # Precompure required terms
  L_tilde = spla.cholesky(K_tilde, lower=True)
  s_tilde = np.sqrt(np.diag(K_tilde))

  # Condition x1 | x0
  xorr = K_tilde[0, 1]/np.prod(s_tilde)
  k_11 = K_tilde[1, 1]*(1 - xorr**2) #conditional cov

  # Draw samples from prior distribution of 1-UCB 
  # integral evaluated at x0
  rvs = npr.randn(num_samples, q)
  r0s = np.sqrt(K_tilde[0, 0]) * rvs[:, :1] #residuals

  # Condition on y0 ~ N(mu0; beta_hat*sigma0)
  xorr = K_tilde[0, 1]/np.prod(s_tilde)
  d_mu = xorr*s_tilde[1]/s_tilde[0]
  k_11 = K_tilde[1, 1]*(1 - xorr**2) #conditional cov

  # Sample y1|y0 and compute 2-UCB as E[max(y0, y1|y0)]
  num_shards = int(np.ceil(num_samples**2/shard_size))
  r0_shards = np.array_split(r0s, num_shards, axis=0)

  seq_2ucb_shards = []
  loop_start = default_timer()
  for i, r0_shard in enumerate(r0_shards):
    mu1 = mu[1] + d_mu*r0_shard
    y1s = mu1 + np.abs(np.sqrt(k_11) * rvs[:, 1])
    extrema = np.maximum(mu[0] + np.abs(r0_shard), y1s)
    seq_2ucb_shards.append(np.mean(extrema, axis=-1)) #avg. over samples
    utils.progress_bar((i+1)/num_shards, default_timer() - loop_start)
  seq_2ucb_shards = np.asarray(seq_2ucb_shards)
  seq_2ucb = np.mean(seq_2ucb_shards)

  # Directly estimate q-UCB via Monte Carlo integration
  qUCB = confidence_bound(lower=False, beta=beta)
  ucb_op = qUCB._monte_carlo\
  (
    means=tf.constant(mu),
    cov=tf.constant(K),
    sample_rvs=tf.constant(rvs),
  )
  joint_2ucb = np.squeeze(sess.run(ucb_op))

  # [!] Same as the above but using placeholders
  # qUCB = confidence_bound(lower=False, beta=beta)
  # _mu = tf.placeholder(tf.as_dtype(mu.dtype), shape=[None, None])
  # _K = tf.placeholder(tf.as_dtype(K.dtype), shape=[None, None])
  # _rvs = tf.placeholder(tf.as_dtype(rvs.dtype), shape=[None, None])
  # ucb_op = qUCB.monte_carlo(means=_mu, cov=_K, sample_rvs=_rvs)
  # joint_2ucb = np.squeeze(sess.run(ucb_op, {_mu:mu, _K:K, _rvs:rvs}))

  return joint_2ucb, seq_2ucb


if __name__ == '__main__':
  # Define bivariate normal distribution
  q = 2 #parallelism
  K = epmgp_cov_prior(q)
  mu = npr.randn(q, 1)
  print('Mean:\n', mu, '\n\nCovariance:\n', K)

  # Test marginal UCB
  ucb_mc, ucb_cf = test_1ucb(mu[0], K[0,0], beta=2.0)

  # Compute joint and sequential q-UCB estimates
  with tf.Session() as sess:
    joint_2ucb, seq_2ucb = test_2ucb(sess, mu, K)

  rel_diff = np.divide(seq_2ucb - joint_2ucb, np.abs(seq_2ucb))
  print(rel_diff, seq_2ucb, joint_2ucb)
  bp()