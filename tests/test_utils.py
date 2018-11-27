#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
Unit tests for utilities methods

Authored: 2016-12-28
Modified: 2018-11-27
'''

# --- Dependencies
import os, sys, unittest
import tensorflow as tf
import numpy as np
import numpy.random as npr
import scipy.linalg as spla
import scipy.stats as sps
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.join(sys.path[0], '..'))
from src import utils

# ---- Constants 
__TYPE__ = 'float64'
__EPS__  = np.finfo(__TYPE__).eps
__SEED__ = 1234

# -- TensorFlow placeholders
refs =\
{
  'X' : tf.placeholder(__TYPE__),
  'Y' : tf.placeholder(__TYPE__),
}

from pdb import set_trace as bp
# ==============================================
#                                     test_utils
# ==============================================
class utils_tests(unittest.TestCase):
  def test_pdist2(self):
    rng = npr.RandomState(__SEED__)
    N, M, D = 256, 128, 32
    X = rng.randn(N, D)
    Y = rng.randn(M, D)

    with tf.Session() as sess:
      test_fn = utils.pdist2(refs['X'], refs['Y'])
      dist_XX = sess.run(test_fn, feed_dict={refs['X']:X, refs['Y']:X})
      dist_XY = sess.run(test_fn, feed_dict={refs['X']:X, refs['Y']:Y})

    self.assertTrue(np.all(dist_XY >= 0.0))
    self.assertTrue(np.all(dist_XX >= 0.0))

    true_XX = cdist(X, X, metric='sqeuclidean')
    true_XY = cdist(X, Y, metric='sqeuclidean')

    def test_rel_abs_diff(A, B, ERROR_TOL=None):
      if (ERROR_TOL is None): ERROR_TOL = __EPS__
      abs_diff = np.abs(A - B).max()
      norm = spla.norm(A)
      self.assertTrue(abs_diff/norm < ERROR_TOL)

    test_rel_abs_diff(true_XX, dist_XX)
    test_rel_abs_diff(true_XY, dist_XY)

  def test_update_matrix_inverse(self):
    rng = npr.RandomState(__SEED__)

    def epmgp_cov_prior(ndim, scale=10.0, rng=None):
      '''
      Prior over covariance matrices from Cunningham's EPMGP paper.
      '''
      if (rng is None): rng = npr
      eigen_vals = np.diag(rng.exponential(scale, size=[ndim]))
      orthog_mat = spla.svd(rng.randn(ndim, ndim))[0]
      covariance = np.dot(orthog_mat, np.inner(eigen_vals, orthog_mat))
      return covariance

    K = epmgp_cov_prior(17)

    K_00 = K[:-1, :-1]
    K_01 = K[-1:, :-1]
    K_11 = K[-1:, -1:]

    R_test = spla.inv(K)
    R_00 = spla.inv(K_00)

    with tf.Session() as sess:
      R_rank1 = sess.run(utils.update_matrix_inverse\
      (
        tf.constant(R_00), 
        tf.constant(K_01),
        tf.constant(K_11)
      ))

    def test_rel_abs_diff(A, B, ERROR_TOL=None):
      if (ERROR_TOL is None): ERROR_TOL = __EPS__
      abs_diff = np.abs(A - B).max()
      norm = spla.norm(A)
      self.assertTrue(abs_diff/norm < ERROR_TOL)

    test_rel_abs_diff(R_test, R_rank1, 1e-8)


if __name__ == '__main__':
  unittest.main()
