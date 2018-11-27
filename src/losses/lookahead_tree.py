#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
Tree-structured approximation to lookahead
integral.

Authored: 2017-02-24
Modified: 2018-01-28
'''

# -------- Dependencies
import numpy as np
import tensorflow as tf
from src import utils

# from numpy.polynomial.legendre import leggauss
from numpy.polynomial.hermite import hermgauss
from pdb import set_trace as bp

# ==============================================
#                                 lookahead_tree
# ==============================================
class lookahead_tree(object):
  def __init__(self, integrand, arity=2, height=2, model=None, **kwargs):
    self.integrand = integrand
    self.arity = arity
    self.height = height
    self.model = model

    # Quadrature terms (pre-multiplied)
    # abscissae, importances = leggauss(self.arity)
    abscissae, importances = hermgauss(self.arity)
    self._abscissae = np.sqrt(2)*abscissae
    self._importances = importances/np.sqrt(np.pi)


  def evaluate(self, *args, **kwargs):
    with tf.name_scope('evaluate'):
      return self.root_node(*args, **kwargs)


  def precompute(self, tree_inputs, prev_inputs, prev_outputs,
    model=None, is_continguous=True, **kwargs):
    '''
    Precompute required terms updated when evaluating 
    the lookahead tree.
    '''
    if (model is None): model = self.model
    with tf.name_scope('precompute') as scope: 
      # Reorder node inputs into contiguous subtrees
      if not is_continguous:
        raise Exception('Subroutine contiguous subtree indices is broken...')
        tree_height = self.get_tree_height(tree_inputs, arity)
        indices = self.index_subtree(tree_height, arity=arity)
        tree_inputs = tf.gather(tree_inputs, indices)

      # [!] Only a sparse subset of k(tree, tree) is needed...
      mean_t = model.mean_fn(tree_inputs, prev_outputs)
      cov_tt = model.covar_fn(tree_inputs, tree_inputs)
      xov_pt = model.covar_fn(prev_inputs, tree_inputs)

      cov_pp = model.covar_fn(prev_inputs, prev_inputs, noisy=True)
      inv_pp = model.compute_precision(prev_inputs, cov=cov_pp, noisy=False)
      return mean_t, cov_tt, xov_pt, inv_pp


  def root_node(self, tree_inputs, prev_inputs, prev_outputs, 
    required_terms=None, arity=None, height=None, **kwargs):
    '''
    Subroutine for root node of lookahead tree.
    '''
    if (arity is None): arity = self.arity
    if (height is None): height = self.height
    with tf.name_scope('root_node') as scope: 
      if (required_terms is None):
        required_terms = self.precompute(tree_inputs, prev_inputs, 
                                          prev_outputs, arity=arity, 
                                          **kwargs)
      return self.build_node(height, tree_inputs, prev_inputs, prev_outputs,
                              *required_terms, arity=arity, **kwargs)


  def build_node(self, height, *args, **kwargs):
    if height == 0:
      return self.terminal_node(height, *args, **kwargs)
    else: 
      return self.nonterminal_node(height, *args, **kwargs)


  def nonterminal_node(self, height, tree_inputs, prev_inputs,
    # prev_outputs, mean_t, cov_tt, xov_pt, inv_pp, fantasies=None,
    prev_outputs, mean_t, cov_tt, xov_pt, inv_pp,
    arity=None, noise=None, jitter=None, **kwargs):
    '''
    Subroutine for non-terminal nodes (i.e. root and internal nodes).
    '''
    if (arity is None):
      arity = self.arity
    if (noise is None and self.model): 
      noise = getattr(self.model, 'noise', 0.0)
    if (jitter is None and self.model): 
      jitter = getattr(self.model, 'jitter', 1e-6)

    with tf.name_scope('nonterminal_node') as scope:
      # Compute posterior about current node
      global_mean = tf.reduce_mean(prev_outputs)
      prior_mean = tf.fill([1, 1], global_mean)
      prior_var = cov_tt[..., :1, :1]
      residuals = prev_outputs - global_mean
      post_mean, post_var = self.get_marginal_posterior(prior_mean, prior_var,
                                            xov_pt[...,:1], inv_pp, residuals)

      # Apply noise/jitter to prior variance
      if (noise is not None): prior_var = prior_var + noise
      if (jitter is not None): prior_var = prior_var + jitter

      # Update observed input terms
      new_prev_inputs = tf.concat([prev_inputs, tree_inputs[:1]], axis=0)
      new_inv_pp = utils.update_matrix_inverse(inv_pp, xov_pt[...,0], prior_var)
      new_xov_pt = tf.concat([xov_pt, cov_tt[...,:1,:]], axis=-2)
  
      # Integrate out stochastic part of update via quadrature
      subtree_size = self.get_tree_size(height - 1, arity=arity)
      def quadrature_fn(mapped_args, kwargs=kwargs):
        # Update observed outcome terms
        k, fantasy = mapped_args

        # Compute local result given k-th fantasy and grow prev. outputs
        result_k = self.integrand(samples=fantasy,
                                  keep_dims=False,
                                  **kwargs)
        prev_outputs_k = tf.concat([prev_outputs, fantasy], axis=0)

        # Condition keyword arguments on fantasy; [?] improve me
        kwargs_k = self.integrand.update_kwargs(new_prev_inputs, 
                                                prev_outputs_k,
                                                prev_slice=slice(-1,None),
                                                **kwargs)

        # Spawn child node
        slice_k = slice(1 + subtree_size*k, 1 + subtree_size*(k+1))
        subtree_result = self.build_node(height - 1,
                                tree_inputs[..., slice_k, :],
                                new_prev_inputs,
                                prev_outputs_k,
                                mean_t[..., slice_k, :], 
                                cov_tt[..., slice_k, slice_k],
                                new_xov_pt[..., slice_k],
                                new_inv_pp,
                                arity=arity,
                                **kwargs_k)

        return result_k + subtree_result

      fantasies = self.get_fantasies(post_mean, post_var)
      mapped_args = (tf.range(arity), fantasies)
      results = tf.map_fn(quadrature_fn, mapped_args, mean_t.dtype)
      result = tf.tensordot(self.importances, results, [[-1], [0]])
      return result


  def terminal_node(self, height, tree_inputs, prev_inputs, prev_outputs,
    mean_t, cov_tt, xov_pt, inv_pp, arity=None, **kwargs):
    # mean_t, cov_tt, xov_pt, inv_pp, fantasies=None, arity=None, **kwargs):
    '''
    Subroutine for terminal nodes (i.e. leaf nodes).
    '''
    with tf.name_scope('terminal_node') as scope:
      # Compute posterior about current node
      global_mean = tf.reduce_mean(prev_outputs)
      prior_mean = tf.fill([1], global_mean)
      prior_var = cov_tt[...,0,0]
      residuals = prev_outputs - global_mean
      post_mean, post_var = self.get_marginal_posterior(prior_mean, prior_var,
                                              xov_pt[...,0], inv_pp, residuals)
      # Fantasize outcomes
      fantasies = self.get_fantasies(post_mean, post_var,
                              proj=self.abscissae[:,None], axis=-2)

      # Integrate over marginal posterior to compute local result
      result = self.integrand(samples=fantasies,
                              weights=self.importances,
                              **{**kwargs, 'keep_dims':False})
      return result


  def get_fantasies(self, mean, var, stdv=None, proj=None, axis=0):
    '''
    [!] Should be generalized to handle arbitary tensor
        shapes and axes.
    '''
    if (proj is None): proj = self.abscissae
    with tf.name_scope('get_fantasies') as scope:
      if (stdv is None): stdv = tf.sqrt(var)
      _mean = tf.expand_dims(mean, axis=axis)
      _stdv = tf.expand_dims(stdv, axis=axis)
      if axis == 0:
        proj = utils.expand_as(proj, _stdv, axis=-1)
      fantasies = _mean + _stdv*proj
      return fantasies


  def get_marginal_posterior(self, mean, var, xov, precis, resid):
    ''' 
    Compute marginal Gaussian posteriors given prior terms.
    # [?] use einsum for better parallel computation
    '''
    rank = utils.tensor_rank(xov)
    axes = [[max(0, rank - 2)], [0]]
    beta = tf.tensordot(xov, precis, axes)
    if rank > 1: xov = utils.swap_axes(xov, -1, -2) #matrix transpose
    return\
    (
      mean + tf.tensordot(beta, resid, [[-1], [0]]),
      var - tf.reduce_sum(beta*xov, axis=-1, keep_dims=True)
    )


  def get_tree_height(self, array, arity=None, dtype='int32'):
    '''
    Depth of a perfect tree of a given arity, with node-wise
    values stored in as an array.
    '''
    if (arity is None): arity = self.arity

    arity = tf.cast(arity, 'float64')
    num_nodes = tf.cast(tf.shape(array)[0], 'float64')
    height = tf.log((arity-1)*num_nodes + 1)/tf.log(arity) - 1
  
    return tf.cast(height, dtype=dtype)


  def get_tree_size(self, height, arity=None, dtype='int32'):
    '''
    Number of nodes in a perfect tree with given height and arity.
    '''
    if (arity is None): arity = self.arity
    return tf.cast((arity**(height + 1) - 1)/(arity - 1), dtype=dtype)


  #[!] temp hacks: not a good idea too add these to the graph
  # each time; but, this covers various edge-cases better.
  @property
  def abscissae(self):
    return tf.constant(self._abscissae)

  @property 
  def importances(self):
    return tf.constant(self._importances)

  def __call__(self, *args, **kwargs):
    return self.evaluate(*args, **kwargs)


# ==============================================
#                            Developers' Section
# ==============================================
# ------------ Scrap work goes here ------------
'''
'''