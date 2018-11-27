#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''

Authored: 2017-10-16
Modified: 2018-10-17
'''

# -------- Dependencies
import numpy as np
import tensorflow as tf
from string import ascii_lowercase
from .tensor_ops import\
(
  tensor_rank,
  expand_n,
  expand_to,
  expand_as
)

__all__ =\
[
  'tril',
  'triu',
  'tril_indices',
  'triu_indices',
  'pdist2',
  'jitter_diagonal',
  'jitter_cholesky',
  'jitter_inverse',
  'broadcast_matmul',
  'cholesky_solve',
  'update_cholesky',
  'update_matrix_inverse',
]

# ==============================================
#                                     linalg_ops
# ==============================================
def tril(tensor):
  '''
  Returns a copy of input 'tensor' with all
  but its matrix lower triangle zeroed out
  '''
  with tf.name_scope('tril') as scope:
    return tf.matrix_band_part(tensor, -1, 0)


def triu(tensor):
  '''
  Returns a copy of input 'tensor' with all
  but its matrix upper triangle zeroed out
  '''
  with tf.name_scope('triu') as scope:
    return tf.matrix_band_part(tensor, 0, -1)


def tril_indices(N, dtype='int32'):
  with tf.name_scope('tril_indices') as scope:
    if isinstance(N, tf.Dimension) and N.value is not None: N = N.value
    def cond(iter, indices):
      return tf.less(iter, N)
    def body(iter, indices):
      indices = tf.concat([indices, tf.range(iter+1,dtype=dtype) + N*iter], 0)
      return iter + 1, indices
    loop_vars = [tf.zeros([], dtype=dtype), tf.zeros([0], dtype=dtype)]
    invariants = [loop_vars[0].shape, tf.TensorShape([None])]
    indices = tf.while_loop(cond, body, loop_vars, shape_invariants=invariants)[1]
    if not isinstance(N, tf.Tensor): indices.set_shape([0.5 * (N**2 + N)])
    return indices


def triu_indices(N, dtype='int32'):
  with tf.name_scope('triu_indices') as scope:
    if isinstance(N, tf.Dimension) and N.value is not None: N = N.value
    def cond(iter, indices):
      return tf.less(iter, N)
    def body(iter, indices):
      indices = tf.concat([indices, tf.range(iter, N, dtype=dtype) + N*iter], 0)
      return iter + 1, indices
    loop_vars = [tf.zeros([], dtype=dtype), tf.zeros([0], dtype=dtype)]
    invariants = [loop_vars[0].shape, tf.TensorShape([None])]
    indices = tf.while_loop(cond, body, loop_vars, shape_invariants=invariants)[1]
    if not isinstance(N, tf.Tensor): indices.set_shape([0.5 * (N**2 + N)])
    return indices


def pdist2(A, B):
  '''
  Computes the pairwise squared Euclidean distances
  between pairs (a,b) in A, B.
  '''
  _A = tf.expand_dims(A, axis=-2)
  _B = tf.expand_dims(B, axis=-3)
  return tf.reduce_sum(tf.squared_difference(_A, _B), axis=-1)


def jitter_diagonal(A, jitter=None, name='jittered_tensor',
  rank=None, dtype=None):
  '''
  Add jitter to the diagonal of a square tensor of rank >= 2.
  '''
  if (rank is None): rank = len(A.get_shape().as_list())
  dtype = A.dtype if (dtype is None) else tf.as_dtype(dtype)
  if (jitter is None): jitter = 1e-6 if dtype == tf.float64 else 1e-4
  assert rank >= 2, 'Input tensor must be at least 2D'

  with tf.name_scope('jitter_diagonal') as scope:
    shape = tf.shape(A)
    test_square = tf.assert_equal(shape[-1], shape[-2], name='test_square')
    with tf.control_dependencies([test_square]):
      jittered_eye = jitter * tf.eye(shape[-1], dtype=dtype)

    return tf.add(A, jittered_eye, name=name)
    
    
def jitter_cholesky(A, jitter=None, name='jittered_chol',
  upcast=False, dtype=None, **kwargs):
  '''
  Cholesky factorization of >=2-dimensional square tensor
  'A', stabilized via additition of a small, positive
  constant to its diagonal.
  '''
  dtype_src = A.dtype
  dtype_res = dtype_src if (dtype is None) else tf.as_dtype(dtype)
  with tf.name_scope('jitter_cholesky') as scope:
    if (upcast and dtype_src != tf.float64): A = tf.cast(A, tf.float64)
    elif (dtype_src != dtype_res): A = tf.cast(A, dtype_res)

    _A = jitter_diagonal(A, jitter=jitter, dtype=None, **kwargs)
    _L = tf.cholesky(_A)

    if (_L.dtype != dtype_res): L = tf.cast(_L, dtype_res, name=name)
    elif (name is not None): L = tf.identity(_L, name=name)
    else: L = _L
    return L


def jitter_inverse(tensor, jitter=None, name='jittered_inverse', **kwargs):
    with tf.name_scope('jitter_inverse') as scope:
        jittered_tensor = jitter_diagonal(tensor, jitter=jitter, **kwargs)
        return tf.matrix_inverse(jittered_tensor, name=name)


def broadcast_matmul(A, B, transpose_a=False, transpose_b=False,
  rank_a=None, rank_b=None):
  if (rank_a is None): rank_a = len(A.get_shape().as_list())
  if (rank_b is None): rank_b = len(B.get_shape().as_list())
  assert min(rank_a, rank_b) > 1 and max(rank_a, rank_b) < 26

  def get_einsum_convention():
    if rank_a >= rank_b:
      lhs_a = ascii_lowercase[:rank_a]
      lhs_b = lhs_a[-rank_b:-2] + lhs_a[-1] + ascii_lowercase[rank_a]
      rhs = lhs_a[:-2]
    else:
      lhs_b = ascii_lowercase[:rank_b]
      lhs_a = lhs_b[-rank_a:-2] + ascii_lowercase[rank_b] + lhs_b[-2]
      rhs = lhs_b[:-2]

    rhs += lhs_a[-2] + lhs_b[-1]
    if transpose_a: lhs_a = lhs_a[:-2] + lhs_a[-1] + lhs_a[-2]
    if transpose_b: lhs_b = lhs_b[:-2] + lhs_b[-1] + lhs_b[-2]
    return '{:s},{:s}->{:s}'.format(lhs_a, lhs_b, rhs)

  convention = get_einsum_convention()
  return tf.einsum(convention, A, B)


def cholesky_solve(L, B):
  '''
  Simultaneously solve for one or more linear systems
  of the form AX = B given the Cholesky factor A = LL'.
  '''
  L_rank = len(L.get_shape().as_list())
  B_rank = len(B.get_shape().as_list())
  d = B_rank - L_rank #change in rank

  if d == 0:
    X = tf.cholesky_solve(L, B)
  elif d > 0:
    axes = list(range(B_rank))
    shape = tf.shape(B)
    B_matrix = tf.reshape\
    (
      tf.transpose(B, axes[d:-1] + axes[:d] + axes[-1:]),
      tf.concat([shape[-2:-1], [-1]], 0),
    )

    X = tf.transpose\
    (
      tf.reshape\
      (
        tf.cholesky_solve(L, B_matrix),
        tf.concat([shape[d:-1], shape[:d], shape[-1:]], 0),
      ),
      axes[-(d + 1): -1] + axes[:-(d + 1)] + axes[-1:]
    )
  else:
    # [!] test out use of, e.g, tf.linalg.solve
    # Expand and tile Cholesky factor
    tiling = tf.unstack(tf.shape(B)[:-L_rank]) + L_rank*[1]
    L_tile = tf.tile(utils.expand_as(L, B, axis=0), tiling)
    X = utils.swap_axes(tf.cholesky_solve(L_tile, B), -1, -2)

  return X


def update_cholesky(chol, x, pad_chol=True, lower=True, downdate=False):
  '''
  Rank-1 update/downdate of Cholesky decomposition matrix 'chol' 
  via innovation 'x':

    K_{t+1} <- K_{t} +/- <x, x>

  where '<>' denotes an inner-product, 'K_{t} := <chol, chol>',
  and '+/-' depends on whether an update or downdate is 
  being performed.
  '''
  add_sub = tf.add if not downdate else tf.subtract
  with tf.control_dependencies([tf.assert_rank(x, 1)]):
    num_iters = tf.size(x)

  def cond(i, *args):
    return tf.less(i, num_iters)
  
  def body(i, L, x):
    # Compute required terms
    L_ii, x_i = L[i, i], x[i]
    r2 = add_sub(L_ii**2, x_i**2)
    with tf.control_dependencies([tf.assert_non_negative(r2)]):
      r = tf.sqrt(L_ii**2 + x_i**2)
      c = r/L_ii
      s = x_i/L_ii
      l = add_sub(L[i+1:, i:i+1], s*x[i+1:])/c

    # Update Cholesky decomposition 'L'
    L_shape = L.get_shape() #shape is maintained
    L_prev = L[:i]
    L_next = tf.concat([L[i, :i], r, L[i, i+1:]], 0)[None,:]
    L_post = tf.concat([L[i+1:, :i], l, L[i+1:, i+1:]], 1)
    L = tf.concat([L_prev, L_next, L_post], axis=0)
    L.set_shape(L_shape)

    # Update innovation vector 'x'
    x_shape = x.shape
    x = tf.concat([x[:i+1], c*x[i+1:] - s*L[i+1:, i:i+1]], 0)
    x.set_shape(x_shape)
    return i+1, L, x

  if not lower: chol = tf.transpose(chol)
  if pad_chol: chol = tf.pad(chol, [[0, 1], [0, 1]])
  new_chol = tf.while_loop(cond, body, [0, chol, innov])[1]
  if not lower: new_chol = tf.transpose(new_chol)
  return new_chol


def update_matrix_inverse(inverse, xov, var=None, name=None):
  '''
  Rank one update to the inverse of a partitioned matrix.
  '''
  with tf.name_scope('update_matrix_inverse') as scope:
    if (var is None): var = tf.ones_like(xov[...,:1,:1])
    xov = expand_to(xov, 2, axis=-2)

    _beta = tf.matmul(inverse, xov, transpose_b=True)
    inv_eta = 1.0/(var - tf.matmul(xov, _beta))
    beta = inv_eta * _beta

    d_invK00 = tf.matmul(_beta, beta, transpose_b=True)

    new_inverse = tf.concat\
    ([
      tf.concat
      ([
        inverse + d_invK00,
        tf.negative(beta)
      ], axis=-1),
      tf.concat\
      ([
        tf.negative(tf.transpose(beta)),
        inv_eta
      ], axis=-1)
    ], axis=0, name=name)

    with tf.control_dependencies\
    ([
        tf.assert_equal(tf.shape(inverse), tf.shape(d_invK00))
    ]):
      return new_inverse


# ==============================================
#                            Developers' Section
# ==============================================
# ------------ Scrap work goes here ------------
'''
'''