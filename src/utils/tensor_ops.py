#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
Utility methods for operating on TensorFlow 
tensor objects.

Authored: 2017-10-16
Modified: 2018-05-29
'''

# -------- Dependencies
import numpy as np
import tensorflow as tf

__all__ =\
[
  'tensor_rank',
  'set_partial_shape',
  'set_shape_as',
  'expand_n',
  'expand_to',
  'expand_as',
  'squeeze_n',
  'squeeze_to',
  'squeeze_as',
  'tile_as',
  'swap_axes',
  'ravel_multi_index',
  'vectorized_gather',
  'safe_concat',
  'nnz',
  'nan_count',
  'val_count',
  'extract_last_n',
]

# ==============================================
#                                      array_ops
# ==============================================
def tensor_rank(tensor):
  '''
  Returns the dimensionality of a (static) tensor
  as a raw int value
  '''
  return len(tensor.get_shape().as_list())


def set_partial_shape(src, shape_map=None):
  if (shape_map is None): shape_map = {}
  rank, shape = tensor_rank(src), []
  for d in range(rank):
    shape += [shape_map[d] if d in shape_map else src.get_shape()[d]]
  src.set_shape(shape); return src


def set_shape_as(src, ref, shape_map=None):
  if (shape_map is None): shape_map = {}
  rank, shape = tensor_rank(src), []
  for d in range(rank):
    shape += [shape_map[d] if d in shape_map else ref.get_shape()[d]]
  src.set_shape(shape); return src


def expand_n(tensor, num_ranks, axis=0, set_shape=False):
  with tf.name_scope('expand_n') as scope:
    if num_ranks == 0: return tf.identity(tensor) #noop
    new_shape = tensor.get_shape().as_list()
    for k in range(num_ranks):
      tensor = tf.expand_dims(tensor, axis=axis)
      if set_shape: new_shape.insert(axis, 1)
    if set_shape: tensor.set_shape(new_shape)
    return tensor


def expand_to(tensor, rank, **kwargs):
  with tf.name_scope('expand_to') as scope:
    return expand_n(tensor, rank - tensor_rank(tensor), **kwargs)


def expand_as(a, b, axis=0, rank_a=None, rank_b=None, **kwargs):
  with tf.name_scope('expand_as') as scope:
    if (rank_a is None): rank_a = tensor_rank(a)
    if (rank_b is None): rank_b = tensor_rank(b)
    with tf.control_dependencies([tf.greater_equal(rank_b, rank_a)]):
      return expand_n(a, rank_b - rank_a, axis=axis, **kwargs)


def squeeze_n(tensor, n_rank, axis=0):
  for iter in range(n_rank):
    tensor = tf.squeeze(tensor, axis=axis)
  return tensor


def squeeze_to(tensor, rank, **kwargs):
  with tf.name_scope('squeeze_to') as scope:
    return squeeze_n(tensor, rank - tensor_rank(tensor), **kwargs)


def squeeze_as(a, b, rank_a=None, rank_b=None, **kwargs):
  with tf.name_scope('squeeze_as') as scope:
    if (rank_a is None): rank_a = tensor_rank(a)
    if (rank_b is None): rank_b = tensor_rank(b)
    with tf.control_dependencies([tf.less_equal(rank_b, rank_a)]):
      return squeeze_n(tensor, rank_a - rank_b, **kwargs)


def tile_as(a, b, **kwargs):
  with tf.name_scope('tile_as') as scope:
    shape_a = tf.unstack(tf.shape(a))
    shape_b = tf.unstack(tf.shape(b))
    tiling = [tf.div(*sizes) for sizes in zip(shape_b, shape_a)]
    return tf.tile(a, tiling)


def swap_axes(src, axis1, axis2, name=None):
  if (axis1 == axis2): return tf.identity(src, name=name) #noop
  perm = list(range(tensor_rank(src))) #only works for static shapes...
  a1, a2 = perm[axis1], perm[axis2] #allows use of negative axes
  perm[axis1] = a2
  perm[axis2] = a1
  return tf.transpose(src, perm=perm, name=name)


def ravel_multi_index(multi_index, shape):
  '''
  Equivalent of 'numpy.ravel_multi_index' implemented 
  in TensorFlow
  '''
  with tf.name_scope('ravel_multi_index') as scope:
    # Start with the innermost dimension
    n_dims = len(multi_index)
    flat_index, stride = 0, 1
    for d in range(1, n_dims+1):
      flat_index += stride * multi_index[-d]
      stride *= shape[-d]
    return flat_index


def vectorized_gather(src, multi_index=None, flat_index=None):
  '''
  Vector indexing-based variant of 'tf.gather'
  '''
  with tf.name_scope('vectorized_gather') as scope:
    shape = tf.shape(src)
    size = tf.reduce_prod(shape)
    if (flat_index is None):
      flat_index = ravel_multi_index(multi_index, shape)
    return tf.gather(tf.reshape(src, [size]), flat_index)


def safe_concat(*tensors, filter_op=None, dtype=None, **kwargs):
  '''
  Format one or more tensors as a single input tensor, while
  filtering out None-valued list entries.
  '''
  with tf.name_scope('safe_concat') as scope:
    if (filter_op is not None):
      tensors = filter(filter_op, tensors)
    if (dtype is not None):
      tensors = [tf.cast(tensor, dtype) for tensor in tensors]
    return tf.concat(list(tensors), **kwargs)


def nnz(tensor, axis=-1, dtype='int32', skip_abs=False):
  '''
  Compute the number of non-zero tensor slices across
  a given (ordered) set of axis.
  '''
  if not skip_abs: tensor = tf.abs(tensor)
  if isinstance(axis, (list, tuple)):
    if len(axis) > 1:
      tensor = tf.reduce_max(tensor, axis=axis[:-1])
    axis = axis[-1]
  nonzero = tf.sign(tensor)
  nnz = tf.reduce_sum(nonzero, axis=axis)
  return tf.cast(nnz, dtype)


def nan_count(tensor, axis=-1,dtype='int32'):
  tensor = tf.is_nan(tensor)
  if isinstance(axis, (list, tuple)):
    if len(axis) > 1:
      tensor = tf.reduce_any(tensor, axis=axis[:-1])
    axis = axis[-1]
  nan_count = tf.reduce_sum(tf.to_int32(tensor), axis=axis)
  return tf.cast(nan_count, dtype)


def val_count(tensor, val, axis=-1, dtype='int32'):
  non_nan = tf.assert_equal(tf.reduce_any(tf.is_nan(val)), False)
  with tf.control_dependencies([non_nan]):
    tensor = tf.equal(tensor, val)
    if isinstance(axis, (list, tuple)):
      if len(axis) > 1:
        tensor = tf.reduce_any(tensor, axis=axis[:-1])
      axis = axis[-1]
    count = tf.reduce_sum(tf.to_int32(tensor), axis=axis)
    return tf.cast(count, dtype)


def extract_last_n(sequences, lengths, n):
  with tf.name_scope('extract_last_n') as scope:
    n_sequences = tf.shape(sequences)[0]
    full_length = tf.shape(sequences)[1]
    feature_dim = sequences.shape[2].value  #can cause problems if unknown
    if (feature_dim is None):
      feature_dim = tf.shape(sequences)[2]

    checks = [tf.assert_positive(n), tf.assert_positive(lengths), 
              tf.assert_less_equal(n, full_length)]
    with tf.control_dependencies(checks):
      indices = tf.reshape(tf.expand_dims(
                    lengths + full_length*tf.range(n_sequences), -1)\
                  + tf.range(-n, 0)[None, :], [n * n_sequences])#implict -1

    values = tf.reshape(sequences, [-1, feature_dim])
    shape = [n_sequences, n, feature_dim]
    return tf.reshape(tf.gather(values, indices), shape)


# ==============================================
#                            Developers' Section
# ==============================================
# ------------ Scrap work goes here ------------
'''
'''