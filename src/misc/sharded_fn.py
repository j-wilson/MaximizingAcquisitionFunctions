#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
Authored: 2018-01-20 
Modified: 2018-04-11

To Do:
  - Better testing for: exact_sharding==False
'''

# ---- Dependencies
import tensorflow as tf

# ==============================================
#                                     sharded_fn
# ==============================================
def join_shards(shards, shape=None):
  '''
  Reshape shards by merging across the first two axes.
  '''
  if isinstance(shards, (tuple, list)):
    if (shape is None): shape = len(shards) * (None,)
    zipper = zip(shards, shape)
    return shards.__class__([join_shards(S,s) for S,s in zipper])
  else:
    if (shape is None): #combine first two axes
      shape = tf.concat([[-1], tf.shape(shards)[2:]], 0)
    return tf.reshape(shards, shape)


def adjust_shapes(shapes, num_remainder):
  '''
  Modify provided shapes to account for excluded remainder.
  '''
  if isinstance(shapes, (tuple, list)):
    return shapes.__class__([adjust_shapes(s, num_remainder) for s in shapes])
  else:
    return tf.concat([shapes[:1] - num_remainder, shapes[1:]], axis=0)


def append_remainder(retvals, remainder):
  '''
  Append additional return values to existing ones.
  '''
  if isinstance(retvals, (tuple, list)):
    zipper = zip(retvals, remainder)
    return retvals.__class__([append_remainder(Y, y) for Y,y in zipper])
  else:
    return tf.concat([retvals, remainder], axis=0)


def shard_tensor(tensor, shard_limit, exact_sharding=False):
  '''
  Reshape a tensor s.t. it's first axis represents manageable
  shards over which computations can be run seperately.

  Unless indicated otherwise, the tensor will be broken up into a
  set of shards and a remainder.
  '''
  shape = tf.shape(tensor)
  num_slices = shape[0] #num. slices along the first axis
  shard_size = tf.minimum(num_slices, shard_limit)

  # Split up X into mapped shards and a remainder
  if exact_sharding:
    num_shards = tf.floor_div(num_slices, shard_size)
    num_mapped = num_slices
    mapped_part, remainder = tensor, None
  else:
    num_shards = tf.cast(tf.ceil(num_slices/shard_size) - 1,
                          dtype=shard_size.dtype)
    num_mapped = num_shards * shard_size #num. slices computed via map_fn
    partitions = [num_mapped, num_slices - num_mapped]
    mapped_part, remainder = tf.split(tensor, partitions)

  # Reshape mapped part of tensor into (n+1)-dimensional shards
  sharding = tf.concat([num_shards[None], shard_size[None], shape[1:]], 0)
  shards = tf.reshape(mapped_part, sharding)
  return shards, remainder


def shard_tensors(tensors, shard_limit, exact_sharding=False):
  '''
  Wrapper for sharding multiple tensors with potentially
  nested structure (e.g. list of tuples).
  '''
  shards, remainder = [], []
  for k, item in enumerate(tensors):
    if isinstance(item, (tuple, list)):
      shards_k, remainder_k = shard_tensors(item, shard_limit, exact_sharding)
    else:
      shards_k, remainder_k = shard_tensor(item, shard_limit, exact_sharding)
    shards.append(shards_k)
    remainder.append(remainder_k)

  # Maintain typing of input structure
  shards = tensors.__class__(shards)
  remainder = tensors.__class__(remainder)
  return shards, remainder


def sharded_fn(fn, *args, retval_shapes=None, shard_limit=2**10,
  map_kwargs=None, exact_sharding=False, **kwargs):
  '''
  Help limit memory footprint by mapping over shards
  along the first axis of a set of input arguments.
  '''
  if (map_kwargs is None): map_kwargs = dict()
  shards, remainder = shard_tensors(args, shard_limit, exact_sharding)

  # Evaluate mapped part of X in sharded fashion and reshape
  sharded_retvals = tf.map_fn\
  (
    fn=lambda shard: fn(*shard, **kwargs),
    elems=shards,
    **map_kwargs
  )

  # Adjust provided shapes to account for remainder
  if (retval_shapes is not None and not exact_sharding):
    retval_shapes = adjust_shapes(retval_shapes, tf.shape(remainder[0])[0])
  retvals = join_shards(sharded_retvals, retval_shapes)

  # Evaluate remainder and append to retvals
  if not exact_sharding:
    retvals = append_remainder(retvals, fn(*remainder, **kwargs))

  return retvals


# ==============================================
#                            Developers' Section
# ==============================================
# ------------ Scrap work goes here ------------
'''
'''
