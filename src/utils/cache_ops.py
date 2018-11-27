#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
Operations for retrieving/creating cached elements
of a TensorFlow graph.

Authored: 2017-12-27 
Modified: 2018-02-23
'''

# ---- Dependencies
import tensorflow as tf
from functools import reduce

# Ensure dictionaries are ordered
__dict__ = dict #reference to original dict class
from sys import version_info
if (version_info.major + 0.1 * version_info.minor < 3.6):
  from collections import OrderedDict as dict

# Manifest
__all__ =\
[
  'get_or_create_ref',
  'get_or_create_var',
  'get_or_create_node',
]

# Constants
_delim = '/'

# ==============================================
#                                      cache_ops
# ==============================================
def dict2tuple(dictionary):
  items = list() #list of <dict> items
  for key in sorted(dictionary.keys()):
    val = dictionary[key]
    if isinstance(val, (__dict__, dict)):
      items.append((key, dict2tuple(val)))
    else:
      items.append((key, val))
  return tuple(items)


def get_id(**kwargs):
  return dict2tuple(kwargs)


def shape_to_str(shape, delim='_'):
  if (shape is None): return 'unknown' #undefined case
  if len(shape) == 0: return '_' #scalar case

  parts = []
  for dim in shape:
    if dim is None: parts.append('None')
    else: parts.append(str(dim))
  return delim.join(parts)


def get_group(cache, group, permissive=True, delim=_delim):
  if (group is None): return cache #noop
  hierarchy = group.split(delim)
  for term in hierarchy:
    if permissive: #group doesn't have to exist beforehand
      cache = cache.setdefault(term, dict())
    else: #group must already exist
      cache = cache[term]
  return cache


def get_name(group, dtype, shape=None, delim=_delim):
  if not isinstance(dtype, str): dtype = tf.as_dtype(dtype).name
  parts = [group, dtype]
  if (shape is not None):
    if isinstance(shape, str): parts.append(shape)
    else: parts.append(shape_to_str(shape))
  name = delim.join(filter(None.__ne__, parts))
  return name


def get_or_create_node(cache, fn, args=None, kwargs=None,
  group=None, control_dependencies=None, id=None):
  if (args is None): args = tuple()
  if (kwargs is None): kwargs = dict()

  if (id is None):
    id = get_id\
    (
      fn=fn,
      args=args,
      kwargs=kwargs,
      control_dependencies=control_dependencies,
    )

  local_cache = get_group(cache, group)
  op = local_cache.get(id, None)
  if (op is None):
    if (control_dependencies is None):
      op = fn(*args, **kwargs)
    else:
      with tf.control_dependencies(control_dependencies):
        op = fn(*args, **kwargs)
    local_cache[id] = op #store node for later use

  return op


def get_or_create_ref(cache, group, dtype, shape_or_rank=None,
  name=None, control_dependencies=None, id=None, delim=_delim,
  **kwargs):
  if (control_dependencies is not None): #[!] double check me
    raise Exception('Placeholders cannot have control dependencies')

  # Determine shape for <tf.Placeholder>
  if isinstance(shape_or_rank, int): shape = shape_or_rank * (None,)
  elif isinstance(shape_or_rank, list): shape = tuple(shape_or_rank)
  else: shape = shape_or_rank #either a tuple or None 

  super_group = delim.join(filter(None.__ne__, ('refs', group)))
  if (name is None): name = get_name(super_group, dtype, shape, delim=delim)

  return get_or_create_node\
  (
    cache=cache,
    group=super_group,
    fn=tf.placeholder,
    args=(dtype, shape,),
    kwargs={**kwargs, 'name':name},
    id=id,
  )


def get_or_create_var(cache, group, dtype, shape, name=None,
  control_dependencies=None, id=None, delim=_delim, **kwargs):

  super_group = delim.join(filter(None.__ne__, ('vars', group)))
  if isinstance(shape, list):
    shape = tuple(shape)

  if (name is None):
    name = get_name(super_group, dtype, shape, delim=delim)

  return get_or_create_node\
  (
    cache=cache,
    group=super_group,
    fn=tf.get_variable,
    args=(name, shape, dtype),
    kwargs=kwargs,
    control_dependencies=control_dependencies,
    id=id,
  )


# ==============================================
#                            Developers' Section
# ==============================================
# ------------ Scrap work goes here ------------
'''
'''