#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
Authored: 2018-02-22 
Modified: 2018-03-22
'''

# ---- Dependencies
import numpy as np
import tensorflow as tf
from copy import deepcopy
from src.utils import cache_ops
from pdb import set_trace as bp


# Ensure dictionaries are ordered
from sys import version_info
if (version_info.major + 0.1 * version_info.minor < 3.6):
  from collections import OrderedDict as dict

# ==============================================
#                                         module
# ==============================================
class module(object):
  def __init__(self, dtype='float64', cache=None, seed=None, name=None):
    if (name is None): name = self.__class__.__name__
    if (seed is None): seed = np.random.choice(2**31 - 1)
    if (cache is None): cache = dict()

    # Property-style attributes
    self._name = name
    self._dtype = dtype
    self._cache = cache

    # Current handling of pRNG seeds is probably overkill...
    if (seed is None): seed = np.random.choice(2**31 - 1)
    self._seed = np.array([seed], dtype=np.int64)
    self._rng = np.random.RandomState(self._seed)


  def get_or_create_node(self, group, fn, args=None, cache=None, **kwargs):
    if (cache is None): cache = self.cache
    return cache_ops.get_or_create_node\
    (
      cache=cache,
      group=group,
      fn=fn,
      args=args,
      **kwargs,
    )


  def get_or_create_ref(self, group, shape_or_rank=None, dtype=None,
    cache=None, **kwargs):
    if (dtype is None): dtype = self.dtype #assume default data type
    if (cache is None): cache = self.cache

    return cache_ops.get_or_create_ref\
    (
      cache=cache,
      group=group,
      dtype=dtype,
      shape_or_rank=shape_or_rank,
      **kwargs,
    )


  def get_or_create_var(self, group, shape=None, dtype=None, cache=None,
    stateful=False, **kwargs):
    if (dtype is None): dtype = self.dtype #assume default data type
    if (cache is None): cache = self.cache
    return cache_ops.get_or_create_var\
    (
      cache=cache,
      group=group,
      dtype=dtype,
      shape=shape,
      **kwargs,
    )


  def update_dict(self, src, updates=None, as_copy=False):
    '''
    Helper function for updating nested dictionaries.
    '''
    if as_copy: src = deepcopy(src)
    if (updates is None): return src
    for key, new in updates.items():
      old = src.get(key, None)
      if isinstance(old, dict) and isinstance(new, dict):
        src[key] = self.update_dict(old, new, as_copy=False)
      else:
        src[key] = new
    return src


  @property
  def cache(self):
    return self._cache

  @cache.setter
  def cache(self, new_cache):
    assert isinstance(new_cache, dict)
    self._cache = cache


  @property
  def dtype(self):
    return self._dtype

  @dtype.setter
  def dtype(self, new_dtype):
    assert isinstance(new_dtype, str)
    self._dtype = new_dtype


  @property
  def name(self):
    return self._name

  @name.setter
  def name(self, new_name):
    assert isinstance(new_name, str)
    self._name = name


  @property
  def seed(self): 
    return self._seed[0]

  @seed.setter
  def seed(self, new_seed):
    self._seed[:] = new_seed


  @property
  def rng(self):
    self.seed = max(1, np.mod(self.seed + 1, 2**31 - 1)) #incremement seed
    self._rng.seed(self.seed)
    return self._rng

