#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
Abstract base class for search agents.

To Do
  - Doc strings...

Authored: 2017-12-21 
Modified: 2018-11-27
'''

# ---- Dependencies
import os, time, logging
import numpy as np
import tensorflow as tf

from timeit import default_timer
from scipy.special import comb as nCr
from abc import abstractmethod

from src import utils
from src.core import module
from src.third_party.sobol_lib import i4_sobol_generate

from pdb import set_trace as bp
logger = logging.getLogger(__name__)
# ==============================================
#                                   search_agent
# ==============================================
class search_agent(module):
  def __init__(self, *args, timer=None, grids=None, **kwargs):
    super().__init__(*args, **kwargs)
    if (timer is None): timer = default_timer
    if (grids is None): grids = dict()
    self.timer = timer #defaults to wall time
    self.grids = grids #cached inputs sets (e.g. Sobol sequences)


  @abstractmethod
  def suggest_inputs(self, num_suggest, inputs_old, outputs_old,
    *args, options=None, dtype=None, **kwargs):
    raise NotImplementedError\
    (
      'Search agents must implement suggest_inputs()'
    )


  def suggest_answers(self, num_suggest, inputs_old, outputs_old,
    in_order=False):
    '''
    Recommend a set of inputs as final answers. By default,
    suggests the top-k best seen inputs.
    '''
    indices = np.argpartition(outputs_old, num_suggest - 1)[:num_suggest]
    if in_order: #return suggestions in sorted order (descending)
      indices = np.argsort(outputs_old[indices])
    return inputs_old[indices], outputs_old[indices]


  def sample_inputs(self, shape, options=None, use_sobol=False,
    use_cache=False, dtype=None):
    '''
    Wrapper for <search_agent._sample_inputs> enabling use of
    cached input sets.
    '''
    if (dtype is None): dtype = self.dtype
    if (use_cache and options is None):
      key = (use_sobol, tf.as_dtype(dtype).name, tuple(shape))
      if self.grids.get(key, None) is None:
        self.grids[key] = self._sample_inputs(shape, options, use_sobol, dtype)
      return self.grids[key]
    else:
      return self._sample_inputs(shape, options, use_sobol, dtype)


  def _sample_inputs(self, shape, options=None, use_sobol=False,
    dtype=None):
    '''
    Sample a set of input pools either from [0, 1]^{d} or using 
    combinations of a set of specified options.

    Arguments:
      - shape : [..., set_size, input_dim]
      - options : a finite set of inputs [num_options, input_dim]
    '''
    if (dtype is None): dtype = self.dtype

    if (options is None):
      if use_sobol:
        if len(shape) > 2:
          num_sets = int(np.prod(shape[:-2]))
          set_size = np.prod(shape)//num_sets
        else:
          assert len(shape) == 2
          num_sets, set_size = shape

        if (set_size > 40): #code doesn't support >40 dimensional spaces
          logger.warning('Sobol sequence not available for >40d spaces;'\
                          ' using uniform random values instead')
          inputs = self.rng.rand(*shape)
        else:
          inputs = np.reshape\
          (
            i4_sobol_generate(set_size, num_sets, self.rng.randint(2**13-1)).T,
            shape
          )
      else:
        inputs = self.rng.rand(*shape)
    else:
      assert options.shape[-1] == shape[-1], 'Input dimensionality mismatch'
      num_options = len(options)
      num_sets, set_size = np.prod(shape[:-2]), shape[-2]
      num_combs = nCr(num_options, set_size, exact=True)
      if (num_combs > num_sets):
        index_shape = [num_sets, set_size]
        indices = np.empty(index_shape, dtype='int')
        counter = 0
        while (counter < num_sets):
          new_indices = self.rng.choice(num_options, set_size, replace=False)
          if not any(np.equal(indices[:counter], new_indices).all(axis=1)):
            indices[counter] = new_indices
            counter += 1
        inputs = options[indices]
      else:
        if (num_combs != num_sets):
          logger.warning('Requested more unique sets than options allow')
        inputs = np.array(tuple(itertools.combinations(options, set_size)))
    return np.asarray(inputs, dtype=dtype)


  def sample_propto_loss(self, options, losses, num_samples,
    top_k=1, eps=None):
    '''
    Subsample from a set of options inversely proportional to
    corresponding losses.
    '''
    if (eps is None): eps = np.finfo(losses.dtype).eps

    num_options = len(options)
    assert num_options >= len(losses), 'Insufficient options provided'

    weights = np.max(losses) - losses
    mask = np.greater_equal(weights, eps)
    nnz = np.sum(mask)

    # Deterministically include top-k options
    if (top_k > 0 and nnz > 0):
      num_top = min(top_k, nnz)
      num_samples = num_samples - num_top
      top_indices = np.argpartition(losses, num_top - 1)[:num_top]
      nnz, weights[top_indices], mask[top_indices] = nnz - num_top, 0, 0
    else:
      top_indices = None

    # Sample indices
    if (num_samples == 0):
      indices = np.empty([0], dtype='int')
    elif (nnz == num_samples):
      indices = np.where(mask)[0]
    elif (nnz < num_samples):
      nz_indices = np.where(mask)[0]
      rand_indices = self.rng.choice(np.where(np.logical_not(mask))[0],
                                      num_samples - nnz, replace=False)
      indices = np.hstack([nz_indices, rand_indices])
    else:
      partition = np.sum(weights)
      if (partition > eps):
        weights /= np.sum(weights)
        weights[mask] = np.maximum(weights[mask], eps)
      else:
        weights = mask/np.sum(mask)
      indices = self.rng.choice(num_options, num_samples,
                                p=weights, replace=False)

    if (top_indices is not None):
      indices = np.hstack([top_indices, indices])

    return options[indices], indices



# ==============================================
#                            Developers' Section
# ==============================================
# ------------ Scrap work goes here ------------
'''
'''