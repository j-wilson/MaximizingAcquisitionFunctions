#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
Authored: 2018-02-26 
Modified: 2018-02-26

To Do:
  - Integration of caching system and module states.
'''

# ---- Dependencies
import numpy as np
import tensorflow as tf
from src.utils import cache_ops
from src.core import module
from pdb import set_trace as bp

# Ensure dictionaries are ordered
from sys import version_info
if (version_info.major + 0.1 * version_info.minor < 3.6):
  from collections import OrderedDict as dict

# ==============================================
#                                stateful_module
# ==============================================
class stateful_module(module):
  def __init__(self, states=None, active_state=0, **kwargs):
    super().__init__(**kwargs)
    # Property-style attributes
    self._states = dict()

    # Create module states
    if (states is not None):
      if isinstance(states, dict):
        state_defs = states.items()
      elif isinstance(states, (list, tuple, set)):
        state_defs = range(len(states)), states
      else:
        raise Exception('Unrecognized format for module states')

      # If not already, state terms will be converted to <tf.Variable>
      for state_id, state in zip(*state_defs):
        self.create_state(state_id, **state)

    # Indicator for active state
    if not isinstance(active_state, np.ndarray):
      active_state = np.array([active_state], dtype='int')
    self._active_state = active_state


  @classmethod
  def stateful(cls, fn, *args, **kwargs):
    '''
    Decorator for state-specific functions.
    '''
    def stateful_fn(*args, state_id=None, **kwargs):
      if (state_id is not None): #set temporary state
        initial_state = cls.active_state
        cls.active_state = state_id

      retvals = fn(*args, **kwargs)
      if (state_id is not None): #reset prior state
        cls.active_state = initial_state

      return retvals
    return stateful_fn


  def get_or_create_node(self, *args, stateful=False, **kwargs):
    # If stateful, ensure state ID is included in hash signature
    if stateful: #keyword args for 'fn' are passed in as 'kwargs'
      _kwargs = kwargs.setdefault('kwargs', dict())
      state_id = _kwargs.get('state_id', None)
      if (state_id is None): #overwrite <None> valued terms
        _kwargs['state_id'] = self.active_state
    return super().get_or_create_node(*args, **kwargs)


  def create_state(self, state_id=None, reuse=None, dtype=None, 
    scope=None, filter_op=None, **kwargs):
    '''
    [!] To Do:
        - Enable parameter specific keyword arguments
        - Bounds on parameter values?
    '''
    if (dtype is None): dtype = self.dtype
    if (state_id is None): state_id = len(self.states)
    if (filter_op is None): filter_op = lambda kv : kv[1] is not None

    state = self.states[state_id] = dict()
    for key, val in filter(filter_op, kwargs.items()):
      if isinstance(val, tf.Variable):
        state[key] = val
      else:
        if isinstance(val, (int, float, np.ndarray, list, tuple, set)):
          val = np.asarray(val, dtype=dtype) #helps ensure dtype
        self.create_param(key, val, dtype, scope, reuse, state_id)
    return state


  def create_param(self, param_id, initializer, dtype=None,
    scope=None, reuse=None, state_id=None, **kwargs):
    if (dtype is None): dtype = self.dtype
    if (scope is None): scope = self.get_scope(state_id)
    if (state_id is None): state_id = self.active_state
    '''
    Creates a module parameter.
    '''
    state = self._states[state_id]
    with tf.variable_scope(scope, reuse=reuse) as vs:
      state[param_id] = tf.get_variable\
      (
        name=param_id,
        dtype=dtype,
        initializer=initializer,
        **kwargs
      )


  def updates_state(self, state_id=None, filter_op=None, **updates):
    if (state_id is None): state_id = self.active_state
    if (filter_op is None): filter_op = lambda kv: True
    '''
    Update a state dictionary per se.
    '''
    state = self.states[state_id]
    for key, val in filter(filter_op, updates.items()):
      state[key] = val


  def assign_state(self, state_id=None, filter_op=None, **assignments):
    if (state_id is None): state_id = self.active_state
    if (filter_op is None): filter_op = lambda kv: True
    '''
    Create assignment ops to update a state's variables.
    [!] To do: automate cache usage.
    '''
    state = self.states[state_id]
    assign_ops = []
    for key, val in filter(filter_op, assignments.items()):
      param = state.get(key, None)
      if (param is None):
        raise LookupError("Assignment target for '{:s}' not found".format(key))
      elif not isinstance(param, tf.Variable):
        raise LookupError("Assignment target for '{:s}' is not <tf.Variable>".format(key))
      assign_ops.append(tf.assign(param, val))
    return assign_ops


  def clone_state(self, state, clone_id=None):
    if isinstance(state, int): state = self.states[state]
    src = {k : v.initialized_value() for k, v in state.items()}
    return self.create_state(state_id=clone_id, **src)


  def get_scope(self, state_id=None, delim='/'):
    if (state_id is None): state_id = self.active_state
    terms = [self.name, 'state_{:d}'.format(state_id)]
    return '/'.join(filter(None.__ne__, terms))


  @property
  def active_state(self):
    return self._active_state[0]

  @active_state.setter
  def active_state(self, state_id):
    self._active_state[:] = state_id


  @property
  def state(self):
    return self.states[self.active_state]

  @state.setter
  def state(self, new_state):
    assert isinstance(new_state, dict)
    self.states[self.active_state] = new_state


  @property
  def states(self):
    return self._states

  @states.setter
  def states(self, new_states):
    assert isinstance(new_states, dict)
    self._states = new_states


  @property
  def num_states(self):
    return len(self.states)
