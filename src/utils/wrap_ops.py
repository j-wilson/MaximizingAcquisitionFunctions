#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
Python wrappers and decorators for reuse of 
common functional paradigms.

Authored: 2017-10-16 
Modified: 2017-10-24
'''

# ---- Dependencies
import numpy as np
import numpy.random as npr
import tensorflow as tf

# ==============================================
#                               	 			wrap_ops
# ==============================================
class gradient_wrap(object):
  '''
  Utility method for using hand-written gradients.
  The function being differentiated must accept and
  return numpy.ndarrays
  '''
  def __init__(self, grad, Tout=None, name=None):
    self.grad = grad
    self.Tout = Tout
    if (name is None):
      name = 'gradient_wrap{:06d}'.format(int(npr.choice(int(2**32 - 1))))
    self.name = name
    self.op_type_map = {'PyFunc' : name}

  def __call__(self, numpy_op):
    def gradient_wrapped_op(*args, Tout=None, shape_out=None, **kwargs):
      if (Tout is None): Tout = self.Tout
      if isinstance(Tout, str): Tout = tf.as_dtype(Tout)
      try: tf.RegisterGradient(self.name)(self.grad)
      except: pass
      with tf.get_default_graph().gradient_override_map(self.op_type_map):
        res = tf.py_func(numpy_op, args, Tout, name=self.name)
        def _set_shapes(tensors, shapes):
          if isinstance(tensors, (list, tuple)):
            tensors = [_set_shape(t, s) for t,s in zip(tensors, shapes)]

          elif isinstance(shapes, tf.Tensor):
            tensors = tf.reshape(tensors, shapes)
          else:
            tensors.set_shape(shapes)
          return tensors

        if (shape_out is not None):
          res = _set_shapes(res, shape_out)
        return res

    return gradient_wrapped_op