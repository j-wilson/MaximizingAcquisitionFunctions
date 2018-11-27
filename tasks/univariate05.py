#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
Univariate05 benchmarking function

Source:
  http://infinity77.net/global_optimization/
  test_functions_1d.html#go_benchmark.Problem05

Authored: 2016-12-20
Modified: 2018-01-27
'''

# ---- Dependencies
import numpy as np
import tensorflow as tf
from .abstract_task import abstract_task

# ==============================================
#                                   univariate05
# ==============================================
class univariate05(abstract_task):
  def __init__(self,  *args, **kwargs):
    super(univariate05, self).__init__(**kwargs)
    self.input_dim = 1
    self.minimum = -1.48907

  def _tensorflow(self, x, *args, **kwargs):
    # Transfrom X -> Z
    z = 1.2 * x

    # Univariate05 function
    y = (3.0*z - 1.4) * tf.sin(18.0*z)
    return y

  def _numpy(self, x, *args, **kwargs):
    # Transfrom X -> Z
    z = 1.2 * x

    # Univariate05 function
    y = (3.0*z - 1.4) * np.sin(18.0*z)
    return y
