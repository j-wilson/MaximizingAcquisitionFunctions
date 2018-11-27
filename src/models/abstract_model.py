#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
Authored: 2017-12-28 
Modified: 2018-02-26
'''

# ---- Dependencies
from src.core import stateful_module
from abc import abstractmethod

# ==============================================
#                                 abstract_model
# ==============================================
class abstract_model(stateful_module):
  @abstractmethod
  def predict(self, *args, **kwargs):
    raise NotImplementedError('Models must implement a predict() method')


  def __call__(self, *args, **kwargs):
    return self.predict(*args, **kwargs)