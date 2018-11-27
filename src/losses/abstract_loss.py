#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
Authored: 2017-10-19 
Modified: 2018-03-29
'''

# ---- Dependencies
import logging
from abc import abstractmethod
from src.core import module

# ==============================================
#                                  abstract_loss
# ==============================================
class abstract_loss(module):
  @abstractmethod
  def evaluate(self, *args, **kwargs):
    '''
    Main method.
    '''
    raise NotImplementedError\
    (
      '{} does not implement an evaluate() method'.format(self.__class__)
    )


  def __call__(self, *args, **kwargs):
    return self.evaluate(*args, **kwargs)


  def prepare(self, sess, inputs_old, outputs_old, parallelism, **kwargs):
    pass


  def update(self, sess, inputs_old, outputs_old, **kwargs):
    pass

