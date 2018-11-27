#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
Random Search [Bergstra and Bengio, 2012]

Authored: 2017-12-21 
Modified: 2018-03-20
'''

# ---- Dependencies
import logging
import numpy as np
from src.agents import search_agent

# ==============================================
#                                  random_search
# ==============================================
class random_search(search_agent):
  def suggest_inputs(self, num_suggest, inputs_old, outputs_old, **kwargs):
    return self.sample_inputs([num_suggest, inputs_old.shape[-1]], **kwargs)