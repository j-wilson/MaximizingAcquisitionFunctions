#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ =\
[	
	'abstract_loss',

	# Myopic loss functions
	'myopic_loss',
	'simple_regret',
	'negative_ei',
	'negative_pi',
	'confidence_bound',

  # Nonmyopic loss functions
  'nonmyopic_loss',
  'entropy_search',
  'knowledge_gradient',
]
from .abstract_loss import abstract_loss

from .myopic_loss import myopic_loss
from .simple_regret import simple_regret
from .negative_ei import negative_ei
from .negative_pi import negative_pi
from .confidence_bound import confidence_bound

from .nonmyopic_loss import nonmyopic_loss
from .entropy_search import entropy_search
from .knowledge_gradient import knowledge_gradient
