#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
Utility methods for interfacing with the user
and/or for development purposes.

Authored: 2017-03-12
Modified: 2018-01-01
'''

# -------- Dependencies
import os, sys
import json
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

__all__ =\
[
  'session_profiler',
  'uninitialized_variables',
  'progress_bar',
  'assert_close',
]

# ==============================================
#                                       user_ops
# ==============================================
class session_profiler(object):
  def __init__(self, session=None, options=None, run_metadata=None,
    profile=None, save_dir=None):
    if (options is None):
      options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    self.options = options

    if (run_metadata is None):
      run_metadata = tf.RunMetadata()
    self.run_metadata = run_metadata

    self.profile = profile
    self.save_dir = save_dir
    self.session = session

  def run(self, *args, session=None, options=None, run_metadata=None, 
    update=True, **kwargs):
    if (session is None): session = self.session
    if (options is None): options = self.options
    if (run_metadata is None): run_metadata = self.run_metadata

    fetches = session.run(*args, options=options, 
                  run_metadata=run_metadata, **kwargs)

    if update: self.update(run_metadata)
    return fetches

  def update(self, run_metadata=None):
    if (run_metadata is None):
      run_metadata = self.run_metadata

    # Get run statistics as timeline and convert to dict
    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()    
    trace_dict = json.loads(chrome_trace)

    # For first run store full trace
    if (self.profile is None):
      self.profile = trace_dict
    else:
      # Event runtimes are prefixed 'ts'
      runtimes = filter(lambda event: 'ts' in event, trace_dict['traceEvents'])
      self.profile['traceEvents'].extend(runtimes)

  def save(self, filename):
    if (self.save_dir is not None):
      filename = '/'.join([self.save_dir, filename])

    dirname = os.path.dirname(filename)
    pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
    with open(filename, 'w') as file:
        json.dump(self.profile, file)


def uninitialized_variables(sess, var_list=None):
  '''
  Return a list of uninitialized variables for a given session
  '''
  if (var_list is None): var_list = tf.global_variables()
  statuses = sess.run([tf.is_variable_initialized(var) for var in var_list])
  return [var for (var, status) in zip(var_list, statuses) if not status]


def progress_bar(progress, duration=None, title=None,
  char='#', precision=1, bar_length=50):
  assert isinstance(progress, float), "Progress value must be <float>-typed"
  if (title is None): title = 'Progress'
  num_chars = int(round(bar_length * progress))
  bar = num_chars * char + ' '*(bar_length - num_chars*len(char))
  msg = '\r{:s}: [{:s}] {:.{p}f}%'.format(title, bar, 100*progress, p=precision)
  if (duration is not None):
    eta = (1 - progress)/progress * duration
    msg += ', ETA: {:.2e}s'.format(eta)
  if progress >= 1.0: msg += '\r\n'
  sys.stdout.write(msg)
  sys.stdout.flush()


def assert_close(A, B, tol=0.0, name=None):
  with tf.name_scope('assert_close') as scope:
    max_abs_diff = tf.reduce_max(tf.abs(A - B))
    _tol = tf.constant(tol, max_abs_diff.dtype)
    return tf.assert_less(max_abs_diff, _tol, name=name)


# ==============================================
#                            Developers' Section
# ==============================================
# ------------ Scrap work goes here ------------
'''
'''