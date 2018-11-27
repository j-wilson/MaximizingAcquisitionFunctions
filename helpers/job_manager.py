#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
Authored: 2018-01-26 
Modified: 2018-04-17
'''

# ---- Dependencies
import logging
import numpy as np
from pdb import set_trace as bp

logger = logging.getLogger(__name__)
# ==============================================
#                                   job_mananger
# ==============================================
class job_manager(object):
  def __init__(self, task=None, time=None, inputs=None, outputs=None,
    starts=None, runtimes=None, logger=logger):
    if (inputs is None): inputs = dict()
    if (outputs is None): outputs = dict()
    if (runtimes is None): runtimes= dict()
    if (starts is None): starts = dict()

    if (time is None):
      time = np.zeros([1], dtype='float')
    self._time = time

    self.task = task
    self.inputs = inputs
    self.outputs = outputs
    self.runtimes = runtimes
    self.starts = starts
    self.logger = logger

    # [!] temp. hack: save noise-free outputs
    self.ground_truths = dict()


  def submit_jobs(self, inputs, runtimes, starts=None):
    if (starts is None): starts = np.full([len(inputs)], self.time)
    num_existing = len(self.inputs)

    zipper = zip(inputs, runtimes, starts)
    for relative_id, (input, runtime, start) in enumerate(zipper):
      job_id = num_existing + relative_id
      self.inputs[job_id] = input
      self.runtimes[job_id] = runtime
      self.starts[job_id] = start
      logger.info('Submitted x{:d}={} at t={:.2e}s'.format(job_id, input, start))


  def get_status(self, job_ids=None, time=None):
    if (time is None): time = self.time
    if (job_ids is None): job_ids = sorted(self.inputs.keys())
    elif isinstance(job_ids, int): job_ids = job_ids, #cast to tuple
    return {id:(self.starts[id] + self.runtimes[id] <= time) for id in job_ids}


  def get_jobs(self, job_ids=None, exclude_pending=False):
    '''
    Return I/O pairs for existing jobs as <numpy.ndarray>
    '''
    if (job_ids is None): job_ids = sorted(self.inputs.keys())
    elif isinstance(job_ids, int): job_ids = job_ids, #cast to tuple

    inputs, outputs = [], []
    for job_id in job_ids:
      if exclude_pending:
        output = self.outputs.get(job_id, None)
        if (output is not None):
          outputs.append(output)
          inputs.append(self.inputs[job_id])
      else:
        inputs.append(self.inputs[job_id])
        outputs.append(self.outputs.get(job_id, [np.nan]))
    return np.vstack(inputs), np.vstack(outputs)

  
  def get_futures(self, job_ids=None, time=None):
    '''
    Return a dictionary of times at which pending jobs will finish.
    '''
    if (time is None): time = self.time
    if (job_ids is None): job_ids = self.inputs.keys()
    elif isinstance(job_ids, int): job_ids = job_ids, #cast to tuple

    futures = {}
    for job_id in job_ids:
      deadline = self.starts[job_id] + self.runtimes[job_id]
      if deadline > time: futures[job_id] = deadline
    return futures


  def advance(self, increment, *args, **kwargs):
    '''
    Go forward in time, collecting results for any newly completed jobs.
    '''
    old_statuses = self.get_status()
    self.time += increment #go forward
    new_statuses = self.get_status()

    newly_completed = []
    for job_id, new_status in new_statuses.items():
      if new_status and not old_statuses[job_id]:
        newly_completed.append(job_id)

    if len(newly_completed):
      return self.dispatch(newly_completed, *args, **kwargs)
    return None


  def dispatch(self, job_ids, task=None, **kwargs):
    '''
    Evaluate specified jobs.
    '''
    if (task is None): task = self.task
    if isinstance(job_ids, int): job_ids = job_ids, #cast to tuple
    inputs = np.vstack([self.inputs[job_id] for job_id in job_ids])
    outputs = task(inputs, **kwargs)
    ground_truths = task(inputs, **{**kwargs, 'noisy':False})

    num_results = len(outputs)
    results_msg =\
    [
      '{:d} job(s) just finished; {:d} job(s) still pending'\
        .format(num_results, self.num_pending) ,
    ]
    zipper = zip(job_ids, inputs, outputs, ground_truths)
    result_template = '> f(x{:d}={}) = {:.3e}'
    for job_id, input, output, truth in zipper:
      self.outputs[job_id] = output #store results
      self.ground_truths[job_id] = truth

      results_msg.append(result_template.format(job_id, input, output[0]))
    self.logger.info('\n'.join(results_msg + ['']))
    return inputs, outputs


  def add_results(self, inputs, outputs, starts=None, runtimes=None,
    ground_truths=None):
    if (starts is None): starts = np.zeros([len(inputs)])
    if (runtimes is None): runtimes = np.zeros([len(inputs)])

    num_existing = len(self.inputs)
    zipper = zip(inputs, outputs, runtimes, starts)
    for k, (input, output, start, runtime) in enumerate(zipper):
      job_id = num_existing + k
      self.inputs[job_id] = input
      self.outputs[job_id] = output
      self.starts[job_id] = start
      self.runtimes[job_id] = runtime
      if (ground_truths is not None):
        self.ground_truths[job_id] = ground_truths[k]

  @property
  def completed_ids(self):
    statuses = self.get_status()
    id_list = []
    for job_id, status in statuses.items():
      if status: id_list.append(job_id)
    return id_list


  @property
  def pending_ids(self):
    statuses = self.get_status()
    id_list = []
    for job_id, status in statuses.items():
      if not status: id_list.append(job_id)
    if len(id_list): return id_list
    return None


  @property
  def pending_inputs(self):
    pending_ids = self.pending_ids
    if (pending_ids is None): return None
    return np.vstack([self.inputs[id] for id in pending_ids])


  @property
  def num_jobs(self):
    return len(self.inputs)


  @property
  def num_completed(self):
    statuses = self.get_status()
    return sum(statuses.values())


  @property
  def num_pending(self):
    statuses = self.get_status()
    return len(statuses) - sum(statuses.values())


  @property
  def time(self): 
    return self._time[0]


  @time.setter
  def time(self, new_time):
    self._time[:] = new_time

