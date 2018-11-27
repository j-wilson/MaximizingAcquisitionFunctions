#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
Authored: 2018-02-06 
Modified: 2018-10-26
'''

# ---- Dependencies
import os, sys, time
import numpy as np
import hashlib
import logging, argparse
import json, pathlib
from warnings import warn
from datetime import datetime
from pdb import set_trace as bp

__all__ =\
[
  'to_json',
  'check_git',
  'save_experiment',
  'find_experiments',
  'load_experiments',
]

logger = logging.getLogger(__name__)
# ==============================================
#                                        storage
# ==============================================
def to_json(entry):
  # Convert experiment data for JSON format
  if isinstance(entry, dict):
    new_entry = entry.__class__() #maintain subclass
    for key, val in entry.items():
      key = str(key)# json only allows string-valued keys
      new_entry[key] = to_json(val)
    return new_entry
  elif isinstance(entry, (list, tuple, set)):
    return [to_json(term) for term in entry]
  elif isinstance(entry, np.ndarray):
    return entry.tolist()
  else:
    return entry


def check_git():
  try: #attempt to store git info
    import git # pylint: disable=g-import-not-at-top
    repo = git.Repo(search_parent_directories=True)
    return \
    {
      'hash' : repo.head.object.hexsha,
      'dirty' : repo.is_dirty(),
      'branch' : repo.active_branch.name,
    }
  except:
    return {}


def save_experiment(meta, results, root=None, experiment_id=None):
  if hasattr(meta, '__dict__'): meta = meta.__dict__ #strip off excess API
  if (root is None): root = meta['result_dir'] #directory for saved files

  # Store date/time and Git status
  now = datetime.now()
  date_keys = ('day', 'month', 'year')
  time_keys = ('hour', 'minute', 'second')
  meta.setdefault('date', {key:getattr(now, key) for key in date_keys})
  meta.setdefault('time', {key:getattr(now, key) for key in time_keys})
  meta.setdefault('git', check_git())

  # Get unique experiment identifier
  if (experiment_id is None):
    encoding = json.dumps(meta, sort_keys=True).encode('utf-8')
    experiment_id = hashlib.md5(encoding).hexdigest()

  # Convert experiment data for JSON format
  data = to_json(results)

  # Ensure existance of parent directory
  pathlib.Path(root).mkdir(parents=True, exist_ok=True)

  # Create separate meta and data files for experiment
  for (file_id, content) in zip(['meta', 'data'], [meta, data]):
    filename = '{:s}.{:s}'.format(experiment_id, file_id)
    filepath = os.path.join(root, filename)
    with open(filepath, 'w') as file: json.dump(content, file)

  return experiment_id


def evaluate_query(meta, query):
  '''
  Determine whether an experiment's meta
  data matches a given query.
  '''
  if (query is None):
    return True
  elif isinstance(query, dict):
    for key, term in query.items():
      field = meta.get(key, None)
      if not evaluate_query(field, term):
        return False
    return True
  elif callable(query):
    return query(meta)
  else:
    return meta.__eq__(query)


def find_experiments(dirname, query=None, find_all=True):
  '''
  Find experiment IDs. If a query is specified, only
  return IDs for matching experiments.
  '''
  if not os.path.exists(dirname): return []

  handles = os.listdir(dirname)
  experiment_ids = []
  for handle in handles:
    full_path = os.path.join(dirname, handle)
    if os.path.isdir(full_path):
      child_res = find_experiments(full_path, query, find_all)
      experiment_ids.extend(child_res)
    else:
      experiment_id, ext = os.path.splitext(handle)
      if ext != '.meta':        
        # Only need to check meta data files
        continue 
      elif (query is None):
        # No query was specified, get all IDs
        experiment_ids.append(experiment_id)
      else:
        # Only find matching IDs
        with open(full_path, 'r') as file:
          meta = json.load(file)
          if evaluate_query(meta, query):
            experiment_ids.append(experiment_id)

    # Only find ID of first matching experiment
    if not find_all and len(experiment_ids): break
  return experiment_ids


def load_experiments(dirname, query=None, find_all=True):
  '''
  Collect experiments (both meta-data and result data).
  If a query is specified, only return matching experiments.
  '''
  if not os.path.exists(dirname): return []

  handles = os.listdir(dirname)
  experiments = dict()
  for handle in handles:
    full_path = os.path.join(dirname, handle)
    if os.path.isdir(full_path):
      child_res = load_experiments(full_path, query, find_all)
      experiments.update(child_res)
    else:
      experiment_id, ext = os.path.splitext(handle)
      if ext != '.meta': continue

      with open(full_path, 'r') as meta_file:
        meta = json.load(meta_file)

      if evaluate_query(meta, query):
        filepath_data = os.path.join(dirname, experiment_id + '.data')
        if not os.path.isfile(filepath_data):
          warn("Missing data file for experiment with ID: " + experiment_id)
          continue

        try:
          with open(filepath_data, 'r') as data_file:
            experiment = json.load(data_file)
          experiments[experiment_id] = {'meta':meta, **experiment}
        except Exception as e:
          pass
        #   print(e)

    # Only collect first matching experiment
    if not find_all and len(experiments): break

  return experiments
