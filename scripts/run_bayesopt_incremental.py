#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
Authored: 2018-01-25 
Modified: 2018-11-27
'''

# ---- Dependencies
import os, sys, time
import logging, argparse
import yaml, json
import numpy as np
import numpy.random as npr
import tensorflow as tf
import matplotlib.pyplot as plt

from ast import literal_eval
from pdb import set_trace as bp
from timeit import default_timer
from functools import reduce

# Relative imports
sys.path.insert(0, os.path.join(sys.path[0], '..'))
import src, tasks
from src.third_party.sobol_lib import i4_sobol_generate
from helpers import task_utils, storage, plotting
from helpers.job_manager import job_manager

logger = logging.getLogger(__name__)
# ==============================================
#                                   run_bayesopt
# ==============================================
def parse_dict(raw):
  '''
  Helper method for parsing string-encoded <dict>
  '''
  try:
    pattern = raw.replace('\"','').replace("\\'", "'")
    return literal_eval(pattern)
  except Exception as e:
    raise Exception('Failed to parse string-encoded <dict> {} with exception {}'.format(raw, e))


def get_config():
  parser = argparse.ArgumentParser(description='Run configuration')
  parser.register('type', dict, parse_dict)

  parser.add_argument('--seed', type=int, default=np.random.choice(2**31 - 1), help='Experiment seed')
  parser.add_argument('--float', type=str, default='float64', help='primary floating point dtype')
  parser.add_argument('--save', type=int, default=0, choices=[0, 1], help='store experiment results?')
  parser.add_argument('--result_dir', type=str, default='results', help='path to results directory')
  parser.add_argument('--overwrite', type=int, default=0, choices=[0, 1], help='re-run/overwrite existing experiments')

  parser.add_argument('--task', type=str, default='rfgp', help='ID for optimization task')
  parser.add_argument('--loss', type=str, default='negative_ei', help='loss function ID')
  parser.add_argument('--loss_kwargs', type=dict, default='{}', help='loss function keyword arguments')
  parser.add_argument('--input_dim', type=int, default=1, help='input space dimensionality')
  parser.add_argument('--num_initial', type=int, default=3, help='num. initial observed pairs (X, Y)')

  parser.add_argument('--job_limit', type=int, default=256, help='upper bound on job count')
  parser.add_argument('--parallelism', type=int, default=2, help='maximum degree of parallelism')
  parser.add_argument('--asynchronous', type=int, default=0, choices=[0, 1], help='simulate parallel asynchronous jobs')
  parser.add_argument('--avg_runtime', type=float, default=60, help='avg. runtime of scheduled jobs')
  parser.add_argument('--sleep_tic', type=float, default=1, help='sleep duration')
  parser.add_argument('--risk', type=float, default=0.0, help='conservativeness when suggesting answers (like $\beta$ for LCB)')

  parser.add_argument('--kernel', type=str, default='matern52', help='GP kernel covariance function')
  parser.add_argument('--lenscale', type=float, default=None, help='GP characteristic lenscale')
  parser.add_argument('--amplitude', type=float, default=1.0, help='GP covariance amplitude')
  parser.add_argument('--noise', type=float, default=1e-3, help='variance of Gaussian observation noise')
  parser.add_argument('--anisotropic', type=int, default=1, choices=[0,1], help='use anisotropic lengthscales')
  parser.add_argument('--use_true_prior', type=int, default=0, choices=[0, 1], help='use true model prior')

  parser.add_argument('--num_options', type=int, default=None, help='num. starting candidates coneidered by BO')
  parser.add_argument('--num_to_optimize', type=int, default=None, help='num. candidate choices to optimize')
  parser.add_argument('--time_limit', type=float, default=1, help='runtime budget per suggested job')
  parser.add_argument('--schedule', type=str, default=None, help='schedule for runtime budget')
  parser.add_argument('--use_cpu_time', type=int, default=1, choices=[0, 1], help='use CPU time as primary')

  parser.add_argument('--subroutine', type=str, default='random', help='suffix of suggestion optimization subroutine')
  parser.add_argument('--subroutine_kwargs', type=dict, default='{}', help='string-specified dictionary of configs')
  parser.add_argument('--subroutine_defaults', type=str, default='experiments/defaults/subroutines.yaml', help='path to YAML file of default settings')
  parser.add_argument('--use_greedy', type=int, default=1, choices=[0, 1], help='optimize parallel queries in iterative greedy fahshion.')
  parser.add_argument('--num_fantasies', type=int, default=64, help='num. fantasies used to evaluate iterated acq.')

  parser.add_argument('--num_starts_fmin', type=int, default=1024, help='num. starts for task minimizer')
  parser.add_argument('--num_options_fmin', type=int, default=2**16, help='num. random points used to find starts')

  parser.add_argument('--visualize', action='store_true', help='show problem visualization')
  parser.add_argument('--tf_logging', type=str, default='ERROR', help='print level of TensorFlow logging')
  parser.add_argument('--end2bp', action='store_true', help='enter break point prior to terminating')

  config = parser.parse_args()
  config._cmdline = ' '.join(sys.argv) #store command line for safe keeping

  if (config.time_limit < 0):
    config.time_limit = np.inf

  if (config.lenscale is None):
    config.lenscale = np.sqrt(config.input_dim)/4
  return config



def _suggest_jointly(config, sess, agent, num_suggest, inputs_old, 
  outputs_old, inputs_pend=None, num_options=None, num_to_optimize=None,
  time_limit=None, **kwargs):
  '''
  Query the optimization agent for 'num_suggest' new jobs.
  '''
  if (time_limit is None): time_limit = config.time_limit
  if (num_options is None): num_options = config.num_options
  if (num_to_optimize is None): num_to_optimize = config.num_to_optimize

  tic = agent.timer()
  suggestions, loss = agent.suggest_inputs\
  (
    num_suggest=num_suggest,
    inputs_old=inputs_old,
    outputs_old=outputs_old,
    inputs_pend=inputs_pend,
    sess=sess,
    num_options=num_options,
    num_to_optimize=num_to_optimize,
    time_limit=time_limit * num_suggest,
    **kwargs,
  )
  runtime = agent.timer() - tic
  logger.info('Agent returned {:d} joint suggestion(s) with loss {:.3e} in {:.2e}s'\
        .format(len(suggestions), loss, runtime))
  return suggestions, runtime, loss


def _suggest_greedily(config, sess, agent, num_suggest, inputs_old,
  outputs_old, inputs_pend=None, num_options=None, num_to_optimize=None,
  time_limit=None, num_fantasies=None, **kwargs):
  '''
  Query the optimization agent in iterative greedy fashion 
  for 'num_suggest' new jobs.
  '''
  if (time_limit is None): time_limit = config.time_limit
  if (num_options is None): num_options = config.num_options
  if (num_fantasies is None): num_fantasies = config.num_fantasies
  if (num_to_optimize is None): num_to_optimize = config.num_to_optimize

  def draw_samples(num_samples, X1, X0, Y0, model=agent.model):
    _X1 = agent.get_or_create_ref('inputs/new', X1.ndim)
    _X0 = agent.get_or_create_ref('inputs/old', X0.ndim)
    _Y0 = agent.get_or_create_ref('outputs/old', Y0.ndim)

    samples_op = model.get_or_create_node\
    (
      group='greedy',
      fn=model.draw_samples,
      args=(num_samples, _X1, _X0, _Y0),
      stateful=True,
    )

    return sess.run(samples_op, {_X1:X1, _X0:X0, _Y0:Y0})


  suggestions, losses, runtimes  = [], [], []
  for q in range(num_suggest):
    tic = agent.timer()
    suggestion, loss = agent.suggest_inputs\
    (
      num_suggest=1, #suggest one additional job at a time
      inputs_old=inputs_old,
      outputs_old=outputs_old,
      sess=sess,
      num_options=num_options,
      num_to_optimize=num_to_optimize,
      time_limit=time_limit,
      **kwargs,
    )
    runtimes.append(agent.timer() - tic)
    suggestions.append(np.atleast_1d(np.squeeze(suggestion)))
    losses.append(loss)

    if inputs_old.ndim == 2:
      fantasies = draw_samples(num_fantasies, suggestion, inputs_old, outputs_old)
      inputs_old = np.tile(np.vstack([inputs_old, suggestion])[None, None], [num_fantasies, 1, 1 ,1])
      outputs_old = np.dstack([np.tile(outputs_old[None, None], [num_fantasies, 1, 1, 1]), fantasies[:, None, None]])
    else:
      fantasies = draw_samples(1, suggestion, inputs_old, outputs_old)
      inputs_old = np.dstack([inputs_old, np.tile(suggestion[None, None], [num_fantasies, 1, 1, 1])])

      outputs_old = np.dstack([outputs_old, fantasies])

    # print(inputs_old.shape, outputs_old.shape)
    # inputs_new = np.random.rand(16, 1, 1)
    # losses_op, feed_dict, nodes, extras = agent.appraise_inputs(sess, inputs_new, inputs_old, outputs_old)
    # shape_l, losses = sess.run([tf.shape(losses_op), losses_op], feed_dict)
    # bp()

    # _X1 = agent.get_or_create_ref('inputs/new', 3)
    # _X0 = agent.get_or_create_ref('inputs/old', inputs_old.ndim)
    # _Y0 = agent.get_or_create_ref('outputs/old', outputs_old.ndim)
    # X1 = np.random.rand(8, 1, 1)
    # mu, var = sess.run(agent.model.predict(_X1, _X0, _Y0), {_X1:X1, _X0:inputs_old, _Y0:outputs_old})
    # bp()

    # # Treat suggestions from previous iterations as pending jobs
    # if (inputs_pend is None):
    #   inputs_pend = suggestion
    # else:
    #   inputs_pend = np.vstack([inputs_pend, suggestion])



  runtime = np.sum(runtimes)
  logger.info('Agent returned {:d} greedy suggestion(s) with loss {:.3e} in {:.2e}s'\
        .format(len(suggestions), losses[-1] ,runtime))
  return np.asarray(suggestions), runtime, losses[-1]


def suggest_jobs(config, sess, agent, *args, subroutine=None,
  use_greedy=None, **kwargs):
  '''
  Wrapper for query suggestions methods.
  '''
  if (subroutine is None): subroutine = config.subroutine
  if (use_greedy is None): use_greedy = config.use_greedy

  # Augment initial random search with an optimization subroutine?
  if isinstance(subroutine, str):
    subroutine = getattr(agent, '_optimize_inputs_{}'.format(subroutine))

  kwargs['subroutine'] = subroutine
  if use_greedy:
    return _suggest_greedily(config, sess, agent, *args, **kwargs)
  else:
    return _suggest_jointly(config, sess, agent, *args, **kwargs)



def get_answer(config, sess, agent, task, inputs_old, outputs_old,
  risk=None):
  '''
  Query agent for recommended final answer and record function value.
  '''
  if (risk is None): risk = config.risk
  tic = agent.timer()
  answer, expectation = agent.suggest_answers\
  (
    num_suggest=1,
    inputs_old=inputs_old,
    outputs_old=outputs_old,
    risk=risk,
    sess=sess,
  )
  runtime = agent.timer() - tic

  # Evaluate noise-free ground truth value (not observed by agent)
  ground_truth = np.squeeze(task.numpy(answer, noisy=False))
  logger.info('Agent recommended f(x*={})={:.3e} in {:.2e}s'\
              .format(answer, ground_truth, runtime))

  return (answer, ground_truth, expectation), runtime


def update_state(config, sess, agent, job_man, task, stats, flags, **kwargs):
  '''
  Update the state of the simulation to reflect new information.
  '''
  # Get current simulation data
  simtime = job_man.time
  inputs_old, outputs_old = job_man.get_jobs(exclude_pending=True)

  # Update model/loss functions
  if flags['should_fit'] and not config.use_true_prior:
    stats['hyperparameters'][simtime] = agent.update(sess, inputs_old,
                                                outputs_old, **kwargs)
    flags['should_fit'] = False
  else:
    agent.loss_fn.update(sess, inputs_old, outputs_old,
                      **{'model':agent.model, **kwargs})

  # Get current recommended answer
  stats['answer'][simtime], stats['runtimes']['answer'][simtime] =\
    get_answer(config, sess, agent, task, inputs_old, outputs_old)

  # Report current best seen
  argmin = np.argmin(outputs_old)
  logger.info('Best outcome after {:d} observations: f(x{:d}) = {:.3e}'\
              .format(len(outputs_old), argmin, outputs_old[argmin, 0]))

  # Report log regret of recommended answer
  if hasattr(task, 'minimum'):
    regret = stats['answer'][simtime][1] - task.minimum
    logger.info('Log10 Immediate Regret after {:d} observations: {:.3e}'\
                .format(len(outputs_old), np.log10(regret)))

  return agent, stats, flags


def run_bayesopt(config, sess, agent, job_man, runtime_fn,
  parallelism=None, job_limit=None, task=None, schedule=None,
  sleep_tic=None, visualize=None):
  '''
  Immitate real-world usage case for BO with automated stopping.
  '''
  if (parallelism is None): parallelism = config.parallelism
  if (job_limit is None): job_limit = config.job_limit
  if (sleep_tic is None): sleep_tic = config.sleep_tic
  if (visualize is None): visualize = config.visualize

  # Status indicators for subroutines
  flags =\
  {
    'should_stop' : False,
    'should_fit' : True,
    'should_plot' : True,
  }

  # Recorded statistics
  stats =\
  {
    'answer' : {},
    'hyperparameters' : {},
    'runtimes' : {'fit':{}, 'answer':{}, 'suggest':{}},
  }

  logger.info('Beginning episode; maximum of {:d} new jobs will be run.'\
        .format(job_limit - job_man.num_jobs))

  # Iterate until there are no pending jobs and stopping signal has been tripped
  while job_man.num_pending > 0 or not flags['should_stop']:
    # Get updated state of optimization episode
    _ = update_state(config, sess, agent, job_man, task, stats, flags)

    # Consider scheduling new jobs
    num_workers_free = parallelism - job_man.num_pending
    budget_remaining = job_limit - job_man.num_jobs
    num_suggest = min(num_workers_free, budget_remaining)
    if (num_suggest > 0) and not flags['should_stop']:
      inputs_old, outputs_old = job_man.get_jobs(exclude_pending=True)
      inputs_pend = job_man.pending_inputs

      # Determine runtime limit for job suggestions
      if (schedule is None):
        time_limit = config.time_limit
      else:
        num_pending = 0 if (inputs_pend is None) else len(inputs_pend)
        num_observed = job_man.num_jobs - num_pending
        _parallelism = num_pending + num_suggest
        try:
          query = [inputs_old.shape[-1], num_observed, _parallelism]
          time_limit = reduce(lambda S,s: S[str(s)], [schedule] + query)
        except:
          # [!] temp. hack: last iteration may consist of fewer suggestions
          query = [inputs_old.shape[-1], num_observed, parallelism]
          time_limit = reduce(lambda S,s: S[str(s)], [schedule] + query)
        time_limit /= num_suggest #express per job

      # Suggested additional jobs for free workers
      suggestions, rt, loss = suggest_jobs(config, sess, agent,
                                          num_suggest, inputs_old,
                                          outputs_old, inputs_pend,
                                          time_limit=time_limit)
      stats['runtimes']['suggest'][job_man.time] = rt

      # Visualize current state
      if (visualize and flags['should_plot']):
        logger.info('Visualizing episode state')
        _, axes = plotting.plot_state(agent, sess, inputs_old, outputs_old,
                                      job_man.pending_inputs,
                                      answer=stats['answer'][job_man.time][0],
                                      query=suggestions,
                                      task=task)
      plt.show()
      flags['should_plot'] = False

      # Schedule suggested jobs
      runtimes = runtime_fn(suggestions) #generate job runtimes
      job_man.submit_jobs(suggestions, runtimes)
      flags['should_plot'] = True #new pending job(s)

    # Determine how far ahead in time to advance the episode
    can_request = not flags['should_stop']\
                  and job_man.num_pending < num_suggest\
                  and job_man.num_jobs < job_limit 

    if can_request: #can a request between now and the next job comlpetion?
      d_time = sleep_tic #advance by a single tic
    elif job_man.num_pending > 0:
      futures = job_man.get_futures() #when will pending jobs finish?
      num_tics = np.divide(min(futures.values()) - job_man.time, sleep_tic)
      d_time = np.ceil(num_tics) * sleep_tic #advance to next event time
    else:
      d_time = 0 #we're done, there are no pending jobs and we should stop

    # Go forward in time, collecting results for newly completed jobs
    if (d_time > 0):
      new_results = job_man.advance(d_time)
      if (new_results is not None): flags['should_fit'] = True

    # Has the job budget been exhausted?
    if job_man.num_jobs == job_limit:
      flags['should_stop'] = True

  # Get final state of optimization episode
  _ = update_state(config, sess, agent, job_man, task, stats, flags)
  return stats


def get_subroutine_configs(config, defaults=None, optim_id=None):
  if (optim_id is None): optim_id = config.subroutine
  if (optim_id is None): return dict() #short-circuit
  if (defaults is None): defaults = config.subroutine_defaults
  configs = config.subroutine_kwargs.copy()

  # Load default settings
  if isinstance(defaults, str): #path to YAML file
    with open(defaults, 'rb') as file: all_defaults = yaml.load(file)

  defaults = all_defaults.get(optim_id, None)
  if (defaults is None): return configs
  defaults = defaults.get('greedy' if config.use_greedy else 'joint', dict())

  # Configs should be nested; see <bayesian_optimization.__init__>
  if configs.get(optim_id, None) is None:
    configs[optim_id] = defaults
  else:
    configs[optim_id] = {**defaults, **configs[optim_id]}

  # [!] Ugly hacks; allows us to specify default for other settings
  for key in ('num_options','num_to_optimize'):
    val = configs[optim_id].pop(key, None)
    if (val is None): continue
    if getattr(config, key, None) is None:
      logger.info('Setting config.{:s} to {}'.format(key, val))
      setattr(config, key, val)

  return configs


def run_experiment(config):
  # Can update the config, so should go first
  subroutine_configs = get_subroutine_configs(config)

  if config.save and not config.overwrite: #check if experiment already exists
    exclude_ids =\
    [# These settings don't influence outcomes
      'save', 'overwrite', 'visualize', 'end2bp',
      'tf_logging', '_cmdline',
    ]
    query = dict([kv for kv in config.__dict__.items() if kv[0] not in exclude_ids])
    match = storage.find_experiments(config.result_dir, query, find_all=False)
    if len(match): #found entries with matching metadata
      logger.info('Found existing experiment(s) with '
                  'matching metadata; terminating run.')
      return None

  np.set_printoptions(precision=3)
  with tf.Session() as sess:
    # Experiment seed
    tf.set_random_seed(config.seed)
    rng = npr.RandomState(config.seed)
    logger.info('pRNG seed: {:d}'.format(config.seed))

    # Initialize task
    if config.task.lower() in ['rf', 'rfgp', 'random_fourier']:
      task = task_utils.build_rfgp_task(config, sess, config.input_dim, rng=rng)
    else:
      constructor = getattr(tasks, config.task)
      task = constructor(config.input_dim, noise=config.noise,
                          dtype=config.float, seed=config.seed)

    # Estimate true task minimium (for evaluating regret)
    if not hasattr(task, 'minimum') and config.num_starts_fmin > 0:
      _, task.minimum = task_utils.minimize_task\
      (
        config,
        sess,
        task,
        config.num_starts_fmin,
        rng=rng,
      )

    if hasattr(task, 'minimum'):
      logger.info('Task minimum: {:.3e}'.format(task.minimum))

    if (config.visualize and task.input_dim > 1):
      logger.warning('Visualization(s) not yet implemented for >1D')

    # Initialize surrogate model
    model_kwargs =\
    {
      'kernel_id':config.kernel,
      'dtype':config.float,
      'name':'gp',
      'seed':config.seed,
    }
    if config.use_true_prior: #use true hyperparameters
      assert config.task.lower() in ['rfgp']
      logger.info('Using true GP prior.')
      model_kwargs['log_lenscales'] = np.log(config.lenscale).astype(config.float)
      model_kwargs['log_amplitude'] = np.log(config.amplitude).astype(config.float)
      if (config.noise is None or config.noise == 0):
        model_kwargs['log_noise'] = None
      else:
        model_kwargs['log_noise'] = np.log(config.noise).astype(config.float)
      model_kwargs['mean'] = task_utils.estimate_task_mean(config, task, rng=rng)

    if config.anisotropic:
      log_lenscales = np.asarray(model_kwargs.get('log_lenscales', 0.0))
      if log_lenscales.size == 1:
        model_kwargs['log_lenscales'] = np.full([task.input_dim], log_lenscales)

    model = src.models.gaussian_process(**model_kwargs)
    model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, model.name)
    sess.run(tf.variables_initializer(model_vars))

    # Initialize loss function
    loss_kwargs = config.loss_kwargs.copy()
    loss_kwargs.setdefault('seed', config.seed)
    if config.visualize: loss_kwargs['cache_rvs'] = True
    if config.loss in ['negative_pi', 'entropy_search']\
    and config.subroutine in [None, 'sgd', 'scipy']:
      loss_kwargs.setdefault('temperature', 0.01)
    loss_fn = getattr(src.losses, config.loss)(**loss_kwargs)

    # Create search agent
    timer = time.process_time if config.use_cpu_time else default_timer
    agent = src.agents.bayesian_optimization\
    (
      loss_fn=loss_fn,
      model=model,
      name='bayesopt',
      seed=config.seed,
      dtype=task.dtype,
      timer=timer,
      configs=subroutine_configs,
    )

    # Create initial i/o sequence for optimization episode
    initial_inputs = i4_sobol_generate\
    (
      task.input_dim,
      config.num_initial,
      skip=np.mod(config.seed, 99901), #a large prime...
    ).T
    initial_outputs = task.numpy(initial_inputs)

    # Build agent subroutine(s)
    if not config.use_true_prior:
      param_vals = agent.update(sess, initial_inputs, initial_outputs)
    else:
      loss_fn.update(sess, initial_inputs, initial_outputs, model=agent.model)
    _ = suggest_jobs(config, sess, agent, config.parallelism, initial_inputs,
                    initial_outputs, time_limit=5)

    num_suggest_final = (config.job_limit - config.num_initial) % config.parallelism
    if num_suggest_final:
      _ = suggest_jobs(config, sess, agent, num_suggest_final, initial_inputs,
                      initial_outputs, time_limit=5)

    # Create a job manager
    job_man = job_manager(task=task.numpy)
    job_man.add_results(initial_inputs,
                        initial_outputs,
                        ground_truths=task.numpy(initial_inputs, noisy=False))

    # Dictates runtime (i.e. cost) of individual experiments
    def runtime_generator(inputs, avg_runtime=config.avg_runtime):
      num_runtimes = len(inputs)
      if (config.asynchronous): #schedule jobs to finish at different times
        runtimes = rng.exponential(avg_runtime, num_runtimes)
      else: #schedule jobs to finish simultaneously
        runtimes = np.full([num_runtimes], avg_runtime, dtype='float')
      return runtimes

    if (config.schedule is None):
      schedule = None
    else:
      with open(config.schedule, 'r') as file:
        schedule = json.load(file)

    # Simluate optimization episode
    stats = run_bayesopt(config, sess, agent, job_man,
                            runtime_generator, task=task,
                            schedule=schedule)

  # Pertinent experiment data
  experiment =\
  {
    **stats,
    'inputs' : job_man.inputs,
    'outputs' : job_man.outputs,
    'ground_truths' : job_man.ground_truths,
    'starts' : job_man.starts,
    'costs' : job_man.runtimes,
    'f_min' : getattr(task, 'minimum', None),
  }

  return experiment


if __name__ == '__main__':
  tic_wall, tic_cpu = default_timer(), time.process_time()
  config = get_config()

  # Global logging configuration
  tf.logging.set_verbosity(config.tf_logging.upper())
  logger = logging.getLogger('run_experiment')
  logging.basicConfig\
  (
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    format='%(asctime)-4s %(levelname)s:%(name)s:%(message)s'
  )

  experiment = run_experiment(config)
  if config.save and experiment is not None: #store results?
    storage.save_experiment(config, experiment, config.result_dir)

  toc_wall, toc_cpu = default_timer(), time.process_time()
  logger.info('Runtime: {:.6e}s (wall), {:.6e}s (CPU)'\
        .format(toc_wall - tic_wall, toc_cpu - tic_cpu))

  if config.end2bp: bp()

# ==============================================
#                            Developers' Section
# ==============================================
# ------------ Scrap work goes here ------------
'''
'''