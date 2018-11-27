#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
Authored: 2018-01-27 
Modified: 2018-06-04
'''

# ---- Dependencies
import numpy as np
import numpy.random as npr
import scipy.stats as sps
import tensorflow as tf
import matplotlib.pyplot as plt
from pdb import set_trace as bp

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=11)

# ==============================================
#                                       plotting
# ==============================================
def plot_posterior(ax, x, mu, var, samples=None, color=None, alpha=1.0,
  zorder=0, line_kwargs=None, fill_kwargs=None, samples_kwargs=None):

  line_kwargs =\
  {
    'alpha': 0.9 * alpha,\
    'linewidth' : 1.5,
    'color' : color,
    'zorder' : zorder + 1,
    **(line_kwargs or {})
  }

  fill_kwargs =\
  {
    'alpha' : 0.2 * alpha,
    'color' : color,
    'zorder' : zorder,
    **(fill_kwargs or {}),
  }

  samples_kwargs =\
  {
    'alpha' : 0.66 * alpha,
    'linewidth' : 0.25,
    'color' : color,
    'zorder' : zorder + 1,
    **(samples_kwargs or {}),
  }

  # Visualize posteriors and sample paths
  if (x.ndim > 1): x = np.squeeze(x)
  if (mu.ndim > 1): mu = np.squeeze(mu)
  if (var.ndim > 1): var = np.squeeze(var)

  bound = 2 * np.sqrt(var)
  fill = ax.fill_between(x, mu + bound, mu - bound, **fill_kwargs)
  line = ax.plot(x, mu, **line_kwargs)
  if (samples is not None):
    ax.plot(x, samples, **samples_kwargs)

  return line, fill


def _plot_acquisition_1d(ax, x, values, query_and_val=None,
  color='tab:green', zorder=1, line_kwargs=None, fill_kwargs=None,
  query_kwargs=None, rescale=False, show_query=True, eps=None):
  if (eps is None): eps = np.finfo(values.dtype).eps

  line_kwargs =\
  {
    'alpha': 0.9,
    'linewidth' : 1.5,
    'color' : color,
    'zorder' : zorder + 1,
    **(line_kwargs or {})
  }

  fill_kwargs =\
  {
    'alpha' : 0.2,
    'color' : color,
    'zorder' : zorder,
    **(fill_kwargs or {}),
  }

  query_kwargs =\
  {
    'marker':'X',
    's':64,
    'color' : 'tab:pink',
    'edgecolor' : 'k',
    'linewidth' : 0.5,
    'label' : 'Query',
    'zorder' : zorder + 2,
    **(query_kwargs or {}),
  }

  # Visualize posteriors and sample paths
  if (x.ndim > 1): x = np.squeeze(x)
  if (values.ndim > 1): values = np.squeeze(values)

  if (show_query and query_and_val is None):
    argmax = np.argmax(values)
    query_and_val = (x[argmax], values[argmax])

  acq_min, acq_max = np.min(values), np.max(values)
  if show_query:
    _, val = query_and_val
    acq_min = min(acq_min, val)
    acq_max = max(acq_max, val)

  if rescale:
    denom = max(acq_max - acq_min, eps)
    if show_query:
      query, val = query_and_val
      query_and_val = (query, np.divide(val - acq_min, denom))
    values = np.divide(values - acq_min, denom)

  line = ax.plot(x, values, **line_kwargs)
  fill = ax.fill_between(x, values, acq_min, **fill_kwargs)
  if show_query: ax.scatter(*query_and_val, **query_kwargs)

  acq_min, acq_max = np.min(values), np.max(values)
  ax.set_ylim([acq_min, acq_max + 0.1*(acq_max - acq_min)])

  return line, fill


def _plot_acquisition_2d(ax, x, values, query_and_val=None, zorder=1,
  surface_kwargs=None, query_kwargs=None, rescale=True, show_query=True,
  eps=None, cmap=None):
  if (eps is None): eps = np.finfo(values.dtype).eps
  if (cmap is None): cmap = plt.get_cmap('Blues')

  query_kwargs =\
  {
    'marker':'X',
    's':64,
    'color' : 'tab:pink',
    'edgecolor' : 'k',
    'linewidth' : 0.5,
    'label' : 'Query',
    'zorder' : zorder + 2,
    **(query_kwargs or {}),
  }

  if (show_query and query_and_val is None):
    argmax = np.argmax(values)
    query_and_val = (x[argmax], values[argmax])

  if rescale:
    acq_min, acq_max = np.min(values), np.max(values)
    if show_query:
      query, val = query_and_val
      acq_min = min(acq_min, val)
      acq_max = max(acq_max, val)
      denom = max(acq_max - acq_min, eps)
      query_and_val = (query, np.divide(val - acq_min, denom))
    else:
      denom = max(acq_max - acq_min, eps)
    values = np.divide(values - acq_min, denom)

  width = int(len(values)**0.5)
  surface = ax.pcolormesh\
  (
    np.reshape(x[:,0], [width, width]),
    np.reshape(x[:,1], [width, width]),
    np.reshape(values, [width, width]),
    cmap=cmap,
    # Get rid of annoying mesh lines
    edgecolor='face',
    rasterized=True,
    linewidth=0.0,
  )

  ax.plot(ax.get_xlim(), ax.get_ylim(), '--k',  linewidth=1.0)

  if show_query:
    query = np.squeeze(query_and_val[0])
    ax.scatter(query[0], query[1], **query_kwargs)
    ax.scatter(query[1], query[0], **query_kwargs)


  return surface


def plot_acquisition(ax, x, *args, **kwargs):
  acq_dim = x.shape[-2] if x.ndim > 2 else 1
  if (acq_dim == 1):
    _plot_acquisition_1d(ax, x, *args, **kwargs)
  elif (acq_dim == 2):
    _plot_acquisition_2d(ax, x, *args, **kwargs)
  else:
    raise Exception('Plotting unavailable for '\
                    '{}-dimensional acqusition surfaces'.format(acq_dim))



def plot_rfgp(ax, sess, agent, inputs_old, outputs_old, rfgp=None,
  max_shown=64, rng=None):
  if (rng is None): rng = npr
  if (rfgp is None): rfgp = agent.extras['rfgp']

  dtype = rfgp.dtype
  inputs_src = np.linspace(0, 1, 1024, dtype=dtype)[:, None]
  inputs_ref = agent.get_or_create_ref('inputs/new', 2, dtype)

  # This is ugly because it's not obvious that the RFGP may be
  # using the GP's marginal statistics for output correction...
  inputs_old_ref = agent.get_or_create_ref('inputs/old', 2, dtype)
  outputs_old_ref = agent.get_or_create_ref('outputs/old', 2, dtype)

  # Evaluate random Fourier posterior
  feed_dict =\
  {
    inputs_ref : inputs_src,
    inputs_old_ref : inputs_old,
    outputs_old_ref : outputs_old,
  }
  rfgp_op = agent.get_or_create_node('random_fourier', rfgp, (inputs_ref,))
  sample_paths = sess.run(rfgp_op, feed_dict)
  mu_rf = np.mean(sample_paths, axis=-1, keepdims=True)
  cov_rf = np.cov(sample_paths)
  var_rf = np.diag(cov_rf)

  num_samples = sample_paths.shape[-1]
  if (num_samples < max_shown): #don't show more than 64 sample paths
    samples_shown = sample_paths
  else:
    indices = rng.choice(num_samples, max_shown, replace=False)
    samples_shown = sample_paths[:, indices]
  plot_posterior(ax, inputs_src, mu_rf, var_rf, samples_shown, color='tab:orange')


def plot_state(agent, sess, inputs_old, outputs_old, inputs_pend=None,
  answer=None, query=None, task=None, loss_fn=None, model=None,
  discretization=None, parallelism=None, show_acquisition=True,
  show_noise=False, fig=None):
  '''
  Visualize current state of optimization episode.
  '''
  if (loss_fn is None): loss_fn = agent.loss_fn
  if (model is None): model = agent.model

  if (query is not None):
    query = np.atleast_2d(query)
    if (parallelism is None): parallelism = len(query)
    if (inputs_pend is not None): parallelism += len(inputs_pend)
  elif (parallelism is None): parallelism = 1

  X1_ref = agent.get_or_create_ref('inputs/new', 3)
  X0_ref = agent.get_or_create_ref('inputs/old', 2)
  Y0_ref = agent.get_or_create_ref('outputs/old', 2)
  feed_dict = {X0_ref:inputs_old, Y0_ref:outputs_old}

  def eval_posterior(X1_src, **kwargs):
    while X1_src.ndim < 3: X1_src = np.expand_dims(X1_src, axis=0)
    mean_op, var_op = model.get_or_create_node\
    (
      group='predict',
      fn=model.predict,
      args=(X1_ref, X0_ref, Y0_ref),
      kwargs=kwargs,
    )
    feed = {**feed_dict, X1_ref:X1_src}
    return map(np.squeeze, sess.run((mean_op, var_op), feed))

  def eval_losses(X1_src, parallelism=parallelism, feed_dict=feed_dict,
    **kwargs):
    while X1_src.ndim < 3: X1_src = np.expand_dims(X1_src, axis=1)
    losses_op = loss_fn(X1_ref, X0_ref, Y0_ref, model=model,
                          parallelism=parallelism, **kwargs)
    if isinstance(losses_op, tuple): losses_op, *extras = losses_op
    return np.squeeze(sess.run(losses_op, {**feed_dict, X1_ref:X1_src}))

  if (discretization is None):
    discretization = np.linspace(0, 1, 1024)[:,None]

  if show_acquisition:
    if parallelism == 2:
      if (fig is None): fig = plt.figure(figsize=(12, 6))
      axes =\
      [
        plt.subplot2grid((1, 2), (0, 0)),
        plt.subplot2grid((1, 2), (0, 1)),
      ]
      for k, ax in enumerate(axes):
        ax.set_xlabel('Input space $\mathcal{X}$', fontsize=11)
        if k == 0:
          ax.set_title('Output space $\mathcal{Y}$', fontsize=11)
        elif k == 1:
          ax.set_title('Acquisition value', fontsize=11)
    else:
      if (fig is None): fig = plt.figure(figsize=(12, 8))
      axes =\
      [
        plt.subplot2grid((2, 1), (0, 0)),
        plt.subplot2grid((2, 1), (1, 0)),
      ]      
      axes[0].set_ylabel('Output space $\mathcal{Y}$')
      axes[1].set_ylabel('Acquisition value')
      axes[1].set_xlabel('Input space $\mathcal{X}$', fontsize=11)
  else:
    if (fig is None): fig = plt.figure(figsize=(12, 8))
    axes = [plt.subplot2grid((1, 1), (0, 0))]
    axes[0].set_xlabel('Input space $\mathcal{X}$', fontsize=11)
    axes[0].set_ylabel('Output space $\mathcal{Y}$', fontsize=11)

  if (task is not None):
    fvals = task.numpy(discretization)
    axes[0].plot(discretization, fvals, '--k',
            linewidth=1.5, alpha=0.8, zorder=4)

  axes[0].scatter(inputs_old, outputs_old, label='Observations',
                  marker='.', s=96, color='tab:orange', alpha=0.8,
                  edgecolor='k', linewidth=0.5, zorder=11)

  mean, var = eval_posterior(discretization)
  _ = plot_posterior(axes[0], discretization, mean, var, zorder=2)
  if show_noise:
    var_nz = var + sess.run(model.noise)
    bound = 2 * np.sqrt(var_nz)
    axes[0].fill_between(discretization[:,0], mean + bound, mean - bound,
                          zorder=1, color='tab:purple', alpha=0.2)

  if (inputs_pend is not None):
    mean_pend, var_pend = eval_posterior(inputs_pend)
    axes[0].scatter(inputs_pend, mean_pend, marker='D',
                    s=48, edgecolor='k', c='tab:purple',
                    alpha=0.8, linewidth=0.5, zorder=9)

  if (query is not None):
    mean_q, var_q = eval_posterior(query)
    axes[0].scatter(query, mean_q, marker='X',
                    s=64, edgecolor='k', c='tab:pink',
                    alpha=0.8, linewidth=0.5, zorder=10)

  if (answer is not None):
    mean_ans, var_ans = eval_posterior(answer)
    axes[0].scatter(answer, mean_ans, label='Answer', marker='*', alpha=0.8,
              c='tab:green', edgecolor='k', s=128, linewidth=0.5, zorder=10)

  if hasattr(loss_fn, 'mapping'):
    sample_paths = sess.run(loss_fn.mapping(X0_ref), {X0_ref:discretization})
    mu_rf = np.mean(sample_paths, axis=1)
    var_rf = np.var(sample_paths, axis=1)
    plot_posterior(axes[0], discretization, mu_rf, var_rf,
                  color='tab:orange', alpha=0.75, zorder=5,
                  samples=sample_paths[:, :64],
                  samples_kwargs={'alpha':0.5})

  if show_acquisition:
    if parallelism == 2:
      grid = np.hstack(map(lambda x: np.ravel(x)[:,None,None],
                  np.meshgrid(*(2*[np.linspace(0, 1, 256)]))))
    else:
      grid = discretization

    losses = eval_losses(grid, parallelism=parallelism)
    if query is None:
      query_and_val = None
    else:
      if (inputs_pend is None):
        query_and_val = (query, -eval_losses(query[None], parallelism=parallelism))
      else:
        X = np.vstack([inputs_pend, query])
        query_and_val = (X, -eval_losses(X[None], parallelism=parallelism))
    _ = plot_acquisition(axes[1], grid, -losses, query_and_val=query_and_val)

    if hasattr(loss_fn, 'discretization'):
      for loc in sess.run(loss_fn.discretization):
        axes[1].axvline(loc, linewidth=0.5, color='silver', alpha=0.5)

  for k, ax in enumerate(axes):
    if parallelism != 2 and k+1 < len(axes):
      plt.setp(ax.get_xticklabels(), visible=False)

    ax.set_xlim((0.0, 1.0))
    if k == 0 or parallelism == 1:
      ax.set_xlim((-0.01, 1.01))
    elif parallelism == 2:
      ax.set_ylim((0.0, 1.0))

    ax.set_xticks(np.around(np.linspace(0, 1, 5), 3))
    ax.set_yticks(np.around(np.linspace(*ax.get_ylim(), 5), 3))
  return fig, axes

