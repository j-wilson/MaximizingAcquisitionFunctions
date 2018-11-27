#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
Authored: 2018-01-13 
Modified: 2018-06-04
'''

# ---- Dependencies
import logging
import numpy as np
import tensorflow as tf
from src import utils
from pdb import set_trace as bp

__all__ = \
[
  'sample_spectrum',
  'cosine_process',
  'sample_embedding',
  'predict_theta',
  'sample_theta',
  'random_feature_mapping',
]

logger = logging.getLogger(__name__)
# ==============================================
#                         random_feature_mapping
# ==============================================
def sample_spectrum(kernel_id, shape, amplitude=None, lenscales=None, **kwargs):
  '''
  Generate samples from power spectral density of
  squared exponential kernel.
  '''
  with tf.name_scope('sample_spectrum') as scope:
    if kernel_id in ['squared_exponential']:
      samples = tf.random_normal(shape, **kwargs)
    elif kernel_id in ['matern52']:
      samples = utils.random_multivariate_t(shape, deg_freedom=5, **kwargs)
    elif kernel_id in ['matern32']:
      samples = utils.random_multivariate_t(shape, deg_freedom=3, **kwargs)
    else:
      raise NotImplementedError(
        "No random feature generator for kernel '{}'".format(kernel_id))
    return samples


class cosine_process(object):
  def __init__(self, weights, phases, log_lenscales=None, log_amplitude=None,
    trainable=False, scope=None, reuse=None, name=None):
    self.name = name or self.__class__.__name__
    with tf.variable_scope(scope or self.name, reuse=reuse):
      def build_param(name, init):
        if (init is None): return None #short-cicruit
        elif isinstance(init, tf.Variable): return init #allow passed in vars
        return tf.get_variable(name, initializer=init, trainable=trainable)

      self.weights = build_param('weights', weights)
      self.phases = build_param('phases', phases)
      self.log_lenscales = build_param('log_lenscales', log_lenscales)
      self.log_amplitude = build_param('log_amplitude', log_amplitude)

  def __call__(self, X, weights=None, phases=None, lenscales=None,
    amplitude=None, dtype=None, **kwargs):
    if (dtype is None): dtype = X.dtype
    if (weights is None): weights = self.weights
    if (phases is None): phases = self.phases
    if (lenscales is None): lenscales = self.lenscales
    if (amplitude is None): amplitude = self.amplitude
    with tf.name_scope(self.name) as scope:

      # As needed, cast to the specied dtype
      if not X.dtype.is_compatible_with(dtype):
        X = tf.cast(X, dtype)
      if not weights.dtype.is_compatible_with(dtype):
        weights = tf.cast(weights, dtype)
      if not phases.dtype.is_compatible_with(dtype):
        phases = tf.cast(phases, dtype)
      if not lenscales.dtype.is_compatible_with(dtype):
        lenscales = tf.cast(lenscales, dtype)
      if not amplitude.dtype.is_compatible_with(dtype):
        amplitude = tf.cast(amplitude, dtype)

      X_proj = tf.tensordot(1/lenscales * X, weights, [[-1], [-1]])
      features = amplitude * tf.cos(X_proj + phases)
      return features


  @property
  def amplitude(self):
    if (self.log_amplitude is None): return None
    amplitude = tf.exp(self.log_amplitude)
    return amplitude

  @property
  def lenscales(self):
    if (self.log_lenscales is None): return None
    lenscales = tf.exp(self.log_lenscales)
    return lenscales

  @property
  def dtype(self): #get dtype's string identified
    return np.dtype(self.weights.dtype.base_dtype.as_numpy_dtype).name

  @property
  def input_dim(self):
    return self.weights.get_shape()[-1].value

  @property
  def output_dim(self):
    return self.weights.get_shape()[0].value



def sample_embedding(kernel_id, input_dim, output_dim, dtype=None,
  antithetic=True, embedding_factory=cosine_process, **kwargs):
  '''
  Sample random feature embedding (cosine processes)
  from a stationary kernel's Fourier dual.
  '''
  with tf.name_scope('sample_embedding') as scope:
    # Draw phases uniformly \in [0, 2\pi]
    phases = 2*np.pi*tf.random_uniform([output_dim], dtype=dtype)

    # Reduce variance by using symmetrized sample pairs {x, -x}?
    if antithetic:
      assert output_dim % 2 == 0, 'Number of projections must be even.'
      W_shape = [output_dim//2, input_dim]
    else:
      W_shape = [output_dim, input_dim]

    # Sample random projections from kernel's spectral density.
    weights = sample_spectrum(kernel_id, W_shape, dtype=dtype)
    if antithetic:
      weights = tf.concat([weights, -weights], axis=0)

    # Build embedding function
    embedding = embedding_factory(weights, phases, **kwargs)
    return embedding


def predict_theta(embedding, X, Y=None, Z=None, noise=None, jitter=0.0,
  use_woodbury=True, test_zero_mean=False, err_tol=1e-6, **kwargs):
  '''
  Compute the posterior distribution over weight vector \theta
  given observations (X, Y) for y \sim N(f(x); noise).
  '''
  with tf.name_scope('predict_theta') as scope:
    # Prepare for subroutines
    if (Z is None): Z = embedding(X, **kwargs)
    I, Zt = tf.eye(tf.shape(Z)[1], dtype=Z.dtype), tf.transpose(Z)
    sigma2 = jitter if (noise is None) else tf.cast(noise, Z.dtype) + jitter

    def fewer_observations():
      '''
      Solve using Sherman-Morrison-Woodbury; also, compute
      eigendecomposition for use when sampling \theta.
      '''
      # Feature-wise inner product
      ZZt = tf.matmul(Z, Zt)
      ZZt_nz = ZZt + sigma2*tf.eye(tf.shape(Z)[0], dtype=Z.dtype)
      if (Y is None): #same as Y = [0,...,0]
        mu = tf.zeros([tf.shape(Z)[0], 1], dtype=Z.dtype)
      else:
        chol = tf.cholesky(ZZt_nz)
        A = tf.matmul(Zt, 1/sigma2 * Y)
        beta = tf.cholesky_solve(chol, tf.matmul(Z, A))
        mu = A - tf.matmul(Zt, beta)
      eigen_vals, eigen_vects = tf.self_adjoint_eig(ZZt_nz)
      return mu, (eigen_vals, eigen_vects)

    def fewer_features():
      # Input-wise inner product
      ZtZ = tf.matmul(Zt, Z)
      inv = tf.matrix_inverse(ZtZ + sigma2*I)
      cov = sigma2 * inv
      if (Y is not None):
        mu = tf.matmul(inv, tf.matmul(Zt, Y))
      else:
        mu = tf.zeros([tf.shape(Z)[0], 1], dtype=Z.dtype)
      return mu, cov

    # Test whether posterior mean has been zero'd out?
    tests = []
    if test_zero_mean and (Y is not None):
      global_mean = tf.reduce_mean(Y)
      if not isinstance(err_tol, tf.Tensor):
        err_tol = tf.constant(err_tol, global_mean.dtype)
      tests.append(tf.assert_less(global_mean, err_tol, name='test_zero_mean'))

    # Compute posterior using less expensive subroutine
    with tf.control_dependencies(tests):
      if use_woodbury:
        (mu, eigen_decomp), cov = fewer_observations(), None
      else:
        (mu, cov), eigen_decomp = fewer_features(), None

    return mu, cov, eigen_decomp


def _sample_theta(num_samples, mu, cov=None, eigen_decomp=None, \
  Z=None, base_rvs=None, noise=None, jitter=None, dtype=None):
  '''
  Subroutine for actually drawing samples from a belief over
  the parameters $\theta$ used to combine random features.
  '''
  if (dtype is None): dtype = mu.dtype

  # Draw i.i.d. standard normal r.v.'s
  if (base_rvs is None):
    rvs_shape = (num_samples, tf.shape(mu)[0])
    base_rvs = tf.random_normal(rvs_shape, dtype=dtype)

  # Compute sample residuals for \theta
  if (eigen_decomp is not None and Z is not None):
    # Eigendecomposition terms
    sigma2 = tf.cast(jitter if (noise is None) else noise + jitter, dtype)
    D, U = eigen_decomp[0], eigen_decomp[1]
    test_eigenvalues = tf.assert_non_negative(D, name='test_eigenvalues')
    with tf.control_dependencies([test_eigenvalues]):
      R = tf.reciprocal(D + tf.sqrt(D)*tf.sqrt(sigma2))

    # See appendix B.2 of [Seeger, 2008]
    # 'Bayesian Inference and Optimal Design for the Sparse Linear Model'
    q = base_rvs #here for notational clarity
    Zq = tf.matmul(Z, q, transpose_b=True)
    RUZq = tf.matmul(R*U, Zq, transpose_a=True)
    resid = q - tf.matmul(tf.matmul(U, RUZq), Z, transpose_a=True)
  else:
    assert cov is not None, 'Either a covariance matrix or eigendecomposition'\
                            ' and features Z must be provided'
    chol = utils.jitter_cholesky(cov)
    resid = tf.matmul(base_rvs, chol, transpose_b=True)

  # Add in mean and rescale
  samples = tf.add(tf.squeeze(mu), resid)
  return samples


def sample_theta(embedding, num_samples, X=None, Y=None, Z=None,
  base_rvs=None, noise=None, jitter=0.0, dtype=None, **kwargs):
  '''
  Draw samples from the posterior distribution of \theta
  given observations (X, Y) for purposes of Monte Carlo 
  integrating over a collections of feature embeddings.
  '''
  if (dtype is None): dtype = embedding.dtype
  with tf.name_scope('sample_theta') as scope:
    # Determine rescaling factor
    #  - Numerator from implicit angle identity
    #  - Denominator gives us the expectation
    num_features = embedding.output_dim
    scale = tf.sqrt(2.0/tf.cast(num_features, dtype))

    # Draw i.i.d. standard normal r.v.'s
    if (base_rvs is None):
      rvs_shape = (num_samples, num_features)
      base_rvs = tf.random_normal(rvs_shape, dtype=dtype)

    # Sample from prior: $\theta \sim N(0, I)$
    if (X is None): return scale * base_rvs, None

    # Precompute and rescale features Z (casting enables mixed-precision)
    if (Z is None): Z = scale * tf.cast(embedding(X, **kwargs), dtype)

    # Compute MVN posterior over \theta
    mu, cov, eigen_decomp = predict_theta(embedding, X, Y, Z=Z,
                              noise=noise, jitter=jitter, **kwargs)

    samples = _sample_theta(num_samples, mu, cov, eigen_decomp,
                            Z, base_rvs, noise, jitter, dtype)

    return scale * samples, (mu, cov, eigen_decomp)


class random_feature_mapping(object):
  def __init__(self, embedding, theta, mean_fn=None,
    post_embedding_op=None):
    self.embedding = embedding
    self.theta = theta

    if (mean_fn is None):
      def mean_fn(X, **kwargs):
        return 0.0
    self.mean_fn = mean_fn

    if (post_embedding_op is None):
      def post_embedding_op(Z, theta, **kwargs):
        return tf.tensordot(Z, theta, [[-1], [-1]], **kwargs)
    self.post_embedding_op = post_embedding_op #[?] better name...


  def __call__(self, X, theta=None, embedding=None, mean_fn=None,
    post_embedding_op=None, dtype=None, **kwargs):
    if (dtype is None): dtype = X.dtype
    if (theta is None): theta = self.theta
    if (embedding is None): embedding = self.embedding
    if (mean_fn is None): mean_fn = self.mean_fn
    if (post_embedding_op is None): post_embedding_op = self.post_embedding_op

    # Allow for mixed precision inputs
    if not X.dtype.is_compatible_with(self.dtype):
      X = tf.cast(X, self.dtype)

    # Evaluate random features Z at X
    Z = embedding(X, **kwargs)

    # Compute, e.g., weighted sum of features Z
    Y = post_embedding_op(Z, theta)

    # Adjust output values
    if (mean_fn is not None):
      retvals = mean_fn(X)
      if not isinstance(retvals, tuple): retvals = retvals,
      Y = self.adjust_outputs(Y, *retvals)

    return tf.cast(Y, dtype) #return as specified dtype


  def adjust_outputs(self, Y_hat, mean, var=None):
    if (var is None):
      # Assumed to be a constant-valued mean fn.
      Y = Y_hat + mean
    else:
      # Correct RFGP marginal statistis using those of the GP
      Y_hat -= tf.reduce_mean(Y_hat, axis=-1, keep_dims=True)
      Y_var = tf.reduced_mean(tf.square(Y_hat), axis=-1, keep_dims=True)
      Y = mean + Y_hat*tf.sqrt(utils.safe_div(var, Y_var))
    return Y


  @property
  def input_dim(self): return self.embedding.input_dim

  @property
  def dtype(self):
    return np.dtype(self.theta.dtype.base_dtype.as_numpy_dtype).name


def _build_mappings(sess, model, inputs_old, outputs_old,
  num_paths=1024, num_features=1024, trainable=False):
  '''
  Usage example for <random_feature_mapping> and associated methods.
  '''
  input_dim = inputs_old.shape[-1]

  # Construct placeholders and initial feed_dict
  X = tf.placeholder(FLOAT, shape=[None, 1], name='X')
  Y = tf.placeholder(FLOAT, shape=[None, 1], name='Y')
  feed_dict = {X:inputs_old, Y:outputs_old}

  # Generate random feature embeddings
  embedding = sample_embedding\
  (
    kernel_id=model.kernel_id,
    input_dim=input_dim,
    output_dim=num_features,
    dtype=model.dtype,
    log_lenscales=model.log_lenscales,
    log_amplitude=model.log_amplitude,
    trainable=trainable,
  )
  embedding_params= [embedding.weights, embedding.phases]
  embedding_init_op = tf.variables_initializer(embedding_params)
  _ = sess.run(embedding_init_op)

  # Sample \theta and initialize as <tf.Variable>
  theta_op, theta_posterior = sample_theta\
  (
    embedding,
    num_paths,
    X,
    Y - model.mean,
    noise=model.noise,
    jitter=model.jitter,
  )

  theta_src = sess.run(theta_op, feed_dict)
  theta_var = tf.get_variable('theta', initializer=theta_src, trainable=trainable)
  theta_init_op = tf.variables_initializer([theta_var])
  _ = sess.run(theta_init_op)

  # Build random feature-based sample paths
  mean_fn = lambda *args, **kwargs: model.mean
  sample_paths = random_feature_mapping(embedding, theta_var, mean_fn)
  return sample_paths


if __name__ == '__main__':
  import matplotlib.pyplot as plt
  from src.models import gaussian_process

  # Run settings
  logging.basicConfig\
  (
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    format='%(asctime)-4s %(levelname)s:%(name)s:%(message)s'
  )
  tf.logging.set_verbosity('ERROR') #limit TensorFlow's internal logging
  FLOAT = 'float64'

  # seed = np.random.choice(2**31 - 1)
  seed = 1469946063
  rng = np.random.RandomState(seed)
  tf.set_random_seed(seed)
  logger.info('Seed: {:d}'.format(seed))

  # Construct placeholders and initial feed_dict
  X1_ref = tf.placeholder(FLOAT, shape=[None, 1], name='X1')
  X0_ref = tf.placeholder(FLOAT, shape=[None, 1], name='X0')
  Y0_ref = tf.placeholder(FLOAT, shape=[None, 1], name='Y0')

  X0_src = rng.rand(4, 1).astype(X0_ref.dtype.as_numpy_dtype)
  # X0_src = np.linspace(0, 1, 8, dtype=X0_ref.dtype.as_numpy_dtype)[:, None]
  X1_src = np.linspace(0, 1, 1024, dtype=X1_ref.dtype.as_numpy_dtype)[:, None]
  feed_dict = {X1_ref:X1_src, X0_ref:X0_src}
  
  # Initialize GP and build predict_op
  gp = gaussian_process\
  (
    kernel_id='matern52',
    log_lenscales=np.log(0.1),
    log_amplitude=np.log(1.0),
    log_noise=np.log(1e-3),
    mean=100,
    dtype='float64',
  )
  predict_op = gp.predict(X1_ref, X0_ref, Y0_ref)
  gp_init_op = tf.variables_initializer(list(gp.state.values()))

  # Create TensorFlow session, initialize GP variables, 
  sess = tf.InteractiveSession()
  sess.run(gp_init_op)

  # Sample Y0 from the true prior
  K_xx = gp.covar_fn(X0_ref, X0_ref)
  L_xx = utils.jitter_cholesky(K_xx, jitter=gp.jitter)
  rvs = tf.random_normal([tf.shape(L_xx)[0], 1], dtype=L_xx.dtype)
  sample_y0 = gp.mean + tf.matmul(L_xx, rvs)
  Y0_src = sess.run(sample_y0, {X0_ref : X0_src})
  feed_dict[Y0_ref] = Y0_src

  # Calculate closed-form GP posterior
  mu_exact, var_exact = sess.run(predict_op, feed_dict)

  # Generate random feature-based sample paths
  paths = _build_mappings(sess, gp, X0_src, Y0_src,
                    num_paths=1024, num_features=1024)

  sample_paths = sess.run(paths(X1_ref), {X1_ref:X1_src})


  # Calculate empirical statistics
  mu_emp = np.mean(sample_paths, -1)
  var_emp = np.var(sample_paths, axis=-1)

  # Visualize posteriors and sample paths
  def plot_posterior(ax, x, mu=None, var=None, samples=None, color=None, zorder=0,
    lw=0.5, alpha=0.3):
    if (x.ndim > 1): x = np.squeeze(x)

    if (mu is not None):
      if (mu.ndim > 1): mu = np.squeeze(mu)
      if (var is not None):
        if (var.ndim > 1): var = np.squeeze(var)
        bound = 2 * np.sqrt(var)
        ax.fill_between(x, mu + bound, mu - bound, alpha=0.2, color=color, zorder=zorder+1)
      ax.plot(x, mu, alpha=0.7, color=color, linewidth=1.5, zorder=zorder+4)
    if (samples is not None):
      ax.plot(x, samples, alpha=alpha, color=color, linewidth=lw, zorder=zorder+3)

  plt.rc('text', usetex=True)  
  plt.rc('font', family='serif', size=11)

  fig, ax = plt.subplots(figsize=(12, 4))
  ax.scatter(X0_src, Y0_src, marker='.', s=64, alpha=0.8, color='k', zorder=5)
  plot_posterior(ax, X1_src, mu_exact, var_exact, color='tab:blue')

  ylims = ax.get_ylim()

  num_samples = sample_paths.shape[-1]
  if (num_samples < 64): #don't show more than 64 sample paths
    samples_shown = sample_paths
  else:
    indices = rng.choice(num_samples, 64, replace=False)
    samples_shown = sample_paths[:, indices]
  plot_posterior(ax, X1_src, mu_emp, var_emp, samples_shown, color='tab:orange')
  plot_posterior(ax, X1_src, samples=samples_shown[:, 4:5], color='tab:orange',
                  lw=1.0, alpha=0.8)

  # samples_shown = samples_shown[:, 4:5]

  U = np.ravel(X1_src)
  width, indices = 5e-3, []
  for x in [0.2, 0.4]:
    ax.fill_between([x - width, x + width], *ylims, color='silver',
                    alpha=0.32, zorder=4)
    indices.append(np.argmin(np.abs(U - x)))

  options, x_min, y_min = U[indices], [], []
  for sample in samples_shown.T:
    y_vals = sample[indices]
    argmin = np.argmin(y_vals)
    x_min.append(options[argmin])
    y_min.append(y_vals[argmin])

  ax.scatter(x_min, y_min, marker='*', s=40, color='tab:orange',
            alpha=0.6, linewidth=0.5, edgecolor='k', zorder=10)
  
  ax.set_ylim(*ylims)

  ax.set_xlim([-0.01, 1.01])
  # ax.set_title('GP posteriors and approx. sample paths')
  ax.set_xlabel('Input space $\mathcal{X}$')
  ax.set_ylabel('Output space $\mathcal{Y}$')
  plt.show()


# ==============================================
#                            Developers' Section
# ==============================================
# ------------ Scrap work goes here ------------
'''
''' 