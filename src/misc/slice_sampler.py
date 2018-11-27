#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
TensorFlow implementation of slice sampler, with
Gibbs sampling subroutine.

To Do:
	- Add doc strings...

Authored: 2017-10-13 
Modified: 2017-12-21
'''

# ---- Dependencies
import tensorflow as tf
from pdb import set_trace as bp

# ==============================================
#                                  slice_sampler
# ==============================================
class slice_sampler(object):
	def __init__(self, step_limit=1024, step_size=1.0, use_doubling=False):
		self.step_limit = int(step_limit)
		self.step_size = step_size
		self.use_doubling = use_doubling


	def sample(self, fn, x, proj=None, return_fx=False, **kwargs):
		with tf.name_scope('sample') as scope:
			with tf.control_dependencies([tf.assert_rank(x, 1)]):
				dtype, num_elements = x.dtype, x.get_shape()[0].value

			if (proj is None): #self-normalized Gaussian vector
				proj = tf.random_normal([num_elements], dtype=dtype)
				proj *= tf.reciprocal(tf.norm(proj, ord='euclidean'))

			x, fx = self.projected_slice(fn, x, proj, **kwargs)

			if return_fx: return x, fx
			return x


	def projected_slice(self, fn, x, proj, threshold=None, step_size=None,
		step_limit=None, dtype=None, use_doubling=None, is_log_fn=True):
		if (dtype is None): dtype = tf.as_dtype(x.dtype)
		if (step_size is None): step_size = self.step_size
		if (step_limit is None): step_limit = self.step_limit
		with tf.name_scope('projected_slice') as scope:
			# Sample threshold ~ Uniform(0, fn(x))
			if (threshold is None):
				fx = fn(x)
				rv = tf.random_uniform([], dtype=dtype)
				threshold = fx + tf.log(rv) if is_log_fn else fx*rv

			# Create projected function localized about x
			local_fn = lambda eta: fn(x + eta*proj)

			# Initialize envelope
			upper = step_size * tf.random_uniform([], dtype=dtype)
			lower = upper - step_size

			# 'Stepping out' procedure (expand envelope)
			if step_limit > 0:
				upper, lower = self.step_out(local_fn, threshold, upper, lower, dtype,
																		step_limit=step_limit,step_size=step_size,
																		use_doubling=use_doubling)

			# Generate sample by 'stepping in' (shrink envelope)
			eta, fx = self.step_in(local_fn, threshold, upper, lower, dtype)
			new_x = x + eta*proj
			return new_x, fx


	def step_out(self, local_fn, threshold, upper, lower, dtype=None,
		step_size=None, step_limit=None, use_doubling=None):
		if (dtype is None): dtype = threshold.dtype
		if (step_size is None): step_size = self.step_size
		if (step_limit is None): step_limit = self.step_limit
		if (use_doubling is None):use_doubling = self.use_doubling
		with tf.name_scope('step_out') as scope:
			if not isinstance(step_size, tf.Tensor):
				step_size = tf.constant(step_size, dtype=dtype)

			if use_doubling: subroutine = self._step_out_doubling
			else: subroutine = self._step_out_vanilla
			return subroutine(local_fn, threshold, upper, lower,
												dtype, step_size, step_limit)


	def _step_out_vanilla(self, local_fn, threshold, upper, lower,
		dtype, step_size, step_limit):
		with tf.name_scope('_step_out_vanilla') as scope:
			def cond(step, boundry, delta):
				return tf.logical_and\
				(
					tf.less(step, step_limit),
					tf.greater(local_fn(boundry), threshold),
				)

			def body(step, boundry, delta):
				return step + 1, boundry + delta, delta
	 		
	 		# Grow upper and lower bounds
			new_upper = tf.while_loop(cond, body, [0, upper, step_size])[1]
			new_lower = tf.while_loop(cond, body, [0, lower, -step_size])[1]
			return new_upper, new_lower


	def _step_out_doubling(self, local_fn, threshold, upper, lower,
		dtype, step_size, step_limit):
		with tf.name_scope('_step_out_doubling') as scope:
			def cond(step, upper, lower, fu, fl):
				return tf.logical_and\
				(
					tf.less(step, step_limit),
					tf.logical_or\
					(
						tf.greater(fu, threshold),
						tf.greater(fl, threshold),
					)
				)

			def body(step, upper, lower, fu, fl):
				new_upper, new_lower, new_fu, new_fl = tf.cond\
				(
					tf.less(tf.random_uniform([]), 0.5),
					lambda: (upper, 2*lower - upper, fu, local_fn(2*lower - upper)),
					lambda: (2*upper - lower, lower, local_fn(2*upper - lower), fl),
				)
				return step + 1, new_upper, new_lower, new_fu, new_fl
			
			# Grow upper and lower bounds
			loop_vars = [0, upper, lower, local_fn(upper), local_fn(lower)]
			new_upper, new_lower = tf.while_loop(cond, body, loop_vars)[1:3]
			return new_upper, new_lower


	def step_in(self, local_fn, threshold, upper, lower, dtype=None):
		if (dtype is None): dtype = threshold.dtype
		with tf.name_scope('step_in') as scope:
			def cond(upper, lower, eta, fx):
				return tf.less_equal(fx, threshold)

			def body(upper, lower, eta, fx):
				# Evaluate local_fn(eta) := fn(x + eta*proj)
				eta = lower + (upper - lower)*tf.random_uniform([], dtype=dtype)
				fx = local_fn(eta)

				# Shrink envelope (tighten upper xor lower bound)
				upper, lower = tf.cond\
				(
					eta > 0.0,
					lambda: (eta, lower),
					lambda: (upper, eta),
				)

				# Test boundary case: eta==0.0
				test_envelope = tf.assert_positive(tf.abs(eta), name='test_envelope',
															message='Sampling envelope collapsed to zero.')
				with tf.control_dependencies([test_envelope]):
					return upper, lower, eta, fx

			loop_vars = [upper, lower, tf.zeros([], dtype), threshold - 1]
			eta, fx = tf.while_loop(cond, body, loop_vars,)[-2:]
			return eta, fx


	def gibbs_sample(self, fn, x, return_fx=False, **kwargs):
		with tf.name_scope('gibbs_sample') as scope:
			with tf.control_dependencies([tf.assert_rank(x, 1)]):
				dtype, num_elements = x.dtype, x.get_shape()[0].value

			indices = tf.range(num_elements)
			def gibbs_fn(index, accum):
				old_x = tf.concat([accum, x[index:]], axis=0)
				old_x.set_shape([num_elements]) #starting position
				proj = tf.cast(tf.equal(indices, index), dtype)
				new_x = self.projected_slice(fn, old_x, proj, **kwargs)[0]
				return index + 1, new_x[:index+1]

			x = tf.while_loop\
			(# Iteratively sample x_{i} | x_{\i}
				cond=lambda index, accum: tf.less(index, num_elements),
				body=gibbs_fn,
				loop_vars=(0, tf.zeros([0], dtype=dtype)), 
				shape_invariants=(tf.TensorShape([]), tf.TensorShape([None])),
				parallel_iterations=1, #ensures sequential ordering
			)[1]
			x.set_shape([num_elements])

			if return_fx: return x, fn(x)
			return x



if __name__ == '__main__':
	import numpy as np
	import scipy.linalg as spla

	def epmgp_cov_prior(ndim, scale=10.0, rng=None):
		'''
		Prior over covariance matrices from Cunningham's EPMGP paper.
		'''
		if (rng is None): rng = np.random
		eigen_vals = np.diag(rng.exponential(scale, size=[ndim]))
		orthog_mat = spla.svd(rng.randn(ndim, ndim))[0]
		covariance = np.dot(orthog_mat, np.inner(eigen_vals, orthog_mat))
		return covariance

	num_elements = 3
	loc = tf.placeholder('float64', shape=[num_elements])
	cov_mat = tf.placeholder('float64', shape=[num_elements, num_elements])
	x_start = tf.placeholder('float64', shape=[num_elements])

	test_means = np.zeros(num_elements)
	test_cov = epmgp_cov_prior(num_elements)
	# test_cov *= np.eye(test_cov.shape[-1]) #diagonalize

	chol = spla.cholesky(test_cov, lower=True)
	precis_mat = tf.constant(spla.inv(test_cov)) #precomputed
	def log_prob(x): #MVN log probability (off by a constant)
		return -0.5 * tf.reduce_sum(precis_mat*tf.square(x - loc))

	sampler = slice_sampler()
	sample_op = sampler.sample(fn=log_prob, x=x_start, return_fx=True)

	sess = tf.InteractiveSession()
	feed_dict = {loc:test_means, cov_mat:test_cov}

	def run_mcmc(start, num_samples, thinning=8, burn_in=32):
		assert thinning > 0, 'Thinning factor must be a postive integer.'

		# Preallocate
		samples = np.empty([num_samples, num_elements])
		log_probs = np.empty([num_samples])

		# MCMC sampling looping
		samples[0], enum = start, 0
		while (enum < num_samples - 1):
			# Burn-in phase
			if burn_in > 0:
				samples[enum], log_probs[enum] = sess.run\
				(
					sample_op, 
					{**feed_dict, x_start:samples[enum]}
				)
				burn_in = burn_in - 1

			# Thinned Markov chain
			else:
				enum += 1
				# [!] Hack to use iid starts drawn from the correct distribution
				# samples[enum], log_probs[enum] = sess.run\
				# 	(
				# 		sample_op,
				# 		{**feed_dict, x_start: test_means + np.dot(chol, np.random.randn(num_elements))}
				# 	)
				samples[enum] = samples[enum - 1]
				for step in range(thinning):
					samples[enum], log_probs[enum] = sess.run\
					(
						sample_op,
						{**feed_dict, x_start:samples[enum]}
					)

		return samples, log_probs

	num_samples = 8192
	samples, log_probs = run_mcmc(test_means, num_samples)

	emp_means = np.mean(samples, axis=0)
	emp_cov = np.cov(samples.T)
	print(emp_means, '\n\n', test_cov, '\n\n', emp_cov, '\n\n', test_cov/emp_cov)

	import pylab as pl
	import pymc
	scores = pymc.geweke(log_probs)
	pymc.Matplot.geweke_plot(scores, 'test')
	pymc.raftery_lewis(log_probs, q=0.025, r=0.01)
	pymc.Matplot.autocorrelation(log_probs, 'test')
	pl.show()

	bp()


# ==============================================
#                            Developers' Section
# ==============================================
# ------------ Scrap work goes here ------------
'''
# 'Stepping in' with a mode sophisticated stopping criterion.
def step_in(self, local_fn, threshold, upper, lower, dtype=None):
	if (dtype is None): dtype = threshold.dtype
	with tf.name_scope('step_in') as scope:
		# Source: https://github.com/HIPS/Spearmint/blob/master/spearmint/sampling/mcmc.py
		initial_upper, initial_lower = upper, lower
		def acceptable(eta): 
			def cond(upper, lower, accept):
				return tf.logical_and\
				(
					accept,
					tf.greater(upper - lower, 1.1*self.step_size),
				)

			def body(upper, lower, accept):
				# Precompute shared terms
				middle = 0.5*(upper + lower)
				eta_lt_mid = eta < middle

				# Shrink envelope
				upper, lower = tf.cond\
				(
					eta_lt_mid,
					lambda: (middle, lower),
					lambda: (upper, middle),
				)

				# Update accept/reject decision
				accept = tf.logical_or\
				(
					tf.cond\
					(
						eta_lt_mid,
						lambda: middle > 0,
						lambda: middle <= 0,
					),

					tf.logical_or\
					(
						tf.less(threshold, local_fn(upper)),
						tf.less(threshold, local_fn(lower)),
					),
				)
				return upper, lower, accept

			loop_vars = [initial_upper, initial_lower, tf.constant(True)]
			accept = tf.while_loop(cond, body, loop_vars)[-1]
			return accept

		def cond(upper, lower, eta, fx):
			return tf.logical_or\
			(
				tf.less_equal(fx, threshold),
				tf.logical_not(acceptable(eta)),
			)

		def body(upper, lower, eta, fx):
			# Evaluate local_fn(eta) := fn(x + eta*proj)
			eta = lower + (upper - lower)*tf.random_uniform([], dtype=dtype)
			fx = local_fn(eta)

			# Shrink envelope (tighten upper xor lower bound)
			upper, lower = tf.cond\
			(
				eta > 0.0,
				lambda: (eta, lower),
				lambda: (upper, eta),
			)

			# Test boundary case: eta==0.0
			test_envelope = tf.assert_positive(tf.abs(eta), name='test_envelope',
														message='Sampling envelope collapsed to zero.')
			with tf.control_dependencies([test_envelope]):
				return upper, lower, eta, fx

		loop_vars = [upper, lower, tf.zeros([], dtype), threshold - 1]
		eta, fx = tf.while_loop(cond, body, loop_vars,)[-2:]
		return eta, fx
'''