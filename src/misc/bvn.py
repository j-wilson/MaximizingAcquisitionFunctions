#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
Implementation of Alan Genz's BVN in TensorFlow.

Authored: 2017-10-21 
Modified: 2018-01-16
'''

# ---- Dependencies
import tensorflow as tf
from math import pi as _pi
from pdb import set_trace as bp

# ---- Constants
_inf = float('inf')
_2pi = 2.0 * _pi
_neg_inv_sqrt2 = -1.0/(2.0 ** 0.5)

# ==============================================
#                                            bvn
# ==============================================
def bvn(xl, xu, yl, yu, r):
  '''
  BVN
    A function for computing bivariate normal probabilities.
    bvn calculates the probability that 
      xl < x < xu and yl < y < yu, 
    with correlation coefficient r.
     p = bvn(xl, xu, yl, yu, r)

     Author
        Alan Genz, Department of Mathematics
        Washington State University, Pullman, Wa 99164-3113
        Email : alangenz@wsu.edu
  '''
  p = bvnu(xl,yl,r) - bvnu(xu,yl,r) - bvnu(xl,yu,r) + bvnu(xu,yu,r)
  return tf.clip_by_value(p, 0.0, 1.0)


def bvnu(dh, dk, r):
  # Special cases admitting closed-form solutions
  def empty_region(dh, dk, r): return tf.zeros_like(dh)
  def unbounded_integral(dh, dk, r): return tf.ones_like(dh)
  def marginal_prob(dh, dk, r): return phid(-dh)
  def independent_prob(dh, dk, r): return phid(-dh) * phid(-dk)

  is_empty = tf.logical_or(tf.equal(dh, _inf), tf.equal(dk, _inf))
  is_unbounded_dh = tf.equal(dh, -_inf)
  is_unbounded_dk = tf.equal(dk, -_inf)
  is_unbounded = tf.logical_and(is_unbounded_dh, is_unbounded_dk)
  is_independent = tf.equal(r, 0.0)

  p = tf.case\
  ([
    (is_empty, lambda: empty_region(dh, dk, r)),
    (is_unbounded, lambda: unbounded_integral(dh, dk, r)),
    (is_unbounded_dk, lambda: marginal_prob(dh, dk, r)),
    (is_unbounded_dh, lambda: marginal_prob(dk, dh, r)),
    (is_independent, lambda: independent_prob(dh, dk, r)),
  ], default=lambda: _bvnu(dh, dk, r), exclusive=False)

  return p


def _bvnu(dh, dk, r, thresholds=None, dtype=None):
  '''
  Primary subroutine for bvnu()
  '''
  if (dtype is None): dtype = dh.dtype
  if (thresholds is None): thresholds = dict()
  else: thresholds = thresholds.copy() #stricly local changes

  # Provide default threshold values
  thresholds.setdefault('r', 0.925)
  thresholds.setdefault('hk', -100)
  thresholds.setdefault('asr', -100)

  # Precompute required terms
  tp = tf.cast(_2pi, dtype)
  h = dh
  k = dk
  hk = h*k
  bvn = tf.zeros_like(dh)
  x, w = gauss_legendre(r, dtype=dtype)

  def moderate_corr(h=h, k=k, hk=hk, bvn=bvn):
    hs = 0.5 * (h*h + k*k)
    asr = 0.5 * tf.asin(r)
    sn = tf.sin(asr*x)
    bvn = tf.reduce_sum(w * tf.exp((sn*hk - hs)/(1.0 - sn**2)), axis=-1)
    bvn = bvn*asr/tp + phid(-h)*phid(-k)
    return bvn

  def strong_corr(h=h, k=k, hk=hk, bvn=bvn):
    corr_sign = tf.sign(r)
    k *= corr_sign
    hk *= corr_sign

    def partial_corr(h=h, k=k, hk=hk, bvn=bvn):
      _as = 1.0 - r**2;
      a = tf.sqrt(_as)
      bs = (h - k)**2
      asr = -0.5 * (bs/_as + hk)
      c = 0.125 * (4.0 - hk)
      d = 0.0125 * (12.0 - hk)

      def asr_gt_threshold():
        bvn = a*tf.exp(asr)*(1.0 - c*(bs-_as)*(1-d*bs)/3 + c*d*_as**2)
        return bvn
      bvn = tf.cond(tf.greater(asr, thresholds['asr']), asr_gt_threshold, lambda: bvn)

      def hk_gt_threshold():
        b = tf.sqrt(bs)
        sp = tf.sqrt(tp) * phid(-b/a)
        d_bvn = -tf.exp(-0.5 * hk)*sp*b*(1.0 - c*bs*(1-d*bs)/3)
        return bvn + d_bvn
      bvn = tf.cond(tf.greater(hk, thresholds['hk']), hk_gt_threshold, lambda: bvn)

      a *= 0.5
      xs = (a * x)**2
      asr = -0.5 * (bs/xs + hk)

      ix = tf.squeeze(tf.where(tf.greater(asr, thresholds['asr'])), axis=-1)
      xs = tf.gather(xs, ix)
      sp = 1.0 + c*xs*(1.0 + 5.0*d*xs)
      rs = tf.sqrt(1.0 - xs)
      ep = tf.exp(-0.5 * tf.divide(hk*xs, (1.0 + rs)**2))/rs

      deltas = tf.gather(w, ix) * tf.exp(tf.gather(asr, ix)) * (sp - ep)
      d_bvn = tf.reduce_sum(deltas, axis=-1)
      bvn = tf.divide(a*d_bvn - bvn, tp)
      return bvn

    bvn = tf.cond(tf.less(tf.abs(r), 1.0), partial_corr, lambda: bvn)

    def positive_corr():
      return bvn + phid(-tf.maximum(h, k))
    def default_case():
      def negative_h():
        return phid(k) - phid(h)
      def nonegative_h():
        return phid(-h) - phid(-k)
      L = tf.cond(tf.less(h, 0.0), negative_h, nonegative_h)
      return L - bvn

    bvn = tf.case\
    ([
      (tf.greater(r, 0.0), positive_corr),
      (tf.greater_equal(h, k), lambda: -bvn),
    ], default=default_case, exclusive=False)
    return bvn

  bvn = tf.cond(tf.less(tf.abs(r), thresholds['r']), moderate_corr, strong_corr)
  p = tf.clip_by_value(bvn, 0.0, 1.0)
  return p


def phid(x):
  '''
  Standard normal CDF.
  '''
  return 0.5 * tf.erfc(_neg_inv_sqrt2 * x)


def gauss_legendre(corr, dtype):
  '''
  Returns Gauss-Legendre abscissae and weights for fixed 
  order polynomials.
  '''
  def order6():
    half_abscissae = tf.constant\
    ([
      0.9324695142031522, 0.6612093864662647, 0.2386191860831970
    ], dtype=dtype)

    half_weights = tf.constant\
    ([
      0.1713244923791705, 0.3607615730481384, 0.4679139345726904
    ], dtype=dtype)
    return half_abscissae, half_weights

  def order12():
    half_abscissae = tf.constant\
    ([
      0.9815606342467191, 0.9041172563704750, 0.7699026741943050,
      0.5873179542866171, 0.3678314989981802, 0.1252334085114692
    ], dtype=dtype)

    half_weights = tf.constant\
    ([
      0.04717533638651177, 0.1069393259953183, 0.1600783285433464,
      0.2031674267230659, 0.2334925365383547, 0.2491470458134029
    ], dtype=dtype)
    return half_abscissae, half_weights

  def order20():
    half_abscissae = tf.constant\
    ([
      0.9931285991850949, 0.9639719272779138, 0.9122344282513259,
      0.8391169718222188, 0.7463319064601508, 0.6360536807265150,
      0.5108670019508271, 0.3737060887154196, 0.2277858511416451,
      0.07652652113349733
    ], dtype=dtype)

    half_weights = tf.constant\
    ([
      0.01761400713915212, .04060142980038694, .06267204833410906,
      0.08327674157670475, 0.1019301198172404, 0.1181945319615184,
      0.1316886384491766, 0.1420961093183821, 0.1491729864726037,
      0.1527533871307259
    ], dtype=dtype)
    return half_abscissae, half_weights

  half_abscissae, half_weights = tf.case\
  ([
    (tf.less(tf.abs(corr), 0.30), order6), 
    (tf.less(tf.abs(corr), 0.75), order12),
  ], default=order20, exclusive=False)

  abscissae = tf.concat([1.0 - half_abscissae, 1.0 + half_abscissae], axis=0)
  weights = tf.concat([half_weights, half_weights], axis=0)
  return abscissae, weights