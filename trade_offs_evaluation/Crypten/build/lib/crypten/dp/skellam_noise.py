# Copyright 2021, Google LLC. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


""" DP Accounting with composition for Gaussian and DDGaussian."""
""" Code from https://github.com/google-research/federated/blob/master/distributed_dp/accounting_utils.py#L491 """


import math

import numpy as np
from scipy import optimize
from scipy import special
from .rdp_accountant import get_privacy_spent

RDP_ORDERS = tuple(range(2, 129)) + (256,)
DIV_EPSILON = 1e-22


def _compute_rdp_subsampled(alpha, gamma, eps, upper_bound=True):
  """Computes RDP with subsampling.

  Reference: http://proceedings.mlr.press/v97/zhu19c/zhu19c.pdf.

  Args:
    alpha: The RDP order.
    gamma: The subsampling probability.
    eps: The RDP function taking alpha as input.
    upper_bound: A bool indicating whether to use Theorem 5 of the referenced
      paper above (if set to True) or Theorem 6 (if set to False).

  Returns:
    The RDP with subsampling.
  """
  if isinstance(alpha, float):
    assert alpha.is_integer()
    alpha = int(alpha)
  assert alpha > 1
  assert 0 < gamma <= 1

  if upper_bound:
    a = [0, eps(2)]
    b = [((1 - gamma)**(alpha - 1)) * (alpha * gamma - gamma + 1),
         special.comb(alpha, 2) * (gamma**2) * (1 - gamma)**(alpha - 2)]

    for l in range(3, alpha + 1):
      a.append((l - 1) * eps(l) + log_comb(alpha, l) +
               (alpha - l) * np.log(1 - gamma) + l * np.log(gamma))
      b.append(3)

  else:
    a = [0]
    b = [((1 - gamma)**(alpha - 1)) * (alpha * gamma - gamma + 1)]

    for l in range(2, alpha + 1):
      a.append((l - 1) * eps(l) + log_comb(alpha, l) +
               (alpha - l) * np.log(1 - gamma) + l * np.log(gamma))
      b.append(1)

  return special.logsumexp(a=a, b=b) / (alpha - 1)


def rounded_l2_norm_bound(l2_norm_bound, beta, dim):
  """Computes the L2 norm bound after stochastic rounding to integers.

  Note that this function is *agnostic* to the actual vector whose coordinates
  are to be rounded, and it does *not* consider the effect of scaling (i.e.
  we assume the input norm is already scaled before rounding).

  See Theorem 1 of https://arxiv.org/pdf/2102.06387.pdf.

  Args:
    l2_norm_bound: The L2 norm (bound) of the vector whose coordinates are to be
      stochastically rounded to the integer grid.
    beta: A float constant in [0, 1). See the initializer docstring of the
      aggregator.
    dim: The dimension of the vector to be rounded.

  Returns:
    The inflated L2 norm bound after stochastic rounding (conditionally
    according to beta).
  """
  assert int(dim) == dim and dim > 0, f'Invalid dimension: {dim}'
  assert 0 <= beta < 1, 'beta must be in the range [0, 1)'
  assert l2_norm_bound > 0, 'Input l2_norm_bound should be positive.'

  bound_1 = l2_norm_bound + np.sqrt(dim)
  if beta == 0:
    return bound_1

  squared_bound_2 = np.square(l2_norm_bound) + 0.25 * dim
  squared_bound_2 += (
      np.sqrt(2.0 * np.log(1.0 / beta)) * (l2_norm_bound + 0.5 * np.sqrt(dim)))
  bound_2 = np.sqrt(squared_bound_2)
  return min(bound_1, bound_2)

def rounded_l1_norm_bound(l2_norm_bound, dim):
  # In general we have L1 <= sqrt(d) * L2. In the scaled and rounded domain
  # where coordinates are integers we also have L1 <= L2^2.
  return l2_norm_bound * min(np.sqrt(dim), l2_norm_bound)


def _skellam_rdp(l1_sens, l2_sens, central_var, scale, order):
  """
    Define the epsilon value for order (alpha) for multidimensional skellam mechanism
    From Corollary 4.1 of The Skellam Mechanism for Differentially Private Federated Learning, Agarwal et al.
	"""
  assert order > 1, f'alpha must be greater than 1. Found {order}.'
  a, s, mu = order, scale, central_var
  rdp = a / (2 * mu) * l2_sens**2
  rdp += min(((2 * a - 1) * s * l2_sens**2 + 6 * l1_sens) / (4 * s**3 * mu**2),
             3 * l1_sens / (2 * s * mu))
  return rdp


def skellam_epsilon(scale,
                    central_stddev,
                    l2_sens,
                    beta,
                    dim,
                    q,
                    steps,
                    delta,
                    l1_sens=None,
                    rounding=True,
                    orders=RDP_ORDERS):
  """Computes epsilon of (distributed) Skellam via RDP."""
  l1_sens = l1_sens or (l2_sens * np.sqrt(dim))
  if rounding:
    l2_sens = rounded_l2_norm_bound(l2_sens * scale, beta, dim) / scale
    l1_sens = rounded_l1_norm_bound(l2_sens * scale, dim) / scale

  orders = [int(order) for order in orders]
  central_var = central_stddev**2

  def eps_fn(order):
    return _skellam_rdp(l1_sens, l2_sens, central_var, scale, order)

  if q == 1:
    rdp = np.array([eps_fn(order) for order in orders])
  else:
    # Take min between subsampled RDP and unamplified RDP, for all orders.
    rdp = np.array([
        min(_compute_rdp_subsampled(order, q, eps_fn), eps_fn(order))
        for order in orders
    ])

  eps, _, order = get_privacy_spent(orders, rdp * steps, target_delta=delta)
  return eps, order


def skellam_local_stddev(epsilon,
                         scale,
                         l2_clip,
                         num_clients,
                         beta,
                         dim,
                         q,
                         steps,
                         delta,
                         orders=RDP_ORDERS):
  """Selects the local stddev for the distributed skellam."""

  def stddev_opt_fn(local_stddev):
    local_stddev += DIV_EPSILON
    central_stddev = local_stddev * np.sqrt(num_clients)
    cur_epsilon, _ = skellam_epsilon(
        scale,
        central_stddev,
        l2_clip,
        beta,
        dim,
        q,
        steps,
        delta,
        orders=orders)
    return (epsilon - cur_epsilon)**2

  local_stddev_result = optimize.minimize_scalar(stddev_opt_fn)
  if not local_stddev_result.success:
    raise ValueError('Cannot compute local_stddev for Skellam.')

  return local_stddev_result.x


def skellam_params(epsilon,
                   l2_clip,
                   bits,
                   num_clients,
                   beta,
                   dim,
                   q,
                   steps,
                   delta,
                   k=3,
                   rho=1,
                   sqrtn_norm_growth=False,
                   orders=RDP_ORDERS):
  """Computes the scaling and local noise stddev for Skellam."""
  n_factor = num_clients**(1 if sqrtn_norm_growth else 2)

  # The implementation optimizes for gamma = 1 / scale for stability.
  def local_stddev(gamma):
    scale = 1.0 / (gamma + DIV_EPSILON)
    return skellam_local_stddev(epsilon, scale, l2_clip, num_clients, beta, dim,
                                q, steps, delta, orders)

  def mod_min(gamma):
    var = rho / dim * l2_clip**2 * n_factor
    var += (gamma**2 / 4 + local_stddev(gamma)**2) * num_clients
    return k * math.sqrt(var)

  def gamma_opt_fn(gamma):
    return (math.pow(2, bits) - 2 * mod_min(gamma) / (gamma + DIV_EPSILON))**2

  gamma_result = optimize.minimize_scalar(gamma_opt_fn)
  if not gamma_result.success:
    raise ValueError('Cannot compute scaling factor.')

  scale = 1. / gamma_result.x
  # Select the local_stddev that gave the best scale.
  local_stddev = skellam_local_stddev(epsilon, scale, l2_clip, num_clients,
                                      beta, dim, q, steps, delta, orders)

  return scale, local_stddev





