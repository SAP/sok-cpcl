"""
Code from https://github.com/google-research/federated/blob/master/distributed_dp/accounting_utils.py#L259


std_dev for gaussian from:
https://github.com/tensorflow/federated/blob/v0.73.0/tensorflow_federated/python/aggregators/differential_privacy.py#L220-L255
"""


from scipy import optimize, special
from .rdp_accountant import get_privacy_spent
import numpy as np
import math





RDP_ORDERS = tuple(range(2, 129)) + (256,)


def _log_add(logx, logy):
  """Add two numbers in the log space."""
  a, b = min(logx, logy), max(logx, logy)
  if a == -np.inf:  # adding 0
    return b
  # Use exp(a) + exp(b) = (exp(a - b) + 1) * exp(b)
  return math.log1p(math.exp(a - b)) + b  # log1p(x) = log(x + 1)



def _log_sub(logx, logy):
  """Subtract two numbers in the log space. Answer must be non-negative."""
  if logx < logy:
    raise ValueError("The result of subtraction must be non-negative.")
  if logy == -np.inf:  # subtracting 0
    return logx
  if logx == logy:
    return -np.inf  # 0 is represented as -np.inf in the log space.

  try:
    # Use exp(x) - exp(y) = (exp(x - y) - 1) * exp(y).
    return math.log(math.expm1(logx - logy)) + logy  # expm1(x) = exp(x) - 1
  except OverflowError:
    return logx
  

def _log_comb(n, k):
  return (special.gammaln(n + 1) - special.gammaln(k + 1) -
          special.gammaln(n - k + 1))


def _compute_log_a_int(q, sigma, alpha):
  """Compute log(A_alpha) for integer alpha. 0 < q < 1."""
  assert isinstance(alpha, int)

  # Initialize with 0 in the log space.
  log_a = -np.inf

  for i in range(alpha + 1):
    log_coef_i = (
        _log_comb(alpha, i) + i * math.log(q) + (alpha - i) * math.log(1 - q))

    s = log_coef_i + (i * i - i) / (2 * (sigma**2))
    log_a = _log_add(log_a, s)

  return float(log_a)

def _compute_log_a_frac(q, sigma, alpha):
  """Compute log(A_alpha) for fractional alpha. 0 < q < 1."""
  # The two parts of A_alpha, integrals over (-inf,z0] and [z0, +inf), are
  # initialized to 0 in the log space:
  log_a0, log_a1 = -np.inf, -np.inf
  i = 0

  z0 = sigma**2 * math.log(1 / q - 1) + .5

  while True:  # do ... until loop
    coef = special.binom(alpha, i)
    log_coef = math.log(abs(coef))
    j = alpha - i

    log_t0 = log_coef + i * math.log(q) + j * math.log(1 - q)
    log_t1 = log_coef + j * math.log(q) + i * math.log(1 - q)

    log_e0 = math.log(.5) + _log_erfc((i - z0) / (math.sqrt(2) * sigma))
    log_e1 = math.log(.5) + _log_erfc((z0 - j) / (math.sqrt(2) * sigma))

    log_s0 = log_t0 + (i * i - i) / (2 * (sigma**2)) + log_e0
    log_s1 = log_t1 + (j * j - j) / (2 * (sigma**2)) + log_e1

    if coef > 0:
      log_a0 = _log_add(log_a0, log_s0)
      log_a1 = _log_add(log_a1, log_s1)
    else:
      log_a0 = _log_sub(log_a0, log_s0)
      log_a1 = _log_sub(log_a1, log_s1)

    i += 1
    if max(log_s0, log_s1) < -30:
      break

  return _log_add(log_a0, log_a1)


def _compute_log_a(q, sigma, alpha):
  """Compute log(A_alpha) for any positive finite alpha."""
  if float(alpha).is_integer():
    return _compute_log_a_int(q, sigma, int(alpha))
  else:
    return _compute_log_a_frac(q, sigma, alpha)


def _compute_rdp(q, sigma, alpha):
  """Compute RDP of the Sampled Gaussian mechanism at order alpha.

  Args:
    q: The sampling rate.
    sigma: The std of the additive Gaussian noise.
    alpha: The order at which RDP is computed.

  Returns:
    RDP at alpha, can be np.inf.
  """
  if q == 0:
    return 0

  if q == 1.:
    return alpha / (2 * sigma**2)

  if np.isinf(alpha):
    return np.inf

  return _compute_log_a(q, sigma, alpha) / (alpha - 1)



def compute_rdp(q, noise_multiplier, steps, orders):
  """Computes RDP of the Sampled Gaussian Mechanism.

  Args:
    q: The sampling rate.
    noise_multiplier: The ratio of the standard deviation of the Gaussian noise
      to the l2-sensitivity of the function to which it is added.
    steps: The number of steps.
    orders: An array (or a scalar) of RDP orders.

  Returns:
    The RDPs at all orders. Can be `np.inf`.
  """
  if np.isscalar(orders):
    rdp = _compute_rdp(q, noise_multiplier, orders)
  else:
    rdp = np.array(
        [_compute_rdp(q, noise_multiplier, order) for order in orders])

  return rdp * steps



def get_eps_gaussian(q, noise_multiplier, steps, target_delta, orders):
  """Compute eps for the Gaussian mechanism given the DP params."""
  rdp = compute_rdp(
      q=q, noise_multiplier=noise_multiplier, steps=steps, orders=orders)
  eps, _, _ = get_privacy_spent(orders, rdp, target_delta=target_delta)
  return eps





def get_gauss_noise_multiplier(target_eps,
                               target_delta,
                               target_sampling_rate,
                               steps,
                               orders=RDP_ORDERS):
  """Compute the Gaussian noise multiplier given the DP params."""

  def get_eps_for_noise_multiplier(z):
    eps = get_eps_gaussian(
        q=target_sampling_rate,
        noise_multiplier=z,
        steps=steps,
        target_delta=target_delta,
        orders=orders)
    return eps

  def opt_fn(z):
    return get_eps_for_noise_multiplier(z) - target_eps

  min_nm, max_nm = 0.001, 1000
  result, r = optimize.brentq(opt_fn, min_nm, max_nm, full_output=True)
  if r.converged:
    return result
  else:
    return -1
  

def get_gaussian_stddev(eps, delta, sampling_rate, steps, clipping_threshold, orders=RDP_ORDERS):
  """Compute the Gaussian noise standard deviation given the DP params."""
  noise_multiplier = get_gauss_noise_multiplier(eps, delta, sampling_rate, steps, orders)
  return noise_multiplier * clipping_threshold