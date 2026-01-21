# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from utils.dp_utils import ddgauss_params
import numpy as np
import random
import fractions
from time import time

GT_SAMPLER_SEED = time()


def exact_sampler(scale, num_samples, seed=GT_SAMPLER_SEED):
  """Implementation of the exact discrete gaussian distribution sampler.

  Source: https://arxiv.org/pdf/2004.00010.pdf.

  Args:
    scale: The scale of the discrete Gaussian.
    num_samples: The number of samples to generate.
    seed: The seed for the random number generator to reproduce samples.

  Returns:
    A numpy array of discrete Gaussian samples.
  """

  def randrange(a, rng):
    return rng.randrange(a)

  def bern_em1(rng):
    """Sample from Bernoulli(exp(-1))."""
    k = 2
    while True:
      if randrange(k, rng) == 0:  # if Bernoulli(1/k)==1
        k = k + 1
      else:
        return k % 2

  def bern_emab1(a, b, rng):
    """Sample from Bernoulli(exp(-a/b)), assuming 0 <= a <= b."""
    assert isinstance(a, int)
    assert isinstance(b, int)
    assert 0 <= a <= b
    k = 1
    while True:
      if randrange(b, rng) < a and randrange(k, rng) == 0:  # if Bern(a/b/k)==1
        k = k + 1
      else:
        return k % 2

  def bern_emab(a, b, rng):
    """Sample from Bernoulli(exp(-a/b)), allowing a > b."""
    while a > b:
      if bern_em1(rng) == 0:
        return 0
      a = a - b
    return bern_emab1(a, b, rng)

  def geometric(t, rng):
    """Sample from geometric(1-exp(-1/t))."""
    assert isinstance(t, int)
    assert t > 0
    while True:
      u = randrange(t, rng)
      if bern_emab1(u, t, rng) == 1:
        while bern_em1(rng) == 1:
          u = u + t
        return u

  def dlap(t, rng):
    """Sample from discrete Laplace with scale t.

    Pr[x] = exp(-|x|/t) * (exp(1/t)-1)/(exp(1/t)+1). Supported on integers.

    Args:
      t: The scale.
      rng: The random number generator.

    Returns:
      A discrete Laplace sample.
    """
    assert isinstance(t, int)
    assert t > 0
    while True:
      u = geometric(t, rng)
      b = randrange(2, rng)
      if b == 1: # prob to return 1/2 per run, prob ignoring u to return after x r
        return u
      elif u > 0:
        return -u

  def floorsqrt(x):
    """Compute floor(sqrt(x)) exactly."""
    assert x >= 0
    a = 0  # maintain a^2<=x.
    b = 1  # maintain b^2>x.
    while b * b <= x:
      b = 2 * b
    # Do binary search.
    while a + 1 < b:
      c = (a + b) // 2
      if c * c <= x:
        a = c
      else:
        b = c
    return a

  def dgauss(ss, num, rng):
    """Sample from discrete Gaussian.

    Args:
      ss: Variance proxy, squared scale, sigma^2.
      num: The number of samples to generate.
      rng: The random number generator.

    Returns:
      A list of discrete Gaussian samples.
    """
    ss = fractions.Fraction(ss)  # cast to rational for exact arithmetic
    assert ss > 0
    t = floorsqrt(ss) + 1
    results = []
    trials = 0
    while len(results) < num:
      trials = trials + 1
      y = dlap(t, rng)
      z = (abs(y) - ss / t)**2 / (2 * ss  )
      if bern_emab(z.numerator, z.denominator, rng) == 1:
        results.append(y)
    return results, t, trials

  rng = random.Random(seed)
  return dgauss(scale * scale, num_samples, rng)



if __name__ == "__main__":
  gamma, local_stddev = ddgauss_params(
                        q = 1,
                        epsilon = 1,
                        l2_clip_norm = 1.0,
                        bits = 64,
                        num_clients = 1,
                        dim = 1,
                        delta = 0.0001,
                        beta = 0,
                        steps = 1)
  n_samples = 100
  print(f" gamma = {gamma} \t local_stddev = {local_stddev}")
  tic = time()
  samples, _, trials = exact_sampler((23.1)**2, 1)
  toc = time()
  print(f"Avg time = {(toc-tic)/n_samples}")
  print(trials, samples)