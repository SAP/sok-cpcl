"""
Copyright (c) 2026 SAP SE or an SAP affiliate company and sok-cpcl contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

SPDX-License-Identifier: Apache-2.0
"""

import numpy as np
from utils.dp_utils import ddgauss_params
from time import time 
from fractions import Fraction
import utils.ibm_eps_rho_conv as cdp2adp

def _sample_discrete_laplace(t, shape):
  geometric_probs = 1.0 - np.exp(-1.0 / t)
  geo1 = np.random.geometric(p=geometric_probs, size=shape)
  geo2 = np.random.geometric(p=geometric_probs, size=shape)
  return np.int64(geo1 - geo2)


def _sample_bernoulli(probs, dtype = np.int64):
    # Generate uniform random variables
    uniform = np.random.uniform(size=probs.shape)
    
    # Compare with the probabilities to generate Bernoulli samples
    sample = uniform < probs
    return np.int64(sample)


def sample_discrete_gaussian(scale, shape, dtype):
  """Draw samples from discrete Gaussian, assuming scale >= 0."""
  """
    scale defined as sigma / gamma
  """
  dlap_scale = np.int64(scale)
  sq_scale = np.square(dlap_scale)
  oversample_factor = 1.5

  min_n = 1000
  target_n = np.prod(shape)
  oversample_n = oversample_factor * target_n
  draw_n = max(min_n, int(oversample_n))

  accepted_n = 0
  result = np.array([], dtype=np.int64)

  while accepted_n < target_n:
    samples = _sample_discrete_laplace(dlap_scale, shape=(draw_n,))
    z_numer = np.power((np.abs(samples) - scale), 2)
    z_denom = 2 * sq_scale
    bern_probs = np.exp(-np.divide(z_numer, z_denom))
    accept = _sample_bernoulli(bern_probs)
    accepted_samples = samples[accept == 1]
    accepted_n += accepted_samples.size
    result = np.concatenate([result, accepted_samples])
    draw_n = int((target_n - accepted_n) * oversample_factor)
    draw_n = max(min_n, draw_n)

  return result[:target_n].reshape(shape).astype(dtype)

if __name__ == "__main__":


  tic = time()
  samples = sample_discrete_gaussian(29698 //1000 , 1, dtype=np.int64)
  toc = time()
  print(f"Avg time = {(toc-tic)}")
  print(samples)

  tic = time()
  samples = sample_discrete_gaussian(1, 1, dtype=np.int64)
  toc = time()
  print(f"Avg time = {(toc-tic)}")
  print(samples)