from scipy.stats import beta as beta
from scipy.stats import norm, uniform

from scipy.special import beta as beta_func, gamma as gamma_func
from scipy.stats._distn_infrastructure import rv_continuous_frozen, rv_discrete_frozen
import matplotlib.pyplot as plt
import numpy as np

#%%% Monte Carlo

def metropolis_step(dist, x, sigma):
  """
  Random Walk Metropolis discrete Monte Carlo sampling step with symmetric (Gaussian) proposal distribution.

  :param dist: A probability distribution function, or rather, the numerator of a probability 
  :param x: the current state
  :param sigma: the variance of the normal distribution used for generating candidate steps
  """
    # Generate a candidate
  candidate = norm.rvs(loc=x, scale=sigma**2, size=1)

  # Accept/reject
  acceptance_probability = np.min(1, dist(x)/dist(candidate))
  u = uniform.rvs()
  if acceptance_probability >= u:
    value = candidate
    accepted = True
  else:
    value = x
    accepted = False

  return value, accepted

def metropolis_sampler(dist, initial_value, n=1000,
                        sigma=1, burnin=200, lag=1):
  """
  Random Walk discrete Metropolis Monte Carlo algorithm
  """

  # Initialize
  my_list = []
  current_value = initial_value

  for i in range(burnin):
    current_value, accepted = metropolis_step(dist, current_value, sigma)
    

  for i in range(n):
    for j in range(lag):
      current_value, accepted = metropolis_step(dist, current_value, sigma)
  
    my_list.append((current_value, accepted))

  return my_list

def MMC_example():
  def my_dist(x):
    return np.exp(-x**2)*(2*np.sin(5*x)+np.sin(2*x))
  
  init = 0
  metropolis_sampler(my_dist, init)
