"""
various plotting functions etc., to make the main code file look cleaner
"""

import jax
import itertools
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array
from typing import Callable
from jax.random import PRNGKey
import matplotlib.pyplot as plt
from tqdm import tqdm

def data_statistics(df):
  """
  calculates the empirical statistics of a dataset df
  """
  mean = jnp.mean(df, axis = 1)
  cov  = jnp.cov(df)
 
  return mean, cov

def log_likelihood(df, weights):
  """
  calculates the log likelihood of a Boltzmann machine model
  """
  energy = -0.5 * jnp.einsum("ij,ni,nj->n", weights, df, df)

  all_states = jnp.array(list(itertools.product([0,1], repeat=weights.shape[0])))
  Z = jnp.sum(jnp.exp(-0.5 * jnp.einsum("ij,ni,nj->n", weights, all_states, all_states)))

  return jnp.mean(energy - jnp.log(Z))

def random_small_dataset(key):
  """
  makes a small random dataset with P spins and N trials
  """

  key, subkey = jr.split(key)
  P = jr.uniform(subkey, minval=10, maxval=21) # num spins
  N = 500 # num 
  key, subkey = jr.split(key)
  df = jr.bernoulli(subkey, shape=(N, int(P.item())))
  
  return (1*df).T

def load_data():
  """
  loads the salamander retina data
  """
  # Initialize list to hold each row
  rows = []

  with open('bint.txt', 'r') as f:
    for line in f:
        # Parse each line into a numpy array (adjust 'sep' if needed)
        rows.append(jnp.fromstring(line, sep=' '))

  # Combine rows into a 2D array
  df = jnp.vstack(rows)
  print(df.shape)

  print("Data loaded")
  df = jnp.array(df)
  # restrict the dataset to a smaller one:
  dfs = df[0:10, 0:953]

  return df, dfs

def plot_loglik(logliks):

  n = len(logliks)
  x = jnp.arange(0, n)
  plt.plot(x, logliks)

  plt.show()

  return