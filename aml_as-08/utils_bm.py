"""
various plotting functions etc., to make the main code file look cleaner
"""

import jax
jax.config.update("jax_enable_x64", True)
import itertools
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array
from typing import Callable
from jax.random import PRNGKey
from jax.scipy.special import logsumexp
import matplotlib.pyplot as plt
from tqdm import tqdm

def data_statistics(df):
  """
  calculates the empirical statistics of a dataset df
  """
  mean = jnp.mean(df, axis = 1)
  corr  = jnp.corrcoef(df)
 
  return mean, corr

def model_statistics(w, theta):

  n = len(theta)
  patterns = 2 * jnp.array(list(itertools.product([0,1], repeat=n))) - 1

  logZ = logsumexp(-0.5 * jnp.einsum("ij,ni,nj->n", w, patterns, patterns, precision=jax.lax.Precision.HIGHEST))

  energies = -0.5*jnp.einsum("ij,ni,nj->n", w, patterns,patterns)
  lr = -jnp.sum(patterns * jnp.squeeze(theta), axis=1)
  logprobs = - energies - lr - logZ
  probs = jnp.exp(logprobs)

  mean = jnp.sum(patterns * probs[:, None], axis=0)
  correlations = jnp.einsum('ni,nj,n->ij', patterns, patterns, probs)  
  
  return mean, correlations


def log_likelihood(df, w, theta):
  """
  calculates the log likelihood of a Boltzmann machine model
  """
  # calculate the energy of each sample/experiment n
  
  energy = -jnp.sum(df.T * theta, axis=1) -0.5*jnp.einsum("ij,in,jn->n", w, df, df)

  patterns = 2 * jnp.array(list(itertools.product([0,1], repeat=w.shape[0]))) - 1
  lr = -jnp.sum(patterns * jnp.squeeze(theta), axis=1)

  all_energies = lr - 0.5 * jnp.einsum("ij,ni,nj->n", w, patterns, patterns, precision=jax.lax.Precision.HIGHEST)
  logZ = logsumexp(-all_energies)

  return jnp.sum(-energy - logZ)

def random_small_dataset(key):
  """
  makes a small random dataset with P spins and N trials
  """

  key, subkey = jr.split(key)
  P = jr.uniform(subkey, minval=10, maxval=14) # num spins
  N = 1000 # num 
  key, subkey = jr.split(key)
  df = jr.bernoulli(subkey, shape=(N, int(P.item())))
  df = 2*df - 1

  return df.T

def load_data(key):
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
  indices = []
  dat     = []
  N = df.shape[0]
  while len(dat) < 10:
    key, subkey = jr.split(key)
    i = jr.randint(subkey, (1,), 0, N).item()

    if i in indices:
      break
    else:
      dat.append(df[i, :953])
      indices.append(i)
        
  dfs = jnp.vstack(dat)

  return df, dfs

def plot_loglik(logliks, i):

  x = jnp.arange(0, i+1)
  plt.plot(x, logliks[:i+1])

  plt.title("Fixed point iteration log-likelihood")
  plt.xlabel("Iteration")
  plt.ylabel("Log likelihood")

  plt.show()

  return

def plot_loglik_comparison(logliks, i, labels):

  x = jnp.arange(0, i)

  for j in range(logliks.shape[0]):
    plt.plot(x, logliks[j, :i+1], label=labels[j])

  plt.title("Log-likelihood of learning BM")
  plt.xlabel("Iteration")
  plt.ylabel("Log likelihood")

  plt.legend()

  plt.show()

def plot_schneidman(pred, obs):
  # Plot observed vs. predicted rates
  plt.figure(figsize=(10, 6))
  plt.scatter(obs, pred, color='red')
  # Add diagonal line
  lims = [1e-10, 1e2]
  plt.plot(lims, lims, 'k-')

  plt.title("Exact model prediction vs. observed rates")
  
  # Set log scales and limits
  plt.xscale('log')
  plt.yscale('log')
  plt.xlim( [1e-4, 1e2])
  plt.ylim(lims)
  
  # Labels
  plt.xlabel(r'Observed pattern rate $s^{-1}$')
  plt.ylabel(r'Approximated pattern rate $s^{-1}$')
  
  plt.tight_layout()
  plt.show()
  return