"""
Pseudocode:
1. Compute <s_i>_c, <s_is_j>_c from the data
2. start with a random initial state w_ij, theta_ij
3. for t = 1, 2, ... do:
  4. estimate <s_i>, <s_is_j>
  5. theta_i := theta_i + eta(<s_i>_c - <s_i>)
  6. w_ij := w_ij + eta(<s_is_j>_c - <s_is_j>)
"""
import jax
import itertools
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array
from typing import Callable
from jax.random import PRNGKey
from tqdm import tqdm
import pandas as pd

def data_statistics(df):
  mean = jnp.mean(df, axis = 1)
  cov  = jnp.cov(df)
 
  return mean, cov

def log_likelihood(df, weights):
  energy = -0.5 * jnp.einsum("ij,ni,nj->n", weights, df, df)

  all_states = jnp.array(list(itertools.product([0,1], repeat=weights.shape[0])))
  Z = jnp.sum(jnp.exp(-0.5 * jnp.einsum("ij,ni,nj->n", weights, all_states, all_states)))

  return jnp.mean(energy - jnp.log(Z))

"""
Exercise 1 Description:
- For small models (up to 20 spins) the computation can be done exactly. 
Make a toy problem by generating a random data set with 10-20 spins.
Define as convergence criterion that the change of the paramters of the BM is less than 10e-13. 
Demonstrate the convergence of the BM learning rule.
Show plot of the convergence of the likelihood over learning iterations.
"""

def random_small_dataset(key):

  key, subkey = jr.split(key)
  P = jr.uniform(subkey, minval=10, maxval=21) # num spins
  N = 500 # num 
  key, subkey = jr.split(key)
  df = jr.bernoulli(subkey, shape=(N, int(P.item())))
  
  return (1*df).T

def fixed_point(key, max_iter = 10_000, eps:float = 1e-13):
  """
  For a small BM with no hidden units, solve the fixed point equations in the mean field and linear response approximation
  parameters:
  max_iter: int maximum number of fixed point iterations
  eps:float convergence criterion
  """
  key, subkey = jr.split(key)
  df = random_small_dataset(subkey)

  n = df.shape[0]
  key, subkey_1, subkey_2 = jr.split(key, 3)
  w = jr.normal(subkey_1, n,n)*0.1
  w = (w + w.T)/2 # symmetric
  w = w.at[jnp.diag_indices(n)].set(0) # 0 diagonal
  theta = jr.normal(subkey_2, n) * 0.01

  # tracking loglik for plotting
  logliks = []
  logliks.append(log_likelihood(df, w))

  for t in tqdm(range(max_iter)):

    m = jnp.tanh(jnp.einsum("ij,j->i", w, m) + theta)



    logliks.append(log_likelihood(df, w_new))

  return

def exact(df, key):
  """
  Finds the exact fixed point solutions through the algebraic solutions
  """

  emp_mean, emp_cov = data_statistics(df)
  C = emp_cov - (emp_mean@emp_mean.T)
  delta = jnp.eye(len(emp_mean))

  w     = delta/(1-jnp.pow(emp_mean, 2)) - jnp.linalg.inv(C)
  theta = jnp.arctanh(emp_mean) - w@emp_mean.T

  return w, theta

def load_data():
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

def main():
  key = PRNGKey(754273565)
  key, subkey = jr.split(key)

  df, df_small = load_data()

  # Exercise 1: fixed point iteration on random data

  # Exercise 2: Direct on subset of retinal data
  w, theta = exact(df_small, subkey)


  
  return

if __name__ == "__main__":
  main()