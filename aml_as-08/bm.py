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

from utils_bm import data_statistics, log_likelihood, random_small_dataset, load_data, plot_loglik
# import utils_bm

"""
Exercise 1 Description:
- For small models (up to 20 spins) the computation can be done exactly. 
Make a toy problem by generating a random data set with 10-20 spins.
Define as convergence criterion that the change of the paramters of the BM is less than 10e-13. 
Demonstrate the convergence of the BM learning rule.
Show plot of the convergence of the likelihood over learning iterations.
"""
def fixed_point(key, eta:int = 0.01, max_iter:int = 10_000, eps:float = 1e-13):
  """
  For a small BM with no hidden units, solve the fixed point equations in the mean field and linear response approximation
  parameters:
  max_iter: int maximum number of fixed point iterations
  eps:float convergence criterion
  """
  key, subkey = jr.split(key)
  df = random_small_dataset(subkey)

  emp_mean, emp_cov = data_statistics(df)

  n = df.shape[0]
  key, subkey_1, subkey_2 = jr.split(key, 3)
  w = jr.normal(subkey_1, n,n) * 0.1
  w = (w + w.T)/2 # symmetric
  w = w.at[jnp.diag_indices(n)].set(0) # 0 diagonal
  theta = jr.normal(subkey_2, n) * 0.01

  # tracking loglik for plotting
  logliks = []
  logliks.append(log_likelihood(df, w))

  for t in tqdm(range(max_iter)):

    m = jnp.tanh(jnp.einsum("ij,j->i", w, m) + theta)
    delta = jnp.eye(len(m))
    chi = jnp.linalg.inv(delta/(1 - jnp.pow(m, 2)) - w)
    cov = chi + m@m.T

    theta_new = theta + eta*(emp_mean - m)
    w_new = w + eta*(emp_cov - cov)

    logliks.append(log_likelihood(df, w_new))

    if jnp.max(jnp.abs(w_new - w)) < eps:
      break
    else:
      w = w_new
      theta = theta_new

  return w, theta, logliks

"""
Exercise 2 Description:
- Apply the exact algorithm to 10 randomly selected neurons from the 160 neurons of the salamander retina, as discussed in Schneidman et al., 2006. The original data file has dimension 160 x 283041, which are 297 repeated experiments, each of which has 953 time points. Use only one of these repeats for training the BM, i.e. your data file for training has dimension 10 x 953. Reproduce Schneidman et al. 2006 fig 2a.
"""
def exact(df):
  """
  Finds the exact fixed point solutions through the algebraic solution
  """

  emp_mean, emp_cov = data_statistics(df)
  C = emp_cov - (emp_mean@emp_mean.T)
  delta = jnp.eye(len(emp_mean))

  w     = delta/(1-jnp.pow(emp_mean, 2)) - jnp.linalg.inv(C)
  theta = jnp.arctanh(emp_mean) - w@emp_mean.T

  return w, theta

def main():
  key = PRNGKey(754273565)
  key, subkey = jr.split(key)

  df, df_small = load_data()

  # Exercise 1: fixed point iteration on random data
  w_fp_iter, theta_fp_iter, logliks_fp_iter = fixed_point(subkey)
  plot_loglik(logliks_fp_iter)

  # Exercise 2: exact, direct, on subset of retinal data
  w, theta = exact(df_small)
  
  return

if __name__ == "__main__":
  main()