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
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import itertools
from functools import partial
from jaxtyping import Array
from typing import Callable
from jax.random import PRNGKey
from jax.scipy.special import logsumexp
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils_bm import data_statistics, log_likelihood, random_small_dataset, load_data, plot_loglik, plot_schneidman
# import utils_bm

"""
Exercise 1 Description:
- For small models (up to 20 spins) the computation can be done exactly. 
Make a toy problem by generating a random data set with 10-20 spins.
Define as convergence criterion that the change of the paramters of the BM is less than 10e-13. 
Demonstrate the convergence of the BM learning rule.
Show plot of the convergence of the likelihood over learning iterations.
"""
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
  cov = correlations - jnp.outer(mean, mean)
  
  return mean, cov

@partial(jax.jit, static_argnums=(2, 3, 4))
def exact_learning(df, key, eta:int=0.001, max_iter:int=100_000,eps:float=1e-13):
  """
  For a small BM with no hidden units, solve the fixed point equations exactly by calculating free statistics in each iteration and doing gradient ascent with them
  """
  emp_mean, emp_cov = data_statistics(df)
  n = df.shape[0]
  key, subkey_1, subkey_2 = jr.split(key, 3)
  w = jr.normal(subkey_1, shape=(n,n)) * 0.001
  w = (w + w.T)/2 # symmetric
  w = w.at[jnp.diag_indices(n)].set(0) # 0 diagonal
  theta = jnp.squeeze(jr.normal(subkey_2, n) * 0.001)

  ll = log_likelihood(df, w, theta)
  
  def body_fn(carry, i):

    w, theta, done, conv_iter = carry

    def update():
      m_new, cov_new = model_statistics(w, theta)
      m_new = jnp.clip(m_new, -1 + 1e-7, 1 - 1e-7)
      cov_new = jnp.clip(cov_new, -1 + 1e-7, 1 - 1e-7)

      w_new = w + eta*(emp_cov - cov_new)
      w_new = (w_new + w_new.T)/2
      w_new = w_new.at[jnp.diag_indices(n)].set(0)
      theta_new = jnp.squeeze(theta + eta*(emp_mean - m_new))
      loglik = log_likelihood(df, w_new, theta_new)
      converged = (jnp.max(jnp.abs(w_new - w)) < eps)
      w_diff = jnp.linalg.norm(w_new - w, ord='fro')
      converged = w_diff < eps
      new_conv_iter = jax.lax.cond(converged, 
                                    lambda: i, 
                                    lambda: conv_iter)
      return (w_new, theta_new, converged, new_conv_iter), loglik
    def no_update():
      return (w, theta, done, conv_iter), log_likelihood(df, w, theta)
    
    return jax.lax.cond(~done, update, no_update)

  init = (w, theta,  False, -1)
  (w, theta, _, conv_iter), logliks = jax.lax.scan(body_fn, init, jnp.arange(max_iter))
  logliks = jnp.concatenate([jnp.array([ll]), logliks])
  return w, theta, logliks, conv_iter

@partial(jax.jit, static_argnums=(2, 3, 4))
def fixed_point(df, key, eta:int = 0.01, max_iter:int = 100_000, eps:float = 1e-13):
  """
  For a small BM with no hidden units, solve the fixed point equations in the mean field and linear response approximation
  parameters:
  max_iter: int maximum number of fixed point iterations
  eps:float convergence criterion
  """
  emp_mean, emp_cov = data_statistics(df)
  n = df.shape[0]
  key, subkey_1, subkey_2, subkey_3 = jr.split(key, 4)
  w = jr.normal(subkey_1, shape=(n,n)) * 0.1
  w = (w + w.T)/2 # symmetric
  w = w.at[jnp.diag_indices(n)].set(0) # 0 diagonal
  theta = jr.normal(subkey_2, n) * 0.01
  m = jr.normal(subkey_3, n) * 0.01
  delta = jnp.eye(len(m))

  ll = log_likelihood(df, w, theta)

  def body_fn(carry, i):

    w, theta, m, done, conv_iter = carry

    def update():
      m_new = jnp.tanh(jnp.einsum("ij,j->i", w, m) + theta)
      m_new = jnp.clip(m_new, -0.9999, 0.9999)
      chi = jnp.linalg.inv(delta/(1 - jnp.pow(m_new, 2)) - w)
      cov = chi + jnp.outer(m_new, m_new)
      theta_new = theta + eta*(emp_mean - m_new)
      w_new = w + eta*(emp_cov - cov)
      w_new = (w_new + w_new.T)/2
      w_new = w_new.at[jnp.diag_indices(n)].set(0)
      loglik = log_likelihood(df, w_new, theta_new)
      converged = (jnp.max(jnp.abs(w_new - w)) < eps) & (jnp.max(jnp.abs(theta_new - theta)) < eps)
      new_conv_iter = jax.lax.cond(converged, 
                                    lambda: i, 
                                    lambda: conv_iter)
      return (w_new, theta_new, m_new, converged, new_conv_iter), loglik
    def no_update():
      return (w, theta, m, done, conv_iter), log_likelihood(df, w, theta)
    
    return jax.lax.cond(~done, update, no_update)

  init = (w, theta, m, False, -1)
  (w, theta, m, _, conv_iter), logliks = jax.lax.scan(body_fn, init, jnp.arange(max_iter))
  logliks = jnp.concatenate([jnp.array([ll]), logliks])
  return w, theta, logliks, conv_iter

"""
Exercise 2 Description:
- Apply the exact algorithm to 10 randomly selected neurons from the 160 neurons of the salamander retina, as discussed in Schneidman et al., 2006. The original data file has dimension 160 x 283041, which are 297 repeated experiments, each of which has 953 time points. Use only one of these repeats for training the BM, i.e. your data file for training has dimension 10 x 953. Reproduce Schneidman et al. 2006 fig 2a.
"""
def predict_pattern_rates(df, w, theta):
  """
  Predict spike pattern rates using the maximum entropy model.
  """
  patterns = 2 * jnp.array(list(itertools.product([0,1], repeat=w.shape[0])), dtype=jnp.float64) - 1
  lr = -jnp.sum(patterns * jnp.squeeze(theta), axis=1)
  energies = lr - 0.5 * jnp.einsum("ij,pi,pj->p", w, patterns, patterns, precision=jax.lax.Precision.HIGHEST)
  # energies = jnp.clip(energies, -10000, 10000)
  
  logZ = logsumexp(-energies)
  log_probs = -energies - logZ

  tol = 1e-6
  observed_counts = jnp.array([
      jnp.sum(jnp.all(jnp.abs(df - p.reshape(-1,1)) < tol, axis=0))
      for p in patterns
    ])
  observed_rates = observed_counts / jnp.sum(observed_counts) 

  return jnp.exp(log_probs), observed_rates


"""
Exercise 6
"""
def exact(df, eps):
  """
  Finds the exact fixed point solutions through the algebraic solution,
  as given in the slides.
  """
  emp_mean, emp_cov = data_statistics(df)
  # including epsilon here already to avoid numerical issues
  m = emp_mean.reshape(-1,1)
  C = emp_cov - (m@m.T)
  C = C + eps * jnp.eye(len(m))

  w     = jnp.diag(1/(1-jnp.pow(m, 2))) - jnp.linalg.inv(C)
  theta = jnp.arctanh(m) - (w@m)

  return w, theta


def main():
  key = PRNGKey(54267852)
  # Exercise 1: exact fixed point iteration on random data
  key, subkey = jr.split(key)
  df = random_small_dataset(subkey)
  df = jax.device_put(df)
  print("Starting Exact Fixed Point on toy data")
  key, subkey = jr.split(key)
  w_exact_learn_toy, theta_exact_learn_toy, logliks_exact_learn_toy, conv_iter_exact_learn_toy = exact_learning(df, subkey)
  plot_loglik(logliks_exact_learn_toy, conv_iter_exact_learn_toy)

  # Exercise 2: exact fixed point on subset of retinal data
  key, subkey = jr.split(key)
  df_sal, df_small = load_data(subkey)
  df_sal = 2*df_sal - 1
  df_small = 2*df_small - 1
  print("Starting Fixed Point on small subset of Salamander data")
  key, subkey = jr.split(key)
  w_fp_small, theta_fp_small, _, _ = exact_learning(df_small, subkey)
  pred, obs = predict_pattern_rates(df_small, w_fp_small, theta_fp_small)
  plot_schneidman(pred, obs)
 
  return

if __name__ == "__main__":
  main()