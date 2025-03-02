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
import itertools
import jax.numpy as jnp
import jax.random as jr
from functools import partial
from jaxtyping import Array
from typing import Callable
from jax.random import PRNGKey
from jax.scipy.special import logsumexp
import matplotlib.pyplot as plt

from utils_bm import data_statistics, model_statistics, log_likelihood, random_small_dataset, load_data, plot_loglik, plot_loglik_comparison, plot_schneidman
# import utils_bm

def exercise_1(key):
  """
  Exercise 1 Description:
  For small models (up to 20 spins) the computation can be done exactly. 
  Make a toy problem by generating a random data set with 10-20 spins.
  Define as convergence criterion that the change of the paramters of the BM is less than 10e-13. 
  Demonstrate the convergence of the BM learning rule.
  Show plot of the convergence of the likelihood over learning iterations.
  """

  key, subkey = jr.split(key)
  df = random_small_dataset(subkey)
  df = jax.device_put(df)
  print("Starting Exact Fixed Point on toy data")
  key, subkey = jr.split(key)
  w_exact_learn_toy, theta_exact_learn_toy, logliks_exact_learn_toy, conv_iter_exact_learn_toy = exact_learning(df, subkey)
  if conv_iter_exact_learn_toy == -1:
    conv_iter_exact_learn_toy = logliks_exact_learn_toy.shape[0]
    print("Convergence not hit, plotting log likelihoods for all iterations")
  else:
    print(f"Converged after {conv_iter_exact_learn_toy} iterations")

  plot_loglik(logliks_exact_learn_toy, conv_iter_exact_learn_toy)

  return

def exercise_2(key, df):
  """
  Exercise 2 Description:
  - Apply the exact algorithm to 10 randomly selected neurons from the 160 neurons of the salamander retina, as discussed in Schneidman et al., 2006. The original data file has dimension 160 x 283041, which are 297 repeated experiments, each of which has 953 time points. Use only one of these repeats for training the BM, i.e. your data file for training has dimension 10 x 953. Reproduce Schneidman et al. 2006 fig 2a.
  """
  print("Starting Fixed Point on small subset of Salamander data")
  key, subkey = jr.split(key)
  w_fp_small, theta_fp_small, exact_fp_logliks, exact_iters_conv = exact_learning(df, subkey)
  if exact_iters_conv == -1:
    exact_iters_conv = exact_fp_logliks.shape[0]
    print("Convergence not hit")
  else:
    print(f"Converged after {exact_iters_conv} iterations")
  pred, obs = predict_pattern_rates(df, w_fp_small, theta_fp_small)
  plot_schneidman(pred, obs)

  return

@partial(jax.jit, static_argnums=(2, 3, 4))
def exact_learning(df, key, eta:int=0.001, max_iter:int=100_000,eps:float=1e-13):
  """
  For a small BM with no hidden units, solve the fixed point equations exactly by calculating free statistics in each iteration and doing gradient ascent with them
  """
  emp_mean, emp_corr = data_statistics(df)
  n = df.shape[0]
  key, subkey_1, subkey_2 = jr.split(key, 3)
  w = jr.normal(subkey_1, shape=(n,n)) * 0.001
  w = (w + w.T)/2 # symmetric
  w = w.at[jnp.diag_indices(n)].set(0) # 0 diagonal
  theta = jnp.squeeze(jr.normal(subkey_2, n) * 0.0001)

  ll = log_likelihood(df, w, theta)
  
  def body_fn(carry, i):

    w, theta, done, conv_iter = carry

    def update():
      m_new, corr_new = model_statistics(w, theta)
      m_new = jnp.clip(m_new, -1 + 1e-7, 1 - 1e-7)

      w_new = w + eta*(emp_corr - corr_new)
      w_new = (w_new + w_new.T)/2
      w_new = w_new.at[jnp.diag_indices(n)].set(0)
      theta_new = jnp.squeeze(theta + eta*(emp_mean - m_new))
      loglik = log_likelihood(df, w_new, theta_new)
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

def exercise_3(key):
  """
  Exercise 3 Description:
  For larger problems, implement a Metropolis Hastings sampling method using single spin flips to estimate the free statistics in each learning step. 
  Produce a plot of the likelihood over learning iterations that compares the accuracy of your sampled gradient with the exact evaluation of the gradient for small systems.
  Investigate how many Monte Carlo samples are required so that the gradients are sufficiently accurate for the BM learning.
  Since the gradient is not exact and fluctuates from iteration to iteration, a convergence criterion is less straightforward. Propose a convergence criterion.
  """
  key, subkey = jr.split(key)
  df = random_small_dataset(subkey)
  df = jax.device_put(df)

  num_samples = [10, 100, 1000, 10000]

  print("Starting Metropolis Hastings and Exact Fixed Point on toy data")
  key, subkey_exact, subkey_mh = jr.split(key, 3)
  _, _, logliks_exact_learn_toy, conv_iter_exact_learn_toy = exact_learning(df, subkey_exact)

  if conv_iter_exact_learn_toy == -1:
    conv_iter_exact_learn_toy = logliks_exact_learn_toy.shape[0]
    print("Convergence for exact not hit")
  else:
    print(f"Exact converged after {conv_iter_exact_learn_toy} iterations")

  logliks = logliks_exact_learn_toy

  for n_samples in num_samples:
    _, _, logliks_mh_learn_toy, conv_iter_mh_learn_toy, avg_accept_ratio = metropolis_hastings(df, subkey_mh, n_samples=n_samples)

    if conv_iter_mh_learn_toy == -1:
      conv_iter_mh_learn_toy = logliks_mh_learn_toy.shape[0]
      print(f"Convergence for MH at {n_samples} samples not hit, with an average acceptance ratio of {avg_accept_ratio}")
    else:
      print(f"MH with {n_samples} converged after {conv_iter_mh_learn_toy} iterations, with an average acceptance ratio of {avg_accept_ratio}")

    conv_iter_mh = jnp.maximum(conv_iter_mh, conv_iter_mh_learn_toy)
    logliks = jnp.stack([logliks, logliks_mh_learn_toy])

  conv_iter = max(conv_iter_exact_learn_toy, conv_iter_mh)
  labels = ["Exact"] + [f"MH {n_samples}" for n_samples in num_samples]
  plot_loglik_comparison(logliks, conv_iter, labels)

def mcmc_sampling(key, w, theta, n_samples):
  """
  n_samples:int number of samples to take in each iteration
  """

  key, subkey = jr.split(key)
  current_state = jr.choice(subkey, jnp.array([-1,1]), shape=(theta.shape[0],))
  current_state = current_state.astype(jnp.float64)
  current_energy = -jnp.sum(current_state * theta) - 0.5*jnp.einsum("ij,i,j->", w, current_state, current_state)

  def body_fn(carry, _):
    key, current_state, current_energy, accepts = carry

    key, subkey = jr.split(key)
    # select a random spin index
    i = jr.randint(subkey, shape = (), minval=0, maxval=theta.shape[0])
    # flip it
    proposal = current_state.at[i].set(-current_state[i])
    proposal_energy = -jnp.sum(proposal * theta) - 0.5*jnp.einsum("ij,i,j->", w, proposal, proposal)

    delta_energy = proposal_energy - current_energy

    p = jnp.exp(-delta_energy)
    acceptance_prob = jnp.where(p > 1, 1, p)

    key, subkey = jr.split(key)
    accepted = jr.uniform(subkey) < acceptance_prob

    current_state = jnp.where(accepted, proposal, current_state)
    current_energy = jnp.where(accepted, proposal_energy, current_energy)
    accepts += accepted

    return (key, current_state, current_energy, accepts), current_state
  
  init = (key, current_state, current_energy, 0)

  (key, _, _, accepts), samples = jax.lax.scan(body_fn, init, length=n_samples)

  mean, corr = data_statistics(samples.T)
  
  acceptance_ratio = accepts / n_samples
  return mean, corr, acceptance_ratio

def mh_convergence_check(logliks, i, window, eps):

  enough_samples = i >= window
      
  start_idx = jnp.maximum(0, i - window)
  recent_lls = jax.lax.dynamic_slice(logliks, (start_idx,), (window,))
  
  # Compute mean change in log-likelihood over window
  changes = jnp.diff(recent_lls)
  mean_change = jnp.abs(jnp.mean(changes))
  # stability of changes    
  std_change = jnp.std(changes)

  # converged if mean change is small and stable
  converged = jnp.logical_and(mean_change < eps, std_change < eps)
  return jnp.logical_and(enough_samples, converged)
  
@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def metropolis_hastings(df, key, n_samples:int, eta:int=0.001, max_iter:int=100_000,eps:float=1e-4):
  """
  For a BM with no hidden units, solve the fixed point equations by computing free statistics using Metropolis Hastings sampling.
  n_samples:int number of samples to take in each iteration
  """

  emp_mean, emp_corr = data_statistics(df)
  n = df.shape[0]
  key, subkey_1, subkey_2 = jr.split(key, 3)
  w = jr.normal(subkey_1, shape=(n,n)) * 0.001
  w = (w + w.T)/2 # symmetric
  w = w.at[jnp.diag_indices(n)].set(0) # 0 diagonal
  theta = jnp.squeeze(jr.normal(subkey_2, n) * 0.0001)

  ll = log_likelihood(df, w, theta)

  all_logliks = jnp.zeros(max_iter + 1)
  all_logliks = all_logliks.at[0].set(ll)

  def body_fn(carry, i):

    w, theta, done, conv_iter, logliks, avg_accept_ratio = carry

    def update():
      m_new, corr_new, accept_ratio = mcmc_sampling(jr.fold_in(key, i), w, theta, n_samples)

      new_avg_acc_ratio = (i * avg_accept_ratio + accept_ratio) / (i + 1)

      m_new = jnp.clip(m_new, -1 + 1e-7, 1 - 1e-7)

      w_new = w + eta*(emp_corr - corr_new)
      w_new = (w_new + w_new.T)/2
      w_new = w_new.at[jnp.diag_indices(n)].set(0)
      theta_new = jnp.squeeze(theta + eta*(emp_mean - m_new))
      # do convergence check on the smoothed loglikelihood instead
      loglik = log_likelihood(df, w_new, theta_new)  
      new_logliks = logliks.at[i+1].set(loglik)

      converged = mh_convergence_check(new_logliks, i, 1000, eps)

      new_conv_iter = jax.lax.cond(converged, 
                                    lambda: i, 
                                    lambda: conv_iter)
      return (w_new, theta_new, converged, new_conv_iter, new_logliks, new_avg_acc_ratio), loglik
    def no_update():
      loglik = log_likelihood(df, w, theta) 
      new_logliks = logliks.at[i+1].set(loglik)
      return (w, theta, done, conv_iter, new_logliks, avg_accept_ratio), loglik
    
    return jax.lax.cond(~done, update, no_update)

  init = (w, theta,  False, -1, all_logliks, 0.)
  (w, theta, _, conv_iter, all_logliks, avg_accept_ratio), _ = jax.lax.scan(body_fn, init, jnp.arange(max_iter))
  return w, theta, all_logliks, conv_iter, avg_accept_ratio

"""
Exercise 4
Mean Field, and comparisons with exact and MH
"""
@partial(jax.jit, static_argnums=(2, 3, 4))
def mean_field(df, key, eta:int=0.001, max_iter:int=100_000,eps:float=1e-13):
  """
  For a BM with no hidden units, solve the fixed point equations in the mean field and linear response approximation
  parameters:
  max_iter:int maximum number of fixed point iterations
  eps:float convergence criterion
  """
  emp_mean, emp_corr = data_statistics(df)
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
      corr = chi + jnp.outer(m_new, m_new)
      theta_new = theta + eta*(emp_mean - m_new)
      w_new = w + eta*(emp_corr - corr)
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
Exercise 5
Big boy dataset
"""

"""
Exercise 6
Exact solution
"""
def exact(df, eps):
  """
  Finds the exact fixed point solutions through the algebraic solution,
  as given in the slides.
  """
  emp_mean, emp_corr = data_statistics(df)
  # including epsilon here already to avoid numerical issues
  m = emp_mean.reshape(-1,1)
  C = emp_corr - (m@m.T)
  C = C + eps * jnp.eye(len(m))

  w     = jnp.diag(1/(1-jnp.pow(m, 2))) - jnp.linalg.inv(C)
  theta = jnp.arctanh(m) - (w@m)

  return w, theta


def main():
  key = PRNGKey(4536184365)
  # Exercise 1: exact fixed point iteration on random data
  key, subkey = jr.split(key)
  # exercise_1(subkey)

  # data for rest of exercises
  key, subkey = jr.split(key)
  df_sal, df_small = load_data(subkey)
  df_sal = 2*df_sal - 1
  df_small = 2*df_small - 1

  # Exercise 2: exact fixed point on subset of retinal data
  key, subkey = jr.split(key)
  # exercise_2(subkey, df_small)

  # Exercise 3: Metropolis Hastings
  key, subkey = jr.split(key)
  # exercise_3(subkey)

  # Exercise 4: MH vs. Mean Field vs. Exact
  key, subkey = jr.split(key)
  df = random_small_dataset(subkey)
  df = jax.device_put(df)

  print("Starting MH vs. Mean Field vs. Exact on toy data")
  _, _, logliks_exact_learn_toy, conv_iter_exact_learn_toy = exact_learning(df, subkey)

  if conv_iter_exact_learn_toy == -1:
    conv_iter_exact_learn_toy = logliks_exact_learn_toy.shape[0]
    print("Convergence not hit, plotting log likelihoods for all iterations")
  else:
    print(f"Converged after {conv_iter_exact_learn_toy} iterations")

  n_samples=1000
  _, _, logliks_mh_learn_toy, conv_iter_mh_learn_toy, avg_accept_ratio = metropolis_hastings(df, subkey, n_samples=n_samples)

  if conv_iter_mh_learn_toy == -1:
    conv_iter_mh_learn_toy = logliks_mh_learn_toy.shape[0]
    print(f"Convergence for MH at {n_samples} samples not hit, with an average acceptance ratio of {avg_accept_ratio}")
  else:
    print(f"MH with {n_samples} converged after {conv_iter_mh_learn_toy} iterations, with an average acceptance ratio of {avg_accept_ratio}")

  conv_iter = jnp.maximum(conv_iter_exact_learn_toy, conv_iter_mh_learn_toy)
  logliks = jnp.stack([logliks_exact_learn_toy, logliks_mh_learn_toy])

  _, _, logliks_mf_learn_toy, conv_iter_mf_learn_toy = mean_field(df, subkey)

  if conv_iter_mf_learn_toy == -1:
    conv_iter_mf_learn_toy = logliks_mf_learn_toy.shape[0]
    print(f"Convergence for Mean Field not hit.")
  else:
    print(f"Mean Field converged after {conv_iter_mf_learn_toy} iterations.")

    
  conv_iter = jnp.maximum(conv_iter, conv_iter_mf_learn_toy)
  logliks = jnp.stack([logliks, logliks_mf_learn_toy])
  labels = ["Exact", "MH", "Mean Field"]

  plot_loglik_comparison(logliks, conv_iter, labels)

  return

if __name__ == "__main__":
  main()