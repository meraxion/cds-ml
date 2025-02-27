"""
General Boltzmann Machine Learning
---------------------------------
This module provides a unified interface for training Boltzmann machines
using different learning algorithms:
1. Exact learning (for small models)
2. Metropolis-Hastings sampling
3. Mean field + linear response approximation
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
from functools import partial
from jaxtyping import Array
from typing import Callable, Tuple, Dict, Literal, Optional, Union
from jax.random import PRNGKey

from utils_bm import data_statistics, model_statistics, log_likelihood

@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def learn_boltzmann_machine(
    df: Array,
    key: PRNGKey,
    method: Literal['exact', 'metropolis_hastings', 'mean_field'] = 'exact',
    eta: float = 0.001,
    max_iter: int = 100_000,
    eps: float = 1e-13,
    params: Optional[Dict] = None
) -> Tuple[Array, Array, Array, int, Optional[float]]:
    """
    General Boltzmann machine learning function that supports multiple learning methods.

    Parameters
    ----------
    df : Array
        Data matrix of shape (n_neurons, n_samples) with values -1 and 1
    key : PRNGKey
        JAX random key
    method : str
        Learning method, one of 'exact', 'metropolis_hastings', 'mean_field'
    eta : float
        Learning rate
    max_iter : int
        Maximum number of iterations
    eps : float
        Convergence threshold
    params : Dict, optional
        Additional parameters for specific methods:
        - 'metropolis_hastings': {'n_samples': int}
        - 'mean_field': (no additional parameters needed)

    Returns
    -------
    w : Array
        Weight matrix
    theta : Array
        Bias terms
    logliks : Array
        Log-likelihood values for each iteration
    conv_iter : int
        Iteration at which convergence was reached (-1 if not converged)
    acceptance_ratio : float, optional
        Average acceptance ratio (only for 'metropolis_hastings')
    """
    if params is None:
        params = {}
    
    # Initialize common variables
    emp_mean, emp_corr = data_statistics(df)
    n = df.shape[0]
    key, subkey_1, subkey_2 = jr.split(key, 3)
    
    # Initialize weights and biases
    w = jr.normal(subkey_1, shape=(n, n)) * 0.001
    w = (w + w.T) / 2  # symmetric
    w = w.at[jnp.diag_indices(n)].set(0)  # 0 diagonal
    theta = jnp.squeeze(jr.normal(subkey_2, n) * 0.0001)
    
    # Initial log likelihood
    ll = log_likelihood(df, w, theta)
    
    # Choose learning method
    if method == 'exact':
        return _exact_learning(df, w, theta, emp_mean, emp_corr, ll, eta, max_iter, eps)
    elif method == 'metropolis_hastings':
        n_samples = params.get('n_samples', 1000)
        return _metropolis_hastings(df, key, w, theta, emp_mean, emp_corr, ll, n_samples, eta, max_iter, eps)
    elif method == 'mean_field':
        key, subkey_3 = jr.split(key)
        m = jr.normal(subkey_3, n) * 0.01
        return _mean_field(df, w, theta, m, emp_mean, emp_corr, ll, eta, max_iter, eps)
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'exact', 'metropolis_hastings', 'mean_field'")

def _exact_learning(
    df: Array,
    w: Array,
    theta: Array,
    emp_mean: Array,
    emp_corr: Array,
    ll: float,
    eta: float,
    max_iter: int,
    eps: float
) -> Tuple[Array, Array, Array, int, None]:
    """
    For a small BM with no hidden units, solve the fixed point equations exactly by calculating 
    free statistics in each iteration and doing gradient ascent with them.
    """
    def body_fn(carry, i):
        w, theta, done, conv_iter = carry

        def update():
            m_new, corr_new = model_statistics(w, theta)
            m_new = jnp.clip(m_new, -1 + 1e-7, 1 - 1e-7)

            w_new = w + eta * (emp_corr - corr_new)
            w_new = (w_new + w_new.T) / 2
            w_new = w_new.at[jnp.diag_indices(len(theta))].set(0)
            theta_new = jnp.squeeze(theta + eta * (emp_mean - m_new))
            loglik = log_likelihood(df, w_new, theta_new)
            w_diff = jnp.max(jnp.abs(w_new - w))
            converged = w_diff < eps
            new_conv_iter = jax.lax.cond(converged, 
                                       lambda: i, 
                                       lambda: conv_iter)
            return (w_new, theta_new, converged, new_conv_iter), loglik
            
        def no_update():
            return (w, theta, done, conv_iter), log_likelihood(df, w, theta)
        
        return jax.lax.cond(~done, update, no_update)

    init = (w, theta, False, -1)
    (w, theta, _, conv_iter), logliks = jax.lax.scan(body_fn, init, jnp.arange(max_iter))
    logliks = jnp.concatenate([jnp.array([ll]), logliks])
    return w, theta, logliks, conv_iter, None

def _mcmc_sampling(
    key: PRNGKey,
    w: Array,
    theta: Array,
    n_samples: int
) -> Tuple[Array, Array, float]:
    """
    Estimate model statistics using Metropolis-Hastings sampling.
    
    Parameters
    ----------
    key : PRNGKey
        JAX random key
    w : Array
        Weight matrix
    theta : Array
        Bias terms
    n_samples : int
        Number of samples to generate
        
    Returns
    -------
    mean : Array
        Estimated mean values
    corr : Array
        Estimated correlation matrix
    acceptance_ratio : float
        Acceptance ratio of the MCMC process
    """
    key, subkey = jr.split(key)
    current_state = jr.choice(subkey, jnp.array([-1, 1]), shape=(theta.shape[0],))
    current_state = current_state.astype(jnp.float64)
    current_energy = -jnp.sum(current_state * theta) - 0.5 * jnp.einsum("ij,i,j->", w, current_state, current_state)

    def body_fn(carry, _):
        key, current_state, current_energy, accepts = carry

        key, subkey = jr.split(key)
        # select a random spin index
        i = jr.randint(subkey, shape=(), minval=0, maxval=theta.shape[0])
        # flip it
        proposal = current_state.at[i].set(-current_state[i])
        proposal_energy = -jnp.sum(proposal * theta) - 0.5 * jnp.einsum("ij,i,j->", w, proposal, proposal)

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
    (key, _, _, accepts), samples = jax.lax.scan(body_fn, init, None, length=n_samples)

    mean, corr = data_statistics(samples.T)
    acceptance_ratio = accepts / n_samples
    
    return mean, corr, acceptance_ratio

def _metropolis_hastings(
    df: Array,
    key: PRNGKey,
    w: Array,
    theta: Array,
    emp_mean: Array,
    emp_corr: Array,
    ll: float,
    n_samples: int,
    eta: float,
    max_iter: int,
    eps: float
) -> Tuple[Array, Array, Array, int, float]:
    """
    For a BM with no hidden units, solve the fixed point equations by computing 
    free statistics using Metropolis-Hastings sampling.
    """
    all_logliks = jnp.zeros(max_iter + 1)
    all_logliks = all_logliks.at[0].set(ll)

    def body_fn(carry, i):
        w, theta, done, conv_iter, logliks, avg_accept_ratio = carry

        def update():
            m_new, corr_new, accept_ratio = _mcmc_sampling(jr.fold_in(key, i), w, theta, n_samples)
            
            new_avg_acc_ratio = (i * avg_accept_ratio + accept_ratio) / (i + 1)
            
            m_new = jnp.clip(m_new, -1 + 1e-7, 1 - 1e-7)
            
            w_new = w + eta * (emp_corr - corr_new)
            w_new = (w_new + w_new.T) / 2
            w_new = w_new.at[jnp.diag_indices(len(theta))].set(0)
            theta_new = jnp.squeeze(theta + eta * (emp_mean - m_new))
            
            loglik = log_likelihood(df, w_new, theta_new)
            new_logliks = logliks.at[i+1].set(loglik)
            
            # Check for convergence - implement convergence check inline
            window = jnp.minimum(1000, i)
            enough_samples = i >= window
            
            start_idx = jnp.maximum(0, i - window)
            recent_lls = jax.lax.dynamic_slice(new_logliks, (start_idx,), (window,))
            
            # Compute mean change in log-likelihood over window
            changes = jnp.diff(recent_lls)
            mean_change = jnp.abs(jnp.mean(changes))
            # stability of changes    
            std_change = jnp.std(changes)

            # converged if mean change is small and stable
            is_converged = jnp.logical_and(mean_change < eps, std_change < eps)
            converged = jnp.logical_and(enough_samples, is_converged)
            
            new_conv_iter = jax.lax.cond(converged,
                                       lambda: i,
                                       lambda: conv_iter)
            
            return (w_new, theta_new, converged, new_conv_iter, new_logliks, new_avg_acc_ratio), loglik
            
        def no_update():
            loglik = log_likelihood(df, w, theta)
            new_logliks = logliks.at[i+1].set(loglik)
            return (w, theta, done, conv_iter, new_logliks, avg_accept_ratio), loglik
        
        return jax.lax.cond(~done, update, no_update)

    init = (w, theta, False, -1, all_logliks, 0.)
    (w, theta, _, conv_iter, all_logliks, avg_accept_ratio), _ = jax.lax.scan(body_fn, init, jnp.arange(max_iter))
    
    return w, theta, all_logliks, conv_iter, avg_accept_ratio

def _mean_field(
    df: Array,
    w: Array,
    theta: Array,
    m: Array,
    emp_mean: Array,
    emp_corr: Array,
    ll: float,
    eta: float,
    max_iter: int,
    eps: float
) -> Tuple[Array, Array, Array, int, None]:
    """
    For a BM with no hidden units, solve the fixed point equations in the 
    mean field and linear response approximation.
    """
    delta = jnp.eye(len(m))

    def body_fn(carry, i):
        w, theta, m, done, conv_iter = carry

        def update():
            m_new = jnp.tanh(jnp.einsum("ij,j->i", w, m) + theta)
            m_new = jnp.clip(m_new, -0.9999, 0.9999)
            chi = jnp.linalg.inv(delta / (1 - jnp.pow(m_new, 2)) - w)
            corr = chi + jnp.outer(m_new, m_new)
            
            theta_new = theta + eta * (emp_mean - m_new)
            w_new = w + eta * (emp_corr - corr)
            w_new = (w_new + w_new.T) / 2
            w_new = w_new.at[jnp.diag_indices(len(theta))].set(0)
            
            loglik = log_likelihood(df, w_new, theta_new)
            w_diff = jnp.max(jnp.abs(w_new - w))
            converged = w_diff < eps
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
    
    return w, theta, logliks, conv_iter, None

def predict_pattern_rates(
    df: Array,
    w: Array,
    theta: Array
) -> Tuple[Array, Array]:
    """
    Predict spike pattern rates using the maximum entropy model.
    
    Parameters
    ----------
    df : Array
        Data matrix
    w : Array
        Weight matrix
    theta : Array
        Bias terms
        
    Returns
    -------
    predicted_rates : Array
        Predicted pattern rates
    observed_rates : Array
        Observed pattern rates
    """
    from jax.scipy.special import logsumexp
    import itertools
    
    patterns = 2 * jnp.array(list(itertools.product([0,1], repeat=w.shape[0])), dtype=jnp.float64) - 1
    lr = -jnp.sum(patterns * jnp.squeeze(theta), axis=1)
    energies = lr - 0.5 * jnp.einsum("ij,pi,pj->p", w, patterns, patterns, precision=jax.lax.Precision.HIGHEST)
    
    logZ = logsumexp(-energies)
    log_probs = -energies - logZ

    tol = 1e-6
    observed_counts = jnp.array([
        jnp.sum(jnp.all(jnp.abs(df - p.reshape(-1,1)) < tol, axis=0))
        for p in patterns
    ])
    observed_rates = observed_counts / jnp.sum(observed_counts) 

    return jnp.exp(log_probs), observed_rates

def exact_solution(
    df: Array,
    eps: float = 1e-10
) -> Tuple[Array, Array]:
    """
    Finds the exact fixed point solutions through the algebraic solution,
    as given in the slides.
    
    Parameters
    ----------
    df : Array
        Data matrix
    eps : float
        Small constant to add to diagonal for numerical stability
        
    Returns
    -------
    w : Array
        Weight matrix
    theta : Array
        Bias terms
    """
    emp_mean, emp_corr = data_statistics(df)
    # including epsilon here already to avoid numerical issues
    m = emp_mean.reshape(-1,1)
    C = emp_corr - (m @ m.T)
    C = C + eps * jnp.eye(len(m))

    w = jnp.diag(1/(1-jnp.pow(m, 2))) - jnp.linalg.inv(C)
    theta = jnp.arctanh(m) - (w @ m)

    return w, theta