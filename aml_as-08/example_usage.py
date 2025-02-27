"""
Example usage of the generalized Boltzmann machine learning function.
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
from jax.random import PRNGKey
import matplotlib.pyplot as plt

from utils_bm import random_small_dataset, load_data, plot_loglik, plot_loglik_comparison, plot_schneidman
from general_bm import learn_boltzmann_machine, predict_pattern_rates, exact_solution

def example_exact_learning():
    """Example using exact learning on a small toy dataset."""
    key = PRNGKey(42)
    key, subkey = jr.split(key)
    df = random_small_dataset(subkey)
    
    print("Starting Exact learning on toy data")
    key, subkey = jr.split(key)
    
    w, theta, logliks, conv_iter, _ = learn_boltzmann_machine(
        df, 
        subkey, 
        method='exact',
        eta=0.001,
        max_iter=10000,
        eps=1e-13
    )
    
    if conv_iter == -1:
        conv_iter = logliks.shape[0] - 1
        print("Convergence not hit, plotting log likelihoods for all iterations")
    else:
        print(f"Converged after {conv_iter} iterations")
    
    plot_loglik(logliks, conv_iter)
    
    return w, theta, logliks, conv_iter

def example_mh_learning():
    """Example using Metropolis-Hastings learning on a small toy dataset."""
    key = PRNGKey(42)
    key, subkey = jr.split(key)
    df = random_small_dataset(subkey)
    
    print("Starting Metropolis-Hastings learning on toy data")
    key, subkey = jr.split(key)
    
    w, theta, logliks, conv_iter, avg_accept_ratio = learn_boltzmann_machine(
        df, 
        subkey, 
        method='metropolis_hastings',
        eta=0.001,
        max_iter=10000,
        eps=1e-4,
        params={'n_samples': 1000}
    )
    
    if conv_iter == -1:
        conv_iter = logliks.shape[0] - 1
        print(f"Convergence not hit, with an average acceptance ratio of {avg_accept_ratio}")
    else:
        print(f"Converged after {conv_iter} iterations, with an average acceptance ratio of {avg_accept_ratio}")
    
    plot_loglik(logliks, conv_iter)
    
    return w, theta, logliks, conv_iter

def example_mean_field_learning():
    """Example using mean field learning on a small toy dataset."""
    key = PRNGKey(42)
    key, subkey = jr.split(key)
    df = random_small_dataset(subkey)
    
    print("Starting Mean Field learning on toy data")
    key, subkey = jr.split(key)
    
    w, theta, logliks, conv_iter, _ = learn_boltzmann_machine(
        df, 
        subkey, 
        method='mean_field',
        eta=0.001,
        max_iter=10000,
        eps=1e-13
    )
    
    if conv_iter == -1:
        conv_iter = logliks.shape[0] - 1
        print("Convergence not hit")
    else:
        print(f"Converged after {conv_iter} iterations")
    
    plot_loglik(logliks, conv_iter)
    
    return w, theta, logliks, conv_iter

def example_compare_methods():
    """Compare all three methods on the same dataset."""
    key = PRNGKey(42)
    key, subkey = jr.split(key)
    df = random_small_dataset(subkey)
    
    # Run exact learning
    key, subkey_exact = jr.split(key)
    w_exact, theta_exact, logliks_exact, conv_iter_exact, _ = learn_boltzmann_machine(
        df, 
        subkey_exact, 
        method='exact',
        max_iter=10000
    )
    
    # Run Metropolis-Hastings
    key, subkey_mh = jr.split(key)
    w_mh, theta_mh, logliks_mh, conv_iter_mh, _ = learn_boltzmann_machine(
        df, 
        subkey_mh, 
        method='metropolis_hastings',
        max_iter=10000,
        eps=1e-4,
        params={'n_samples': 1000}
    )
    
    # Run Mean Field
    key, subkey_mf = jr.split(key)
    w_mf, theta_mf, logliks_mf, conv_iter_mf, _ = learn_boltzmann_machine(
        df, 
        subkey_mf, 
        method='mean_field',
        max_iter=10000
    )
    
    # Prepare for plotting
    max_iter = max(
        conv_iter_exact if conv_iter_exact != -1 else logliks_exact.shape[0] - 1,
        conv_iter_mh if conv_iter_mh != -1 else logliks_mh.shape[0] - 1,
        conv_iter_mf if conv_iter_mf != -1 else logliks_mf.shape[0] - 1
    )
    
    logliks = jnp.stack([
        logliks_exact[:max_iter+1], 
        logliks_mh[:max_iter+1], 
        logliks_mf[:max_iter+1]
    ])
    
    labels = ["Exact", "Metropolis-Hastings", "Mean Field"]
    plot_loglik_comparison(logliks, max_iter+1, labels)
    
    return (w_exact, theta_exact), (w_mh, theta_mh), (w_mf, theta_mf)

def example_salamander_data():
    """Example on salamander retina data."""
    key = PRNGKey(42)
    key, subkey = jr.split(key)
    
    # Load salamander data
    df_full, df_small = load_data(subkey)
    
    # Convert from [0,1] to [-1,1]
    df_small = 2 * df_small - 1
    
    print("Running exact learning on small subset of Salamander data")
    key, subkey = jr.split(key)
    
    w, theta, logliks, conv_iter, _ = learn_boltzmann_machine(
        df_small, 
        subkey, 
        method='exact',
        eta=0.001,
        max_iter=10000,
        eps=1e-13
    )
    
    if conv_iter == -1:
        conv_iter = logliks.shape[0] - 1
        print("Convergence not hit")
    else:
        print(f"Converged after {conv_iter} iterations")
    
    plot_loglik(logliks, conv_iter)
    
    # Plot Schneidman-style figure
    pred, obs = predict_pattern_rates(df_small, w, theta)
    plot_schneidman(pred, obs)
    
    return w, theta, logliks, conv_iter

def example_exact_solution():
    """Example using the direct algebraic solution."""
    key = PRNGKey(42)
    key, subkey = jr.split(key)
    df = random_small_dataset(subkey)
    
    print("Computing exact solution using algebraic method")
    w, theta = exact_solution(df)
    
    # Calculate log-likelihood to verify solution
    from utils_bm import log_likelihood
    ll = jnp.array([log_likelihood(df, w, theta)])
    print(f"Log-likelihood of exact solution: {ll[0]}")
    
    return w, theta, ll

def main():
    print("\n====== Example 1: Exact Learning ======")
    example_exact_learning()
    
    print("\n====== Example 2: Metropolis-Hastings Learning ======")
    example_mh_learning()
    
    print("\n====== Example 3: Mean Field Learning ======")
    example_mean_field_learning()
    
    print("\n====== Example 4: Comparing All Methods ======")
    example_compare_methods()
    
    print("\n====== Example 5: Salamander Retina Data ======")
    example_salamander_data()
    
    print("\n====== Example 6: Exact Algebraic Solution ======")
    example_exact_solution()

if __name__ == "__main__":
    main()