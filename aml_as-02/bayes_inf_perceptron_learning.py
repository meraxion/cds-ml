# In this exercise you are asked to sample from the posterior of learning problem. The
# learning task is the perceptron/logistic regression classification problem as explained in
# Mackay chapter 39 and 41. The sampling methods are the Metropolis Hasting method
# and the Hamilton Monte Carlo method as described in MacKay chapter 29 and 30. The
# data are given by the files x.ext (input patterns) and t.ext (output label).
# Write a computer program to sample from the distribution p(w|D, α) as given by
# MacKay Eqs. 41.8-10 using the Metropolis Hasting algorithm. Do the same using the
# Hamilton Monte Carlo method. For both methods, reproduce plots similar to fig. 41.5
# and estimate the burn in time that is required before the sampler reaches the equilibrium
# distribution. Investigate the acceptance ratio for both methods and try to optimize this
# by varying the proposal distribution, the step size  in HMC and the number of leap
# frog steps τ.
import jax
import numpy as np
import jax.numpy as jnp
import pandas as pd
from matplotlib import pyplot as plt
from typing import Callable
import scipy.stats as sps

from metropolis_hastings import metropolis_hastings
from hamiltonian_mc import hmc

# chapter 29, 30 (page 38 something look for A matrix) <- metropolis and hasting explained
# 39 and 41 learning task is the perceptron/logistic regression classification problem
#  weight decay rates α = 0.01, 0.1,
#  We modify the objective function to: M(w) = G(w) + αEW (w) (39.22)
# where the simplest choice of regularizer is the weight decay regularizer (39.23)

# 1/Zw(alpha) = (a/2pi)**(K/2)
# how do we get P(D|alpha)
labels = pd.read_csv('t.ext').to_numpy()[:-1].astype(int)
labels = np.atleast_2d(labels)
xs = pd.read_csv('x.ext', sep=' ').to_numpy()[:-1].astype(float)

def y_function(x, w):
    wx = x @ w.T
    return 1 / (1 + jnp.exp(-wx))

def G_calc(x, w, labels, y:Callable):
    # -np.sum(label)
    # if using matrix multiplication, then we don't have to write sum'
    G = -labels * jnp.log(y(x, w)) + (1 - labels) * jnp.log(1 - y(x, w))

    return jnp.sum(G)

def E_calc(w):
    return 1 / 2 * jnp.sum(w ** 2)

def M(w, x, labels, a, G:Callable, E:Callable, y:Callable):
    g = G(x, w, labels, y)
    e = E(w)
    m = g + a*e

    k = len(w)

    z = jnp.power((a/2*jnp.pi), k/2)
    
    return -m*z

alpha = 0.1


def plots(xs, w1s, w2s, w3s):

    plt.plot(xs, w1s, label="w1")
    plt.plot(xs, w2s, label="w2")
    plt.plot(xs, w3s, label="w3")
    plt.legend()
    plt.show()

def run_hmc(labels, xs):
    
    w0 = sps.multivariate_normal().rvs(xs.shape[1])
    n_samples = 100
    eps = 0.01
    tau = 100

    # parameterizing W by these other values which we already know
    post_energy = lambda w: M(w, xs, labels, alpha, G_calc, E_calc, y_function)

    ws, accepts = hmc(w0, post_energy, n_samples, eps, tau)

    w1s, w2s, w3s = ws[:,0], ws[:,1], ws[:,2]
    xs = np.arange(n_samples)

    plots(xs, w1s, w2s, w3s)

    return ws, accepts

# we don't need to use P(D|alpha) because it would cancel out in the acceptance ratio
def proportional_function_for_M(w, *args):
    r = args[0]
    labels, x = r[0]
    K = x.size
    # maybe K+1 for bias
    # w = sps.multivariate_normal().rvs(xs.shape[1])
    # w = np.random.multivariate_normal(x.shape, np.eye(K))
    exp_G = np.exp(-G_calc(x, w, labels, y_function))
    exp_E = np.exp(-alpha * E_calc(w))
    Zw_part = (alpha / (2 * np.pi)) ** (K / 2)
    return exp_G * exp_E * Zw_part


def run_metro_hastings(labels, xs):

    num_iterations = 10000
    # random choice now
    sigma=1
    w = sps.multivariate_normal().rvs(xs.shape[1])
    X, acceptance_ratio = metropolis_hastings(num_iterations, w, 1, proportional_function_for_M, labels, xs)
    # X, acceptance_ratio = metropolis_hastings(num_iterations, np.array([1,1,1]), 1, proportional_function_for_M, labels)

    w1s, w2s, w3s = X[:, 0], X[:, 1], X[:, 2]
    xs = np.arange(num_iterations)

    plots(xs, w1s, w2s, w3s)

    # print(f"for sigma={sigma:.2f}, the ratio showing how many x were accepted: {acceptance_ratio:.4f}")
    # plt.scatter(X[:, 0], X[:, 1])
    # plt.xlabel('$x_1$')
    # plt.ylabel('$x_2$')
    # plt.title(rf'Bayesian inference problem, $\sigma$={sigma:.2f}, acceptance_ratio={acceptance_ratio:.4f}')
    #
    # plt.savefig(f'bayes_inf{sigma}.png')
    # plt.show()


run_hmc(labels, xs)
# run_metro_hastings(labels, xs)
