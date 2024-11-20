import jax
import numpy as np
import jax.numpy as jnp
import pandas as pd
from matplotlib import pyplot as plt
from typing import Callable
from jax import Array
import scipy.stats as sps

from metropolis_hastings import metropolis_hastings
from hamiltonian_mc import hmc


labels = pd.read_csv('t.ext').to_numpy()[:-1].astype(int)
labels = np.atleast_2d(labels)
xs = pd.read_csv('x.ext', sep=' ').to_numpy()[:-1].astype(float)

@jax.jit
def y_function(x, w):
    wx = x @ w.T
    return 1 / (1 + jnp.exp(-wx))

@jax.jit
def G_calc(x, w, labels, y:Callable):

    G = -labels * jnp.log(y(x, w)) + (1 - labels) * jnp.log(1 - y(x, w))

    return jnp.sum(G)

@jax.jit
def E_calc(w):
    return 1 / 2 * jnp.sum(w ** 2)

@jax.jit
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

    # calculating M
    exp_G = np.exp(-G_calc(x, w, labels, y_function))
    exp_E = np.exp(-alpha * E_calc(w))
    Zw_part = (alpha / (2 * np.pi)) ** (K / 2)
    return exp_G * exp_E * Zw_part


def run_metro_hastings(labels, xs):

    num_iterations = 10000
    # assumption
    sigma = 1
    w = sps.multivariate_normal().rvs(xs.shape[1])
    X, acceptance_ratio = metropolis_hastings(num_iterations, w, sigma, proportional_function_for_M, labels, xs)

    w1s, w2s, w3s = X[:, 0], X[:, 1], X[:, 2]
    xs = np.arange(num_iterations)

    plots(xs, w1s, w2s, w3s)


run_hmc(labels, xs)
# run_metro_hastings(labels, xs)
