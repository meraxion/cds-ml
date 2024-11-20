import jax
import numpy as np
import jax.numpy as jnp
import scipy.stats as sps
import pandas as pd
from matplotlib import pyplot as plt
from typing import Callable
from jax import Array

from metropolis_hastings import metropolis_hastings
from hamiltonian_mc import hmc


labels = pd.read_csv('t.ext').to_numpy()[:-1].astype(int)
labels = np.atleast_2d(labels)
xs = pd.read_csv('x.ext', sep=' ').to_numpy()[:-1].astype(float)

@jax.jit
def y_fn(x, w):
    wx = x @ w.T
    # for numerical stability
    wx = jnp.clip(wx, -50, 50)
    return 1 / (1 + jnp.exp(-wx))

@jax.jit
def G_calc(x, w, labels):
    y = y_fn(x,w)  
    y = jnp.clip(y, 1e-10, 1e10)

    G = -labels * jnp.log(y) + (1 - labels) * jnp.log(1 - y)
    return jnp.sum(G)

@jax.jit
def E_calc(w):
    return 1 / 2 * jnp.sum(w ** 2)

@jax.jit
def M(w, x, labels, a):
    g = G_calc(x, w, labels)
    e = E_calc(w)
   
    m = g + a*e
    k = len(w)
    z = jnp.power((a/2*jnp.pi), k/2)
    return -m*z

alpha = 0.1

def plots(xs, w1s, w2s, w3s, Mw, Gw):
    plt.plot(xs, w1s, label="w1")
    plt.plot(xs, w2s, label="w2")
    plt.plot(xs, w3s, label="w3")
    plt.xlabel('number of iterations')
    plt.ylabel('weight value')
    plt.legend()
    plt.title('Evolution of weights w0, w1 and w2 as a function of number of iterations')
    plt.show()

    plt.scatter(w1s, w2s, s=2)
    plt.ylabel('w2')
    plt.xlabel('w1')
    plt.title('Evolution of weights w1 and w2 in weight space')
    plt.legend()
    plt.show()

    plt.plot(xs, Gw)
    plt.xlabel('number of iterations')
    plt.ylabel('G(w)')
    plt.ylim(-15, 15)
    plt.legend()
    plt.title('The error function G(w) as a function of number of iterations')
    plt.show()

    plt.plot(xs, Mw)
    plt.xlabel('number of iterations')
    plt.ylabel('M(w)')
    plt.ylim(-15, 15)
    plt.legend()
    plt.title('The objective function M(w) as a function of number of iterations')
    plt.show()

def run_hmc(labels, xs):
    
    w0 = jnp.asarray(sps.multivariate_normal().rvs(xs.shape[1]))
    n_samples = 1000
    eps = 0.001
    tau = 50

    # parameterizing W by these other values which we already know
    post_energy = lambda w: M(w, xs, labels, alpha)

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
    exp_G = np.exp(-G_calc(x, w, labels))
    exp_E = np.exp(-alpha * E_calc(w))
    Zw_part = (alpha / (2 * np.pi)) ** (K / 2)
    return exp_G * exp_E * Zw_part, exp_G


def run_metro_hastings(labels, xs):

    num_iterations = 10000
    # assumption
    sigma = 1
    w = sps.multivariate_normal().rvs(xs.shape[1])
    X, acceptance_ratio, result_array = metropolis_hastings(num_iterations, w, sigma, proportional_function_for_M, labels, xs)

    w1s, w2s, w3s = X[:, 0], X[:, 1], X[:, 2]
    xs = np.arange(num_iterations)
    result_array = np.array(result_array)
    plots(xs, w1s, w2s, w3s, result_array[:, 0], result_array[:, 1])


# run_hmc(labels, xs)
run_metro_hastings(labels, xs)
