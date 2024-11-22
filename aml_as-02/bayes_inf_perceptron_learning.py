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
import time

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
    result = -m * z
    return result

alpha = 0.1

# def plots(xs, w1s, w2s, w3s):
def plots(xs, w1s, w2s, w3s, Mw, Gw, title):
    print(Mw)
    print(Gw)
    plt.plot(xs, w1s, label="w1")
    plt.plot(xs, w2s, label="w2")
    plt.plot(xs, w3s, label="w3")
    plt.xlabel('number of iterations')
    plt.ylabel('weight value')

    plt.title(title+',\n Evolution of weights w0, w1 and w2 as a function of number of iterations')
    plt.savefig(title + 'Evolution of weights w0, w1 and w2 as a function of number of iterations.png')
    plt.show()

    plt.scatter(w1s, w2s, s=2)
    plt.ylabel('w2')
    plt.xlabel('w1')
    plt.title(title+',\n Evolution of weights w1 and w2 in weight space')

    plt.savefig(title+'Evolution of weights w1 and w2 in weight space.png')
    plt.show()


    plt.plot(xs, Gw)
    plt.xlabel('number of iterations')
    plt.ylabel('G(w)')


    plt.title(title+',\n The error function G(w) as a function of number of iterations')
    plt.savefig(title + 'The error function G(w) as a function of number of iterations.png')
    plt.show()

    plt.plot(xs, Mw)
    plt.xlabel('number of iterations')
    plt.ylabel('M(w)')


    plt.title(title+',\n The objective function M(w) as a function of number of iterations')
    plt.savefig(title + 'The objective function M(w) as a function of number of iterations.png')
    plt.show()


HMC_runtimes = []
HMC_means = []
HMC_accepts = []
def run_hmc(labels, xs):

    w0 = jnp.asarray(sps.multivariate_normal().rvs(xs.shape[1]))
    n_samples = 1500

    epss = [0.01, 0.001, 0.0005]
    taus = [5, 10, 25, 50, 100]

    for eps in epss:
        for tau in taus:
            print(
                f"Running Hamiltonian Monte Carlo sampling run with: {n_samples} samples, leapfrog step size {eps}, and leapfrog steps {tau}")
            start = time.time()

            # parameterizing W by these other values which we already know
            g = lambda w: G_calc(xs, w, labels)
            post_energy = lambda w: M(w, xs, labels, alpha)

            ws, accepts, M_result, G_result = hmc(w0, post_energy, g, n_samples, eps, tau)

            end = time.time()
            runtime = end - start
            HMC_runtimes.append(runtime)
            HMC_accepts.append(accepts[-1])

            # NO BURN-IN!
            mean = np.mean(ws, axis=0)
            HMC_means.append(mean)


            w1s, w2s, w3s = ws[:,0], ws[:,1], ws[:,2]
            xss = np.arange(n_samples)

            plots(xss, w1s, w2s, w3s, M_result, G_result, f'HMC with eps={eps} and tau={tau} and acceptance rate={accepts:.4f}')



    return ws, accepts


@jax.jit
def proportional_function_for_M(w, x, labels, a):
    g = G_calc(x, w, labels)
    e = E_calc(w)

    m = g + a * e
    k = len(w)
    z = jnp.power((a / 2 * jnp.pi), k / 2)
    result = -m * z
    return result, -g

MHMC_runtimes = []
MHMC_means = []
MHMC_accepts = []

def run_metro_hastings(labels, xs):

    num_iterations = 40000
    sigmas = np.logspace(-4, 0, 9)

    for sigma in sigmas:
        start = time.time()
        w = sps.multivariate_normal().rvs(xs.shape[1])
        X, acceptance_ratio, result_array = metropolis_hastings(num_iterations, w, sigma, proportional_function_for_M, xs, labels, alpha)
        end = time.time()
        runtime = end - start
        MHMC_runtimes.append(runtime)
        MHMC_accepts.append(acceptance_ratio)

        mean = np.mean(X, axis=0)
        MHMC_means.append(mean)
        print(f"for sigma={sigma:.2f}, the ratio showing how many x were accepted: {acceptance_ratio:.4f}")

        w1s, w2s, w3s = X[:, 0], X[:, 1], X[:, 2]
        xss = np.arange(num_iterations)
        result_array = np.array(result_array)
        plots(xss, w1s, w2s, w3s, result_array[:, 0], result_array[:, 1], f'Metropolis Hastings with sigma={sigma:.4f} and acceptance rate={acceptance_ratio:.4f}')


run_metro_hastings(labels, xs)
run_hmc(labels, xs)



MHMC_mean_errors = (np.asarray(MHMC_means) - np.zeros((len(MHMC_means), 2)))**2
HMC_mean_errors  = (np.asarray(HMC_means) - np.zeros((len(HMC_means), 2)))**2

# compare the accuracy as a function of the computation time
plt.scatter(MHMC_runtimes, MHMC_mean_errors[:, 0], marker='o', label="MH - dim 1", color='cyan', alpha = 0.5)
plt.scatter(MHMC_runtimes, MHMC_mean_errors[:, 1], marker='s', label="MH - dim 2", color='blue', alpha = 0.5)
plt.scatter(HMC_runtimes, HMC_mean_errors[:,0], marker="o", label="HMC - dim 1", color = "magenta", alpha = 0.5)
plt.scatter(HMC_runtimes, HMC_mean_errors[:,1], marker="s", label="HMC - dim 2", color = "yellow", alpha = 0.5)
plt.xlabel("Empirical runtime (clock time)")
plt.ylabel("Estimated mean error")
plt.xscale("log")
plt.yscale("log")
plt.title("Accuracy as a function of computation time")
plt.legend()
plt.show()

plt.clf()

plt.figure(figsize=(10,6))
x_mh = np.random.normal(-0.3, 0.04, size=len(MHMC_accepts))
x_hmc = np.random.normal(0.3, 0.04, size=len(HMC_accepts))

plt.scatter(x_mh, MHMC_accepts, alpha=0.5, label='Metropolis Hastings')
plt.scatter(x_hmc, HMC_accepts, alpha=0.5, label='Hamiltonian')
plt.xticks([-0.3, 0.3], ["Metropolis Hastings", "Hamiltonian"])
plt.xlim(-0.5, 0.5)
plt.ylabel("Acceptance Ratio")
plt.title("Distribution of Acceptance Rates by Sampler")
plt.legend()