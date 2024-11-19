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
import numpy as np
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
xs = pd.read_csv('x.ext', sep=' ').to_numpy()[:-1].astype(float)
print(labels)

def y_function(x, w):
    wx = w @ x
    return 1 / (1 + np.exp(-wx))

def G_calc(x, w, labels, y:Callable):
    # -np.sum(label)
    # if using matrix multiplication, then we don't have to write sum'
    G = -labels * np.log(y(x, w)) + (1 - labels) * np.log(1 - y(x, w))

    return G

def E_calc(w):
    return 1 / 2 * np.sum(w ** 2)

def M(w, x, labels, a, G:Callable, E:Callable, y:Callable):
    g = G(x, w, labels, y)
    e = E(w)
    m = g + a*e
    
    return -m

alpha = 0.1

def run_hmc(labels, xs):
    
    w0 = sps.multivariate_normal().rvs(3)
    n_samples = 100
    eps = 0.01
    tau = 100

    # parameterizing W by these other values which we already know
    post_energy = lambda w: M(w, xs, labels, alpha, G_calc, E_calc, y_function)

    ws, accepts = hmc(w0, post_energy, n_samples, eps, tau)

    w1s, w2s, w3s = ws[:,0], ws[:,1], ws[:,2]

    xs = np.arange(n_samples)
    plt.plot(xs, w1s, label="w1")
    plt.plot(xs, w2s, label="w1")
    plt.plot(xs, w3s, label="w1")

    return ws, accepts

run_hmc(labels, xs)

# we don't need to use P(D|alpha) because it would cancel out in the acceptance ratio
def proportional_function_for_M(x):
    K = x.size
    print(K)
    # maybe K+1 for bias
    w = np.random.normal(K)
    exp_G = np.exp(-G_calc(x, w))
    exp_E = np.exp(-alpha * E_calc(w))
    Zw_part = (alpha / (2 * np.pi)) ** (K / 2)
    return exp_G * exp_E * Zw_part

num_iterations = 10
# random choice now
sigma=1
# what do we do with x? do we first precompute the answer with the data?
X, acceptance_ratio = metropolis_hastings(num_iterations, [1,1,1], 1, proportional_function_for_M)

print(f"for sigma={sigma:.2f}, the ratio showing how many x were accepted: {acceptance_ratio:.4f}")
plt.scatter(X[:, 0], X[:, 1])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title(f'Elongated Gaussian, $\sigma$={sigma:.2f}, acceptance_ratio={acceptance_ratio:.4f}')

plt.savefig(f'elongated_gaussian_{sigma}.png')
plt.show()
