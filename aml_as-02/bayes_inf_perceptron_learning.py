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

from metropolis_hastings import metropolis_hastings

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


def G_calc(x, w):
    # -np.sum(label)
    # if using matrix multiplication, then we don't have to write sum'
    G = -labels * np.log(y_function(x, w)) + (1 - labels) * np.log(1 - y_function(x, w))

    return G


def E_calculation(w):
    return 1 / 2 * np.sum(w ** 2)


alpha = 1


# we don't need to use P(D|alpha) because it would cancel out in the acceptance ratio
def proportional_function_for_M(x):
    K = x.size
    print(K)
    # maybe K+1 for bias
    w = np.random.normal(K)
    exp_G = np.exp(-G_calc(x, w))
    exp_E = np.exp(-alpha * E_calculation(w))
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