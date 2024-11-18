import numpy as np
from matplotlib import pyplot as plt
from metropolis_hastings import metropolis_hastings

A = np.array([[250.25, -249.75], [-249.75, 250.25]])


def calc_E(x):
    return 1 / 2 * x.T @ A @ x


# proposal function that is proportional to the one we want to sample Q(x)
def proportional_function_exponent(x):
    return np.exp(-calc_E(x))


num_iterations = 10000
# where to get these from? or they're completely random?
x_init = np.array([0, 0])

X = metropolis_hastings(num_iterations, x_init, proportional_function_exponent)

plt.scatter(X[:, 0], X[:, 1])
plt.show()
