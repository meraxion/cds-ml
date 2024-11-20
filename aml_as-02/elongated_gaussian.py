import numpy as np
from matplotlib import pyplot as plt
from metropolis_hastings import metropolis_hastings
from hamiltonian_mc import hmc

A = np.array([[250.25, -249.75], [-249.75, 250.25]])


def calc_E(x):
    return 1 / 2 * x.T @ A @ x


# proposal function that is proportional to the one we want to sample Q(x)
def proportional_function_exponent(x):
    return np.exp(-calc_E(x))


num_iterations = 10000
# where to get these from? or they're completely random?
x_init = np.array([0., 0.])
sigmas = np.linspace(0, 1, 10)
for sigma in sigmas:
    X, acceptance_ratio = metropolis_hastings(num_iterations, x_init, sigma, proportional_function_exponent)

    print(f"for sigma={sigma:.2f}, the ratio showing how many x were accepted: {acceptance_ratio:.4f}")
    plt.scatter(X[:, 0], X[:, 1])
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title(f'Elongated Gaussian, $\sigma$={sigma:.2f}, acceptance_ratio={acceptance_ratio:.4f}')

    plt.savefig(f'elongated_gaussian_{sigma}.png')
    plt.show()

eps = 0.01
tau = 100
num_iterations = 1000
y, accepts = hmc(x_init, calc_E, num_iterations, eps, tau)
plt.scatter(y[:, 0], y[:, 1])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title(rf'Elongated Gaussian HMC, $\epsilon$={eps:.2f}, $\tau$={tau} acceptance_ratio={acceptance_ratio:.4f}')
plt.show()