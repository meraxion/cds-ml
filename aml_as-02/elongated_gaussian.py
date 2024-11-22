import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from metropolis_hastings import metropolis_hastings
from hamiltonian_mc import hmc

A = np.array([[250.25, -249.75], [-249.75, 250.25]])

def calc_E(x):
    return 1 / 2 * x.T @ A @ x

# proposal function that is proportional to the one we want to sample Q(x)
def proportional_function_exponent(x, *args):
    return np.exp(-calc_E(x)), 0

"""
For Metropolis Hastings:
- Study the acceptance ratio
- Report the optimal values
- Compute the mean, and compare the accuracy as a function of the computation time
"""
num_iterations = 30000
x_init = np.array([0., 0.])
sigmas = np.logspace(-4, 1, 10)
MHMC_runtimes = []
MHMC_means    = []
MHMC_accepts  = []
for sigma in sigmas:

    start = time.time()
    X, acceptance_ratio, _ = metropolis_hastings(num_iterations, x_init, sigma, proportional_function_exponent)
    end = time.time()
    runtime = end - start
    MHMC_runtimes.append(runtime)
    MHMC_accepts.append(acceptance_ratio)

    mean = np.mean(X, axis=0)
    MHMC_means.append(mean)

    print(f"for sigma={sigma:.4f}, the ratio showing how many x were accepted: {acceptance_ratio:.4f}")
    plt.scatter(X[:, 0], X[:, 1])
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title(rf'Elongated Gaussian, $\sigma$={sigma:.4f}, acceptance_ratio={acceptance_ratio:.4f}')

    # plt.savefig(f'elongated_gaussian_{sigma}.png')
    plt.show()

"""
For Hamiltonian Monte Carlo:
- Study the acceptance ratio
- and how well the sampler covers the entire distribution,
    - by varying the step size epsilon
    - and number of leap frog steps
- report the optimal values
- compute the mean and compare the accuracy as a function of the computation time
"""

HMC_runtimes = []
HMC_means    = []
HMC_accepts  = []

epss = [0.01, 0.001, 0.0005]
# epss = [0.01, 0.001, 0.0005]
taus = [5, 10, 25, 50]
num_iterations = 1000

for eps in epss:
    for tau in taus:
        print(f"Running Hamiltonian Monte Carlo sampling run with: {num_iterations} samples, leapfrog step size {eps}, and leapfrog steps {tau}")
        start = time.time()
        y, accepts, _, _ = hmc(x_init, calc_E, calc_E, num_iterations, eps, tau)
        end = time.time()
        runtime = end - start
        HMC_runtimes.append(runtime)
        HMC_accepts.append(accepts[-1])

        # NO BURN-IN!
        mean = np.mean(y, axis=0)
        HMC_means.append(mean)

        plt.scatter(y[:, 0], y[:, 1])
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.title(rf'Elongated Gaussian HMC, $\epsilon$={eps:.5f}, $\tau$={tau} acceptance_ratio={accepts[-1]:.4f}')
        plt.show()

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