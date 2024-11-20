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
    return np.exp(-calc_E(x))

def main():
    """
    For Metropolis Hastings:
    - Study the acceptance ratio
    - Report the optimal values
    - Compute the mean, and compare the accuracy as a function of the computation time
    """
    num_iterations = 10000
    # where to get these from? or they're completely random?
    x_init = np.array([0., 0.])
    sigmas = np.linspace(0, 1, 10)
    MHMC_runtimes = []
    MHMC_means    = []
    MHMC_accepts  = []
    for sigma in sigmas:

        start = time.time()
        X, acceptance_ratio = metropolis_hastings(num_iterations, x_init, sigma, proportional_function_exponent)
        end = time.time()
        runtime = end - start
        MHMC_runtimes.append(runtime)
        MHMC_accepts.append(acceptance_ratio)

        mean = np.mean(X, axis=0)
        MHMC_means.append(mean)

        print(f"for sigma={sigma:.2f}, the ratio showing how many x were accepted: {acceptance_ratio:.4f}")
        plt.scatter(X[:, 0], X[:, 1])
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.title(rf'Elongated Gaussian, $\sigma$={sigma:.2f}, acceptance_ratio={acceptance_ratio:.4f}')

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

    epss = [0.1, 0.01, 0.001, 0.0001]
    taus = [5, 10, 25, 50, 100, 250]
    num_iterations = 1000

    for eps in epss:
        for tau in taus:
            start = time.time()
            y, accepts = hmc(x_init, calc_E, num_iterations, eps, tau)
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
            plt.title(rf'Elongated Gaussian HMC, $\epsilon$={eps:.2f}, $\tau$={tau} acceptance_ratio={accepts[-1]:.4f}')
            plt.show()

    MHMC_mean_errors = np.linalg.norm(np.asarray(MHMC_means)-np.zeros(2))
    HMC_mean_errors  = np.linalg.norm(np.asarray(HMC_means)-np.zeros(2))

    # compare the accuracy as a function of the computation time
    plt.scatter(MHMC_runtimes, MHMC_mean_errors, linestyle = "o", label="Metropolis Hastings", colour = "c")
    plt.scatter(HMC_runtimes, HMC_mean_errors, linestyle="o", label="Hamiltonian Monte Carlo", colour = "m")
    plt.xlabel("Empirical runtime (clock time)")
    plt.ylabel("Estimated mean error")  
    plt.title("Accuracy as a function of computation time")  
    plt.legend()
    plt.show()

    plt.clf()
    plt.figure(figsize=(10,6))

    sns.swarmplot(data=[MHMC_accepts, HMC_accepts], orient="v")
    plt.xticks([0,1], ["Metropolis Hastings", "Hamiltonian"])
    plt.ylabel("Acceptance Ratio")
    plt.title("Distribution of Acceptance Rates by Sampler")

if __name__ == "__main__":
    main()