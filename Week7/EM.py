import pandas as pd
from scipy.stats import multivariate_normal

import numpy as np
import matplotlib.pyplot as plt


def plot_gaussian(mu, cov, ax, color):
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    # Define the multivariate normal distributions
    rv1 = multivariate_normal(mu, cov)

    # Calculate the PDF (probability density function) over the grid
    Z1 = rv1.pdf(pos)

    ax.contour(X, Y, Z1, levels=[0.1], colors=color)


def plot_data(data, mu, cov):
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], s=10, c='green', label='Samples')

    plot_gaussian(mu[0], cov[0], ax, color='blue')
    plot_gaussian(mu[1], cov[1], ax, color='red')

    # Set the plot limits and labels
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')

    plt.show()

def plot_classified_data(data, mu, cov, r_k):
    fig, ax = plt.subplots()

    # Assign each point to the cluster with the highest responsibility
    cluster_assignments = np.argmax(r_k, axis=0)
    cluster_0 = data[cluster_assignments == 0]
    cluster_1 = data[cluster_assignments == 1]

    # Plot colored sata points
    ax.scatter(cluster_0[:, 0], cluster_0[:, 1], s=20, c='blue', label='Cluster 0')
    ax.scatter(cluster_1[:, 0], cluster_1[:, 1], s=20, c='red', label='Cluster 1')

    # Plot the Gaussian distributions as contours
    plot_gaussian(mu[0], cov[0], ax, color='blue')
    plot_gaussian(mu[1], cov[1], ax, color='red')

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')

    plt.legend()
    plt.show()


def prep_data():
    data = pd.read_csv('old_faithful_geyser.txt', delim_whitespace=True, header=None, index_col=0).to_numpy()

    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)

    # Perform the standardization
    standardized_data = (data - mean) / std_dev
    return standardized_data


def gaussian_with_pi_k(pi_probability, big_sigma, a, X):
    """
    Calculates the joint probability of the prior for the classes and the multivariate Gaussian for the data for that class

    :param pi_probability: prior for the classes
    :param big_sigma: covariance matrix for the Gaussian
    :param a: mean vector for the Gaussian
    :param X: data points
    """
    return pi_probability * multivariate_normal.pdf(X, mean=a, cov=big_sigma)

def multi_cluster_gaussian_with_pi_k(pi_k, big_sigma_k, a_k, X, k):
    gaussians_with_pi = []
    # calculating enumerators of both clusters of responsibilities
    for i in range(k):
        gaussians_with_pi.append(gaussian_with_pi_k(pi_k[i], big_sigma_k[i], a_k[i], X))
    return gaussians_with_pi


def E_step(pi_k, big_sigma_k, X, a_k, k):
    gaussians_with_pi = multi_cluster_gaussian_with_pi_k(pi_k, big_sigma_k, a_k, X, k)

    # calculating responsibilities for both clusters
    r_k = []
    for gaussian_pi in gaussians_with_pi:
        r_k.append(gaussian_pi / sum(gaussians_with_pi))

    return r_k


def M_step(r_k, X):
    # I always reference with k if that contains information about both clusters,
    # otherwise I use letter without k
    n = len(X)
    pi_k = []
    big_sigma_k = []
    a_k = []
    # iterate over clusters
    for i, r in enumerate(r_k):
        sum_of_r = sum(r)

        # updating pi
        new_pi = sum_of_r / n
        pi_k.append(new_pi)

        # updating mean
        new_a = (1 / sum_of_r) * (r @ X)
        a_k.append(new_a)

        # updating variance
        diff = X - new_a
        weighted_diff = diff.T * r
        new_cov = np.dot(weighted_diff, diff) / sum_of_r
        big_sigma_k.append(new_cov)

    return pi_k, big_sigma_k, a_k

def calc_likelihood(pi_k, big_sigma_k, a_k, X, k):
    gaussians_with_pi = multi_cluster_gaussian_with_pi_k(pi_k, big_sigma_k, a_k, X, k)
    log_likelihood = sum(np.log(sum(gaussians_with_pi)))
    return log_likelihood

# starting main code:
data = prep_data()

# initialization of values
k = 2
pi_k = np.array([1 / k, 1 / k])

# starting from the values that were given in the book so that we have a matching graph
# mu = np.array([[-1, 1], [1, -1]])
mu = np.array([[-1.5, 1.5], [1.5, -1.5]])
cov = np.array([np.eye(2), np.eye(2)])

# visualizing initial data
plot_data(data, mu, cov)

epsilon = 0.00001
previous_likelihood = 0
# EM iteration
for i in range(60):
    responsibilities = E_step(pi_k, cov, data, mu, k)
    pi_k, cov, mu = M_step(responsibilities, data)
    likelihood = calc_likelihood(pi_k, cov, mu, data, k)
    delta = abs(likelihood - previous_likelihood)
    previous_likelihood = likelihood
    print(delta)
    if epsilon >= delta:
        break

print(f'stopped after iteration {i+1}')
#
plot_classified_data(data, mu, cov, responsibilities)
