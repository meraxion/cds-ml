import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Function to plot a 2D Gaussian distribution as an ellipse
def plot_gaussian(mu, cov, ax, color):
    # Compute the eigenvalues and eigenvectors for the covariance matrix
    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * np.sqrt(vals)

    # Plot the ellipse corresponding to the Gaussian
    ellipse = Ellipse(xy=mu, width=width, height=height, angle=angle, edgecolor=color, fc='None', lw=2)
    ax.add_patch(ellipse)
def plot_data(data, mu, cov):
    # Plot
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], s=10, c='green', label='Samples')

    # Plot the Gaussian distributions as ellipses
    plot_gaussian(mu[0], cov[0], ax, color='blue')
    plot_gaussian(mu[1], cov[1], ax, color='red')

    # Set the plot limits and labels
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')

    plt.show()


def prep_data():
    data = pd.read_csv('old_faithful_geyser.txt', delim_whitespace=True, header=None, index_col=0)
    # data = pd.read_csv('old_faithful_geyser.txt', delimiter="      ",header=None, index_col=0)
    data_n = data.to_numpy()
    # data.columns = ['eruptions', 'waiting']
    # print(data)
    mean = np.mean(data_n, axis=0)
    std_dev = np.std(data_n, axis=0)

    # Perform the standardization
    standardized_data = (data_n - mean) / std_dev
    return standardized_data


def gaussian(sigma, a, x):
    term_one = (1 / np.sqrt(2 * np.pi * sigma ** 2))
    term_two = np.exp(-(x - a) ** 2 / (2 * big_sigma_k))
    return term_one * term_two

def multivariate_gaussian(sigma, a, x):
    D = mu.shape[0]

    # Compute the determinant and inverse of the covariance matrix
    sigma_det = np.linalg.det(sigma)
    sigma_inv = np.linalg.inv(sigma)

    # Compute the normalization factor
    norm_factor = 1 / np.sqrt((2 * np.pi) ** D * sigma_det)

    # Center the data
    diff = x - mu

    # Compute the exponent term
    exponent = np.einsum('...i,ij,...j', diff, sigma_inv, diff)

    # Return the multivariate Gaussian density
    return norm_factor * np.exp(-0.5 * exponent)

def gaussian_w_pi_k(pi_k, big_sigma, a, X):
    gaussian_pi = pi_k * multivariate_normal.pdf(X, mean=a, cov=big_sigma)
    return gaussian_pi


def calc_r_k(pi_k, big_sigma_k, X, a_k):
    gausians_with_pi = []
    for i in range(2):
    # for i, x in enumerate(X.T):
        # todo sigma and a will be different for different k, so I need to take them from a matrix!
        gaussian_pi_k = gaussian_w_pi_k(pi_k, big_sigma_k[i], a_k[i], X)
        gausians_with_pi.append(gaussian_pi_k)
    r_ks = []
    for i, gaussian_pi_k in enumerate(gausians_with_pi):
        r_k = gaussian_pi_k / sum(gausians_with_pi)
        r_ks.append(r_k)
    # denominator = np



# initialization of values
k = 2
pi_k = 1/k

# a_k = np.array([[1, -1], [1, -1]]) # taken from a book [mu1, mu2]
a_k = np.random.uniform(low=-1, high=1, size=1)
# a_k = np.random.uniform(low=-1, high=1, size=2)
# I = np.identity(2) # taken from a book [mu1, mu2]
# big_sigma_k = np.array([I, I])
big_sigma_k = np.array([1])
# big_sigma_k = np.array([1, 1])
data = prep_data()

mu = np.array([[-1.5, 1.5], [1.5, -1.5]])
cov =np.array([np.eye(2), np.eye(2)])



plot_data(data, mu, cov)
# plot_data(data, a_k, big_sigma_k)
r_k = calc_r_k(pi_k, cov, data, mu)





