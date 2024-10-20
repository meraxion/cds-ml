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




def gaussian_w_pi_k(pi_k, big_sigma, a, X):
    gaussian_pi = pi_k * multivariate_normal.pdf(X, mean=a, cov=big_sigma)
    return gaussian_pi


def E_step(pi_k, big_sigma_k, X, a_k):
    gausians_with_pi = []
    for i in range(2):
        gaussian_pi_k = gaussian_w_pi_k(pi_k[i], big_sigma_k[i], a_k[i], X)
        gausians_with_pi.append(gaussian_pi_k)
    r_ks = []
    for i, gaussian_pi_k in enumerate(gausians_with_pi):
        su = sum(gausians_with_pi)
        r_k = gaussian_pi_k / su
        r_ks.append(r_k)
    return r_ks



def M_step(r_ks, X):
    n = len(X)
    pi_k = []
    big_sigma_k = []
    a_k = []
    # iterate over clusters
    for i, r_k in enumerate(r_ks):
        N_k = sum(r_k)
        new_pi_k = N_k / n
        pi_k.append(new_pi_k)
        new_a_k = (1/ N_k ) * (r_k @ X)
        a_k.append(new_a_k)
        rr = (X - new_a_k) @ (X - new_a_k).T
        new_sigma_k = (1 / N_k) * (r_k @ rr)
        new_a_k_v2 = []

        for r, x in zip(r_k, X):
            aaa = (x -new_a_k).reshape(-1, 1)
            bbb = aaa @ aaa.T
            new_a_k_v2.append(r * bbb)

        res = (1/ N_k ) * sum(new_a_k_v2)
        big_sigma_k.append(res)

    return pi_k, big_sigma_k, a_k



# initialization of values
k = 2
pi_k = np.array([1/k, 1/k])

a_k = np.random.uniform(low=-1, high=1, size=1)


data = prep_data()

mu = np.array([[-1, 1], [1, -1]])
# mu = np.array([[-1.5, 1.5], [1.5, -1.5]])
cov =np.array([np.eye(2), np.eye(2)])



plot_data(data, mu, cov)
# plot_data(data, a_k, big_sigma_k)
print(np.array(pi_k).shape)
print(pi_k)
print(np.array(cov).shape)
print(cov)
print(np.array(mu).shape)
print(mu)
for _ in range(50):
    responsibilities = E_step(pi_k, cov, data, mu)
    pi_k, cov, mu = M_step(responsibilities, data)
    print('after')
    print(np.array(pi_k).shape)
    print(pi_k)
    print(np.array(cov).shape)
    print(cov)
    print(np.array(mu).shape)
    print(mu)
plot_data(data, mu, cov)




