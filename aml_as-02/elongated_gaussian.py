import matplotlib.pyplot as plt
import numpy as np


# Q is gaussian centered around current x

def gaussian(x, mean, std):
    print(std)
    if std == 0:
        return np.array([1, 1])
    else:
        return (1 / np.sqrt(2 * np.pi * std ** 2)) * np.exp(-(x - mean) ** 2 / (2 * std ** 2))


def define_Q(x):
    mean = np.mean(x)
    std = np.std(x)
    return mean, std


def calc_E(x, A):
    # is it x.T? in the assignment it's is x' but in the mcKay book it's x.T(page 385 or smth like that under Hamilton method)
    # I assume that A is not a scalar but not sure?
    E = 1 / 2 * x.T @ A @ x
    return E


def sample_x(x_proposed, x_init, A):
    e1 = calc_E(x_proposed, A)
    e2 = calc_E(x_init, A)
    acceptance_ratio = e1 / e2
    u = np.random.uniform(0, 1, size=1)
    if u <= acceptance_ratio:
        print('proposed')
        return x_proposed
    else:
        print('init')
        return x_init
    # return ratio


num_iterations = 10000
A = np.array([[250.25, -249.75], [-49.75, 250.25]])

# where to get these from? or they're completely random?
x_init = np.array([0, 0])

X = []
for t in range(num_iterations):
    # we always sample from normal distribution that is centered around current x
    # todo: I'm not sure if we can simply use this function here, seems like cheating. Idk what std to use and also if I'd use my gaussian function
    x_proposed = np.random.multivariate_normal(x_init, np.eye(2))
    # x_proposed = gaussian(x_init, Q_mean, Q_std)
    print(x_proposed)
    x_init = sample_x(x_proposed, x_init, A)
    X.append(x_init)

X = np.array(X)


plt.scatter(X[:, 0], X[:, 1])
plt.show()