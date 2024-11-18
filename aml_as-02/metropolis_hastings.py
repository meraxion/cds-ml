import numpy as np


# we either accept or reject the proposed x based on the acceptance probability
def sample_x(x_proposed, x_initial, proportional_function):
    acceptance_ratio = proportional_function(x_proposed) / proportional_function(x_initial)
    acceptance_prob = min(1, acceptance_ratio)
    u = np.random.uniform(0, 1, size=1)
    if u <= acceptance_prob:
        return x_proposed
    else:
        return x_initial


def metropolis_hastings(num_iterations, x_init, proportional_function):
    X = []
    for t in range(num_iterations):
        # we always sample from normal distribution that is centered around current x
        # todo: I'm not sure if we can simply use multivariate_normal here, seems like cheating. Idk what std to use and also if I'd use my gaussian function
        x_proposed = np.random.multivariate_normal(x_init, np.eye(2))

        x_sample = sample_x(x_proposed, x_init, proportional_function)
        X.append(x_sample)
        x_init = x_sample
    return np.array(X)
