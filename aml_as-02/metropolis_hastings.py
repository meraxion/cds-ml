import numpy as np


# we either accept or reject the proposed x based on the acceptance probability
def sample_x(x_proposed, x_initial, proportional_function):
    acceptance_ratio = proportional_function(x_proposed) / proportional_function(x_initial)
    acceptance_prob = min(1, acceptance_ratio)
    u = np.random.uniform(0, 1, size=1)
    if u <= acceptance_prob:
        return x_proposed, 1
    else:
        return x_initial, 0


def metropolis_hastings(num_iterations, x_init, sigma, proportional_function):
    X = []
    sum_proposed = 0
    for t in range(num_iterations):
        # we always sample from normal distribution that is centered around current x
        # todo: I'm not sure if we can simply use multivariate_normal here, seems like cheating. Idk what std to use and also if I'd use my gaussian function
        x_proposed = np.random.multivariate_normal(x_init, np.eye(2)*sigma**2)

        x_sample, num_proposed = sample_x(x_proposed, x_init, proportional_function)
        X.append(x_sample)
        x_init = x_sample

        sum_proposed += num_proposed

    # calculating the ratio of proposed x that were accepted
    all_acceptance_ratio = sum_proposed/num_iterations
    return np.array(X), all_acceptance_ratio
