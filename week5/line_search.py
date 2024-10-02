import numpy as np
from matplotlib import pyplot as plt
from scipy.special import expit
from scipy.optimize import minimize, minimize_scalar
from functools import partial
import time

from helpers import cost, check_labels, sigmoid, grad_calc, load_data


# --- Line search
# in each iteration compute the gradient at the current value of w: d = sigmoid_grad_calc(output(weights, data), target, data)
# Then find numerically the value of gamma > 0 such that cost(output(weights+gamma*d, data), target) is minimized
# This is a standard one-dimensional optimization problem.
# either use e.g. scipy.special.minimize, or roll your own implementation

def line_search(train, test, weights, max_iter):
    x, t = train
    x_test, t_test = test
    train_errors = np.zeros(max_iter)
    test_errors = np.zeros(max_iter)

    # partial_output = partial(output, data = x)
    # partial_cost   = partial(cost, target = t)
    # def my_func(w):
    #   return partial_cost(partial_output(weights=w))

    def my_func(gam, w, d, xs, ts):
        return cost(sigmoid(w + gam * d, xs), ts)

    for i in range(max_iter):
        # print(f"iteration {i+1}")
        train_errors[i] = cost(sigmoid(weights, x), t)
        test_errors[i] = cost(sigmoid(weights, x_test), t_test)
        d = -grad_calc(sigmoid(weights, x), t, x)

        guess = (0, 10)  # guess of interval over which to look for gamma
        res = minimize_scalar(my_func, bracket=guess,
                              args=(weights, d, x, t), method="golden")
        gamma = res.x
        weights = weights + gamma * d

    class_rate_train = check_labels(weights, train)
    class_rate_test = check_labels(weights, test)

    return i, weights, train_errors, test_errors, (class_rate_train, class_rate_test)


def line_search_analytics(idx, weights, train_errors, test_errors, class_error, elapsed):
    class_error_train, class_error_test = class_error

    axis = np.arange(idx + 1)

    plt.figure(figsize=(12, 8))
    plt.plot(axis, train_errors, label="Training error")
    plt.plot(axis, train_errors, label="Testing error")
    plt.xlabel("iteration")
    plt.ylabel("Error")
    plt.title("Line Search")
    plt.legend()
    plt.show()

    print(f"Line Search Analytics:")
    print(f"Stopped after {idx + 1} iterations.")
    print(f"Final training error = {train_errors[-1]:.5f}, Final testing error = {test_errors[-1]:.5f}")
    print(f"Training set classification error = {100 * class_error_train:.2f}%")
    print(f"Testing set classification error = {100 * class_error_test:.2f}%")
    print(f"Elapsed wall clock time was {elapsed} seconds")

    return


def polak_ribiere(grad, grad_old):
    return (np.dot(grad - grad_old, grad)) / np.dot(grad_old, grad_old)


def conjugate_gradient_descent(train, test, weights, max_iter):
    """
    Uses line search as a subroutine
    In each step the search direction is d = -grad(w) +beta*d[-1]
    beta is given by the Polak Ribiere rule
    """
    x, t = train
    x_test, t_test = test
    train_errors = np.zeros(max_iter)
    test_errors = np.zeros(max_iter)

    # I wasn't sure how to do the first iteration but this is what I landed on
    d_old = 0
    grad_old = 1

    def my_func(gam, w, d, xs, ts):
        return cost(sigmoid(w + gam * d, xs), ts)

    for i in range(1, max_iter + 1):
        train_errors[i - 1] = cost(sigmoid(weights, x), t)
        test_errors[i - 1] = cost(sigmoid(weights, x_test), t_test)

        grad = grad_calc(sigmoid(weights, x), t, x)
        beta = polak_ribiere(grad, grad_old)
        d = -grad_calc(sigmoid(weights, x), t, x) + beta * d_old

        guess = (0, 10)
        res = minimize_scalar(my_func, bracket=guess,
                              args=(weights, d, x, t), method="golden")
        gamma = res.x

        weights = weights + gamma * d
        d_old = d
        grad_old = grad

    class_rate_train = check_labels(weights, train)
    class_rate_test = check_labels(weights, test)

    return i, weights, train_errors, test_errors, (class_rate_train, class_rate_test)


def conjugate_gd_analytics(idx, weights, train_errors, test_errors, class_error, elapsed):
    class_error_train, class_error_test = class_error

    axis = np.arange(idx)

    plt.figure(figsize=(12, 8))
    plt.plot(axis, train_errors, label="Training error")
    plt.plot(axis, test_errors, label="Testing error")
    plt.xlabel("iteration")
    plt.ylabel("Error")
    plt.title("Conjugate Gradient Descent")
    plt.legend()
    plt.show()

    print(f"Conjugate Gradient Descent Analytics:")
    print(f"Stopped after {idx + 1} iterations.")
    print(f"Final training error = {train_errors[-1]:.5f}, Final testing error = {test_errors[-1]:.5f}")
    print(f"Training set classification error = {100 * class_error_train:.2f}%")
    print(f"Testing set classification error = {100 * class_error_test:.2f}%")
    print(f"Elapsed wall clock time was {elapsed} seconds")

    return


def run_line_search():
    x_train, t_train, x_test, t_test = load_data()
    weights = np.random.rand(x_train.shape[1])

    # Line Search
    start = time.time()
    idx, weights, train_errors, test_errors, class_error = line_search((x_train, t_train), (x_test, t_test), weights,
                                                                       250)
    end = time.time()
    elapsed = end - start
    line_search_analytics(idx, weights, train_errors, test_errors, class_error, elapsed)

    # Conjugate Gradient Descent
    start = time.time()
    idx, weights, train_errors, test_errors, class_error = conjugate_gradient_descent((x_train, t_train),
                                                                                      (x_test, t_test), weights, 100)
    end = time.time()
    elapsed = end - start
    conjugate_gd_analytics(idx, weights, train_errors, test_errors, class_error, elapsed)
