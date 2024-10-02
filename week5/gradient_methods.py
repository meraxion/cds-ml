import numpy as np
from matplotlib import pyplot as plt
from scipy.special import expit
from scipy.optimize import minimize
from functools import partial
import time

from helpers import cost, check_labels, sigmoid, grad_calc, update_rule

# --- Problem definition
"""
consider the the logistic regression problem
the model is given in the exercise pdf, and Bishop section 4.3
"""


# early stopping
def early_stopping(vali_e, prev_vali_e):
    return vali_e > prev_vali_e


# gradient descent
def gradient_descent(input_train, input_vali, input_test, weights, learning_rate, max_iter):
    x, y = input_train
    x_vali, t_vali = input_vali
    x_test, t_test = input_test
    train_e = np.zeros(max_iter)
    validation_e = np.zeros(max_iter)
    test_e = np.zeros(max_iter)
    previous_validation_error = 1000

    classif_error = np.zeros(max_iter)
    for i in range(max_iter):
        weights = update_rule(weights, x, y, learning_rate)
        train_e[i] = cost(sigmoid(weights, x), y)

        validation_e[i] = cost(sigmoid(weights, x_vali), t_vali)
        test_e[i] = cost(sigmoid(weights, x_test), t_test)

        classif_error[i] = check_labels(weights, input_test)

        stop_check = early_stopping(validation_e[i], previous_validation_error)
        if stop_check:
            break
        previous_validation_error = validation_e[i]

    train_e = train_e[:i]
    validation_e = validation_e[:i]
    test_e = test_e[:i]
    return i, weights, train_e, validation_e, test_e, classif_error


def gradient_descent_analytics(par, learning_rate):
    i, weights, train_e, validation_e, test_e, classif_error = par
    plt.figure(figsize=(10, 6))
    plt.plot(train_e, label="Training error")
    plt.plot(validation_e, label="Validation error")
    plt.plot(test_e, label="Test error")
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.title(f"Gradient descent, learning rate: {learning_rate}")
    plt.legend()
    plt.show()

    print(f"Stopped after {i} iterations")
    print(f"E_train: {train_e[-1]}, E_test: {test_e[-1]}")
    print(
        f"Misclassified train set: {classif_error[-1]}")  # this is added so now I am unsure also it says train and test

    return


def experiments_gradient_descent(input_train, input_vali, input_test, weights, max_iter):
    learning_rates = [1, 0.001, 0.01, 0.1, 0.5]
    for learning_rate in learning_rates:
        start = time.time()
        par = gradient_descent(input_train, input_vali, input_test, weights, learning_rate, max_iter)
        gradient_descent_analytics(par, learning_rate)
        end = time.time()
        print(f"CPU time: {end - start}")


# --- Momentum
def momentum_update_rule(current_weights, data, target, learning_rate, momentum, alpha):
    current_output = sigmoid(current_weights, data)
    momentum = - learning_rate * grad_calc(current_output, target, data) + alpha * momentum
    weights = current_weights + momentum
    return weights, momentum


def gradient_descent_momentum(input_train, input_vali, input_test, weights, learning_rate, max_iter, alpha):
    x, t = input_train
    x_vali, t_vali = input_vali
    x_test, t_test = input_test
    train_e = np.zeros(max_iter)
    validation_e = np.zeros(max_iter)
    test_e = np.zeros(max_iter)
    momentum = np.zeros_like(weights)
    classif_error = np.zeros(max_iter)
    previous_validation_error = 1000
    for i in range(max_iter):
        weights, momentum = momentum_update_rule(weights, x, t, learning_rate, momentum, alpha)
        train_e[i] = cost(sigmoid(weights, x), t)

        validation_e[i] = cost(sigmoid(weights, x_vali), t_vali)
        test_e[i] = cost(sigmoid(weights, x_test), t_test)
        stop_check = early_stopping(validation_e[i], previous_validation_error)
        classif_error[i] = check_labels(weights, input_test)
        if stop_check:
            break
        previous_validation_error = validation_e[i]

    train_e = train_e[:i]
    validation_e = validation_e[:i]
    test_e = test_e[:i]

    return i, weights, train_e, validation_e, test_e, classif_error


def momentum_analytics(par, learning_rate, alpha):
    i, weights, train_e, validation_e, test_e, classif_error = par
    plt.figure(figsize=(10, 6))
    plt.plot(train_e, label="Training error")
    plt.plot(validation_e, label="Validation error")
    plt.plot(test_e, label="Test error")
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.title(f"Momentum learning rate: {learning_rate}, alpha: {alpha}")
    plt.legend()
    plt.show()

    print(f"Stopped after {i} iterations")
    print(f"E_train: {train_e[-1]}, E_test: {test_e[-1]}")
    print(
        f"Misclassified train set: {classif_error[-1]}")  # this is added so now I am unsure also it says train and test

    return


def expirements_momentum(input_train, input_vali, input_test, initial_weights, learning_rates, max_iter):
    learning_rates = [0.01, 0.1, 0.5]
    alphas = [0.8, 0.9, 1]
    for learning_rate in learning_rates:
        for alpha in alphas:
            start = time.time()
            par = gradient_descent_momentum(input_train, input_vali, input_test, initial_weights, learning_rate,
                                            max_iter, alpha)
            momentum_analytics(par, learning_rate, alpha)
            end = time.time()
            print(f"CPU time: {end - start}")
    return


# --- Weight Decay
def cost_weight_decay(output, target, weights, data, lambda_):
    eps = 1e-15  # Small epsilon value
    output = np.clip(output, eps, 1 - eps)
    cost = -np.mean(target * np.log(output) + (1 - target) * np.log(1 - output))
    weight_decay = (lambda_ / (2 * data.shape[0]) * np.sum(np.square(weights)))
    return cost + weight_decay


def sigmoid_weight_decay(output, target, data, lambda_, weights):
    N = len(output)
    gradient = ((output - target) @ data) / N + (lambda_ / N) * weights
    return gradient


def decay_update_rule(current_weights, data, target, learning_rate, momentum, alpha, lambda_):
    current_output = sigmoid(current_weights, data)
    momentum = -learning_rate * sigmoid_weight_decay(current_output, target, data, lambda_,
                                                     current_weights) + alpha * momentum
    weights = current_weights + momentum
    return weights, momentum


def gradient_descent_weight_decay(input_train, input_vali, input_test, weights, learning_rate, lambda_, max_iter,
                                  alpha):
    x, t = input_train
    x_vali, t_vali = input_vali
    x_test, t_test = input_test
    train_e = np.zeros(max_iter)
    validation_e = np.zeros(max_iter)
    test_e = np.zeros(max_iter)
    momentum = np.zeros_like(weights)
    classif_error = np.zeros(max_iter)
    previous_validation_error = 1000
    for i in range(max_iter):
        weights, momentum = decay_update_rule(weights, x, t, learning_rate, momentum, alpha, lambda_)
        train_e[i] = cost_weight_decay(sigmoid(weights, x), t, weights, x, lambda_)

        validation_e[i] = cost_weight_decay(sigmoid(weights, x_vali), t_vali, weights, x_vali, lambda_)
        test_e[i] = cost_weight_decay(sigmoid(weights, x_test), t_test, weights, x_test, lambda_)
        stop_check = early_stopping(validation_e[i], previous_validation_error)
        classif_error[i] = check_labels(weights, input_test)
        if stop_check:
            break
        previous_validation_error = validation_e[i]

    train_e = train_e[:i]
    validation_e = validation_e[:i]
    test_e = test_e[:i]

    return i, weights, train_e, validation_e, test_e, classif_error


def weight_decay_analytics(par, lambda_):
    i, weights, train_e, validation_e, test_e, classif_error = par
    plt.figure(figsize=(10, 6))
    plt.plot(train_e, label="Training error")
    plt.plot(validation_e, label="Validation error")
    plt.plot(test_e, label="Test error")
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.title(f"Weight decay lambda {lambda_}")
    plt.legend()
    plt.show()

    print(f"Stopped after {i} iterations")
    print(f"E_train: {train_e[-1]}, E_test: {test_e[-1]}")
    print(f"Misclassified train set: {classif_error[-1]}")
    return


def experiments_weight_decay(input_train, input_vali, input_test, initial_weights, learning_rate, max_iter, alpha):
    lambda_ = 0.1
    start = time.time()
    par = gradient_descent_weight_decay(input_train, input_vali, input_test, initial_weights, learning_rate, lambda_,
                                        max_iter, alpha)
    weight_decay_analytics(par, lambda_)
    end = time.time()
    print(f"CPU time: {end - start}")
    return
