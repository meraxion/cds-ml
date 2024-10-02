import time

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import expit
from scipy.optimize import minimize
from functools import partial

from helpers import update_rule, cost, check_labels, sigmoid, grad_calc


def SGD_analytics(idx, weights, train_errors, test_errors, test_data, train_data):
    axis = np.arange(idx)

    plt.figure(figsize=(12, 8))
    plt.plot(axis, train_errors, label="Training error")
    plt.plot(axis, test_errors, label="Testing error")
    plt.xlabel("iteration")
    plt.ylabel("Cost")
    plt.title("Stochastic Gradient Descent")
    plt.legend()
    plt.show()

    classification_error_test = check_labels(weights, test_data)
    classification_error_train = check_labels(weights, train_data)

    print(f"Stochastic Gradient Descent:")
    print(f"Stopped after {idx + 1} iterations.")
    print(f"Final train E = {train_errors[-1]:.5f}, Final train E = {test_errors[-1]:.5f}")
    print(f"Training set classification error = {100 * classification_error_train:.2f}%")
    print(f"Testing set classification error = {100 * classification_error_test:.2f}%")


def update_rule_SGD(current_weights, X, labels, learning_rate, batch_size):
    num_batches = int(np.ceil(X.shape[0] / batch_size))
    data_batched_x = np.array_split(X, num_batches)
    data_batched_labels = np.array_split(labels, num_batches)
    grad = np.zeros_like(current_weights)
    for X_batch, target_batch in zip(data_batched_x, data_batched_labels):
        current_output = sigmoid(current_weights, X_batch)
        grad += grad_calc(current_output, target_batch, X_batch)
    weight_update = current_weights - learning_rate * grad
    return weight_update


def SGD(train, val, test, weights, learning_rate, max_iter, batch_size):
    x, labels = train
    x_vali, y_vali = val
    x_test, y_test = test
    train_e = np.zeros(max_iter)
    validation_errors = np.zeros(max_iter)
    test_errors = np.zeros(max_iter)

    for i in range(max_iter):
        weights = update_rule_SGD(weights, x, labels, learning_rate, batch_size)
        train_e[i] = cost(sigmoid(weights, x), labels)

        validation_errors[i] = cost(sigmoid(weights, x_vali), y_vali)
        test_errors[i] = cost(sigmoid(weights, x_test), y_test)

    train_e = train_e[:i]
    validation_errors = validation_errors[:i]
    test_errors = test_errors[:i]
    return i, weights, train_e, validation_errors, test_errors


def experiments_SGD(train, val, test, weights, max_iter):
    batch_sizes = [60, 100, 600, 1000]
    learning_rates = [0.5, 0.1, 0.01]
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            start = time.time()
            idx, weights, train_errors, validation_errors, test_errors = SGD(train, val, test, weights,
                                                                             learning_rate, max_iter, batch_size)
            print("Batch size: {}, learning_rate: {}. Elapsed time: {}".format(batch_size, learning_rate,
                                                                               time.time() - start))

            SGD_analytics(idx, weights, train_errors, test_errors, train, test)
