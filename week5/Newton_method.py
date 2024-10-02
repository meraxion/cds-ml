# todo implement hessian with weight decay once we know that weight decay is correct
import time

import numpy as np
from matplotlib import pyplot as plt

from SGD import sigmoid
from gradient_methods import sigmoid, grad_calc, cost, check_labels, sigmoid_weight_decay
from helpers import load_data


def calc_hessian(output, data):
    N = len(output)
    term_y = output * (1 - output)
    weighted_data = data * term_y[:, np.newaxis]
    hessian = (data.T @ weighted_data) / N

    return hessian


def invert_hessian(hessian):
    lambda_reg = 1e-3  # Try increasing this value
    hessian += lambda_reg * np.eye(hessian.shape[0])
    hessian_inv = np.linalg.inv(hessian)
    return hessian_inv


def update_rule_Newton(current_weights, data, target, lambda_, weights):
    current_output = sigmoid(current_weights, data)
    hessian = calc_hessian(current_output, data)
    hessian_inverted = invert_hessian(hessian)
    weight_update = current_weights - hessian_inverted @ sigmoid_weight_decay(current_output, target, data, lambda_,
                                                                              weights)
    return weight_update


def Newton_method(train, val, test, weights, max_iter):
    x, labels = train
    x_vali, y_vali = val
    x_test, y_test = test
    train_e = np.zeros(max_iter)
    validation_errors = np.zeros(max_iter)
    test_errors = np.zeros(max_iter)
    previous_validation_error = 1000
    lambda_ = 0.1

    classif_error = np.zeros(max_iter)
    for i in range(max_iter):
        weights = update_rule_Newton(weights, x, labels, lambda_, weights)
        train_e[i] = cost(sigmoid(weights, x), labels)

        validation_errors[i] = cost(sigmoid(weights, x_vali), y_vali)
        test_errors[i] = cost(sigmoid(weights, x_test), y_test)

        classif_error[i] = check_labels(weights, test)

    train_e = train_e[:i]
    validation_errors = validation_errors[:i]
    test_errors = test_errors[:i]
    return i, weights, train_e, validation_errors, test_errors, classif_error


def newton_analytics(idx, weights, train_errors, test_errors, test_data, train_data):
    axis = np.arange(idx)

    plt.figure(figsize=(12, 8))
    plt.plot(axis, train_errors, label="Training error")
    plt.plot(axis, test_errors, label="Testing error")
    plt.xlabel("iteration")
    plt.ylabel("Cost")
    plt.title("Newton method")
    plt.legend()
    plt.show()

    classification_error_test = check_labels(weights, test_data)
    classification_error_train = check_labels(weights, train_data)

    print(f"Newton Method Analytics:")
    print(f"Stopped after {idx + 1} iterations.")
    print(f"Final train E = {train_errors[-1]:.5f}, Final train E = {test_errors[-1]:.5f}")
    print(f"Training set classification error = {classification_error_train:.2f}%")
    print(f"Testing set classification error = {classification_error_test:.2f}%")



def run_newton_method(train, validation, test, weights, max_iter):
    start = time.time()
    idx, weights, train_errors, validation_errors, test_errors, classif_error = Newton_method(train, validation, test, weights, max_iter)
    end = time.time()
    newton_analytics(idx, weights, train_errors, test_errors, train, test)
    print(f"Elapsed time: {(end-start):.4f} seconds")
