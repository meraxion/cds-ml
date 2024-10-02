import numpy as np
from matplotlib import pyplot as plt
from scipy.special import expit
from scipy.optimize import minimize
from functools import partial
import time


# Input data
def load_data():
    data = np.load('test_data.npz')
    X_test_norm = data['X_test_norm']
    labels_test_norm = data['Y_test_norm']

    data = np.load('train_data.npz')
    X_train_norm = data['X_train_norm']
    labels_train_norm = data['Y_train_norm']

    return X_train_norm, labels_train_norm, X_test_norm, labels_test_norm

def cost(output, target):
    eps = 1e-15  # Small epsilon value
    output = np.clip(output, eps, 1 - eps)
    return -np.mean(target * np.log(output) + (1 - target) * np.log(1 - output))


def check_labels(w, data):
    x, labels = data
    preds = sigmoid(w, x)
    pred_labels = (preds >= 0.5).astype(int)
    comparison = pred_labels == labels
    correct = np.count_nonzero(comparison)
    return (1 - (correct / len(labels))) * 100


# probability
def sigmoid(weights, data):
    return expit(data @ weights)


# gradient

def grad_calc(output, target, data):
    N = len(output)
    gradient = ((output - target) @ data) / N

    return gradient


# --- Gradient Descent
# update rule
def update_rule(current_weights, data, target, learning_rate):
    current_output = sigmoid(current_weights, data)
    weight_update = current_weights - learning_rate * grad_calc(current_output, target, data)
    return weight_update
