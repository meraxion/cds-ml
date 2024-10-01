import numpy as np
from matplotlib import pyplot as plt
from scipy.special import expit
from scipy.optimize import minimize
from functools import partial

from gradient_methods import update_rule, cost, check_labels


def sigmoid_grad_calc(output, target, data):
  N = len(output)
  gradient = np.zeros_like(data[1, :])
  for i, data_i in enumerate(data.T):
    gradient[i] = 1/N * np.sum((output-target)*data_i)

  return gradient


def sigmoid(weights, data):
  return expit(data@weights)

def update_rule_SGD(current_weights, X, labels, learning_rate, batch_size):
  num_batches = int(np.ceil(X.shape[0] / batch_size))
  data_batched_x = np.array_split(X, num_batches)
  data_batched_labels = np.array_split(labels, num_batches)
  grad = np.zeros_like(current_weights)
  for X_batch, target_batch in zip(data_batched_x, data_batched_labels):
    current_output = sigmoid(current_weights, X_batch)
    grad += sigmoid_grad_calc(current_output, target_batch, X_batch)
  weight_update = current_weights - learning_rate * grad
  return weight_update


def SGD(train, val, test, weights, learning_rate, max_iter):
  x,labels = train
  x_vali, y_vali = val
  x_test, y_test = test
  train_e = np.zeros(max_iter)
  validation_errors = np.zeros(max_iter)
  test_errors = np.zeros(max_iter)
  previous_validation_error = 1000

  classif_error = np.zeros(max_iter)
  for i in range(max_iter):
    weights = update_rule_SGD(weights, x, labels, learning_rate, 1000)
    train_e[i] = cost(sigmoid(weights, x), labels)

    validation_errors[i] = cost(sigmoid(weights, x_vali), y_vali)
    test_errors[i] = cost(sigmoid(weights, x_test), y_test)

    classif_error[i] = check_labels(weights, test)



  train_e = train_e[:i]
  validation_errors = validation_errors[:i]
  test_errors = test_errors[:i]
  return i, weights, train_e, validation_errors, test_errors, classif_error


if __name__ == "__main__":
  # Input data
  #to access test data:
  data = np.load('test_data.npz')
  X_test_norm = data['X_test_norm']
  Y_test_norm = data['Y_test_norm']
  X_test_raw = data['X_test_raw']
  Y_test_raw = data['Y_test_raw']

  #to access train data:
  data = np.load('train_data.npz')
  X_train_norm = data['X_train_norm']
  Y_train_norm = data['Y_train_norm']
  X_train_raw = data['X_train_raw']
  Y_train_raw = data['Y_train_raw']
  split = int(0.8*len(X_train_norm))
  X_train_norm_gd = X_train_norm[:split-1]
  X_train_norm_val = X_train_norm[split:]
  Y_train_norm_gd = Y_train_norm[:split-1]
  Y_train_norm_val = Y_train_norm[split:]

  weights = np.random.rand(X_train_norm_gd.shape[1])
  idx, weights, train_errors, validation_errors, test_errors, classif_error = SGD((X_train_norm_gd, Y_train_norm_gd), (X_train_norm_val, Y_train_norm_val),
                               (X_test_norm, Y_test_norm), weights, 0.1, 100)

  plt.plot(classif_error)
  plt.show()
