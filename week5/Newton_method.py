# todo implement hessian with weight decay once we know that weight decay is correct
import numpy as np
from matplotlib import pyplot as plt

from week5.SGD import sigmoid
from week5.gradient_methods import output, sigmoid_grad_calc, cost, check_labels

def sigmoid_grad_calc(output, target, data):
  N = len(output)
  gradient = ((output - target) @ data) / N

  return gradient

def calc_hessian(output, data):
    N = len(output)
    term_y = output*(1- output)
    weighted_data = data * term_y[:, np.newaxis]  # Shape: (N, D)
    hessian = (data.T @ weighted_data) / N

    return hessian


def invert_hessian(hessian):
  lambda_reg = 1e-3  # Try increasing this value
  hessian += lambda_reg * np.eye(hessian.shape[0])
  hessian_inv = np.linalg.inv(hessian)
  return hessian_inv


def update_rule_Newton(current_weights, data, target, learning_rate):
  current_output = output(current_weights, data)
  hessian = calc_hessian(current_output, data)
  hessian_inverted = invert_hessian(hessian)
  weight_update = current_weights - hessian_inverted @ sigmoid_grad_calc(current_output, target, data)
  return weight_update

def Newton_method(train, val, test, weights, learning_rate, max_iter):
  x,labels = train
  x_vali, y_vali = val
  x_test, y_test = test
  train_e = np.zeros(max_iter)
  validation_errors = np.zeros(max_iter)
  test_errors = np.zeros(max_iter)
  previous_validation_error = 1000

  classif_error = np.zeros(max_iter)
  for i in range(max_iter):
    weights = update_rule_Newton(weights, x, labels, learning_rate)
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
  idx, weights, train_errors, validation_errors, test_errors, classif_error = Newton_method((X_train_norm_gd, Y_train_norm_gd), (X_train_norm_val, Y_train_norm_val),
                               (X_test_norm, Y_test_norm), weights, 0.1, 10)

  plt.plot(classif_error)
  plt.show()