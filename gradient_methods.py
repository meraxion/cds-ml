import numpy as np
from scipy.special import expit

# --- Problem definition
"""
consider the the logistic regression problem
the model is given in the exercise pdf, and Bishop section 4.3
"""

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

# Targets

def cost(output, target):
  return -1/len(target)*np.sum(target*np.log(output)+(1-target)*np.log(1-output))

# probability

def output(weights, data):
  return expit(weights.T*data)

# gradient

def sigmoid_grad_calc(output, target, data):
  N = len(output)
  gradient = np.zeros_like(data)
  for i, _ in enumerate(data):
    gradient[i] = 1/N * np.sum((output-target)*data[i])

  return gradient

# Hessian

def hessian(data, output):
  # d = len(data.shape[1])
  # H = np.zeros((d,d))
  # for i in range(d):
  #   for j in range(d):
  #     H[i,j] = np.sum(data[i,:]*output*(1-output)*data[j,:])

  D = output*(1-output)

  return data@D@data.T

  

# --- Gradient Descent
def gradient_descent(grad, val_train, val_test, weights, learning_rate, max_iter):
  x,y = val_train
  x_val, y_val = val_test
  train_errors = np.zeros(len(max_iter))
  validation_errors = np.zeros(len(max_iter))
  for i in range(max_iter):
    weights = update_rule(weights, x, y, learning_rate)
    train_errors[i] = cost(output(weights, x), y)

    validation_errors[i] = cost(output(weights, x_val), y_val)
    stop_check = early_stopping(validation_errors[i], validation_errors[i-1])
    if stop_check:
      break

  final_error = cost(output(weights, x), y)

  return

# update rule
# so in my gradient descent we had multiple gradients, we have one so this is then the one function I think but
# then we call the rule in a loop right?
def update_rule(current_weights, data, target, learning_rate):
  current_output = output(current_weights, data)
  weight_update = current_weights - learning_rate * sigmoid_grad_calc(current_output, target, data)
  return weight_update

# early stopping
def early_stopping(val_error, prev_val_error):
  return val_error > prev_val_error


# --- Momentum

# --- Weight Decay

# --- Newton Method

# --- Line search

# --- Conjugate gradient descent

# --- Stochastic gradient descent

if __name__ == "__main__":
  print("hello world")