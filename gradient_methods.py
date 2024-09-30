import numpy as np
from matplotlib import pyplot as plt
from scipy.special import expit
from scipy.optimize import minimize
from functools import partial

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
split = int(0.8*len(X_train_norm))
X_train_norm_gd = X_train_norm[:split-1]
X_train_norm_val = X_train_norm[split:]
Y_train_norm_gd = Y_train_norm[:split-1]
Y_train_norm_val = Y_train_norm[split:]
# Targets

# def cost(output, target):
#   return -1/len(target)*np.sum(target*np.log(output)+(1-target)*np.log(1-output))

def cost(output, target):
  eps = 1e-15  # Small epsilon value
  output = np.clip(output, eps, 1 - eps)
  return -np.mean(target * np.log(output) + (1 - target) * np.log(1 - output))


# probability

def output(weights, data):
  return expit(data@weights)

# gradient

def sigmoid_grad_calc(output, target, data):
  N = len(output)
  gradient = np.zeros_like(data[1, :])
  for i, data_i in enumerate(data.T):
    gradient[i] = 1/N * np.sum((output-target)*data_i)

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

# gradient descent
def gradient_descent(val_train, val_vali, val_test, weights, learning_rate, max_iter):
  x,y = val_train
  x_vali, y_vali = val_vali
  x_test, y_test = val_test
  train_errors = np.zeros(max_iter)
  validation_errors = np.zeros(max_iter)
  test_errors = np.zeros(max_iter)
  for i in range(max_iter):
    weights = update_rule(weights, x, y, learning_rate)
    train_errors[i] = cost(output(weights, x), y)

    validation_errors[i] = cost(output(weights, x_vali), y_vali)
    test_errors[i] = cost(output(weights, x_test), y_test)
    # stop_check = early_stopping(validation_errors[i], validation_errors[i-1])
    # if stop_check:
    #   break

  train_errors = train_errors[:i]
  validation_errors = validation_errors[:i]
  test_errors = test_errors[:i]

  return i, weights, train_errors, validation_errors, test_errors

def gradient_descent_analytics(par):
  i, weights, train_errors, validation_errors, test_errors = par
  return



# --- Momentum
def momentum_update_rule(current_weights, data, target, learning_rate, momentum, alpha):
  current_output = output(current_weights, data)
  momentum = - learning_rate*sigmoid_grad_calc(current_output, target, data) + alpha*momentum
  weights = current_weights + momentum
  return weights, momentum

def gradient_descent_momentum(val_train, val_vali, val_test, weights, learning_rate, max_iter):
  x,t = val_train
  x_vali, y_vali = val_vali
  x_test, y_test = val_test
  train_errors = np.zeros(max_iter)
  validation_errors = np.zeros(max_iter)
  test_errors = np.zeros(max_iter)
  momentum = np.zeros_like(weights)
  for i in range(max_iter):
    weights, momentum[i+1] = momentum_update_rule(weights, x, t, learning_rate, momentum[i], alpha = 0.8)
    train_errors[i] = cost(output(weights, x), t)

    validation_errors[i] = cost(output(weights, x_vali), y_vali)
    test_errors[i] = cost(output(weights, x_test), y_test)
    stop_check = early_stopping(validation_errors[i], validation_errors[i-1])
    if stop_check:
      break

  train_errors = train_errors[:i]
  validation_errors = validation_errors[:i]
  test_errors = test_errors[:i]

  return i, weights, train_errors, validation_errors, test_errors

def momentum_analytics(par):
  return

# --- Weight Decay
def cost_weight_decay():
  return

# --- Newton Method

# --- Line search

# in each iteration compute the gradient at the current value of w: d = sigmoid_grad_calc(output(weights, data), target, data)

# Then find numerically the value of gamma > 0 such that cost(output(weights+gamma*d, data), target) is minimized

# This is a standard one-dimensional optimization problem.
# either use e.g. scipy.special.minimize, or roll your own implementation

# minimize()
def line_search(train, test, weights, max_iter):
  x,t = train
  x_test, t_test = test
  train_errors = np.zeros(len(max_iter))
  # test_errors  = np.zeros(len(max_iter))

  partial_output = partial(output, data = x)
  partial_cost   = partial(cost, target = t)

  def my_func(w):
    return partial_cost(partial_output(w))

  for i in range(max_iter):
    train_errors[i] = cost(output(weights, x), t)

    gamma = minimize(my_func, weights)
    d = sigmoid_grad_calc(output(weights, data), t, x)
    weights = weights + gamma*d
  
  test_error  = cost(output(weights, x), t)

  return i, weights, train_errors, test_error

def line_search_analytics():

  return

# --- Conjugate gradient descent

# --- Stochastic gradient descent

if __name__ == "__main__":
  weights = np.random.rand(X_train_norm_gd.shape[1])
  idx, weights, train_errors, validation_errors, test_errors = gradient_descent((X_train_norm_gd, Y_train_norm_gd), (X_train_norm_val, Y_train_norm_val), (X_test_norm, Y_test_norm), weights, 0.01, 750)
  plt.plot(train_errors)
  plt.plot(test_errors)
  plt.show()