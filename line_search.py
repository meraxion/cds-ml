import numpy as np
from matplotlib import pyplot as plt
from scipy.special import expit
from scipy.optimize import minimize
from functools import partial

from gradient_methods import cost, check_labels, output, sigmoid_grad_calc

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

def main():
  print("Hello :3")

if __name__ == "__main__":
  main()