import numpy as np
from matplotlib import pyplot as plt
from scipy.special import expit
from scipy.optimize import minimize
from functools import partial

from gradient_methods import cost, check_labels, output, sigmoid_grad_calc

# Input data
def load_data():
  data = np.load('test_data.npz')
  X_test_norm = data['X_test_norm']
  labels_test_norm = data['Y_test_norm']

  data = np.load('train_data.npz')
  X_train_norm = data['X_train_norm']
  labels_train_norm = data['Y_train_norm']

  return X_train_norm, labels_train_norm, X_test_norm, labels_test_norm
# --- Line search

# in each iteration compute the gradient at the current value of w: d = sigmoid_grad_calc(output(weights, data), target, data)

# Then find numerically the value of gamma > 0 such that cost(output(weights+gamma*d, data), target) is minimized

# This is a standard one-dimensional optimization problem.
# either use e.g. scipy.special.minimize, or roll your own implementation

# minimize()
def line_search(train, test, weights, max_iter):
  x,t = train
  x_test, t_test = test
  train_errors = np.zeros(max_iter)
  # test_errors  = np.zeros(len(max_iter))

  # partial_output = partial(output, data = x)
  # partial_cost   = partial(cost, target = t)
  # def my_func(w):
  #   return partial_cost(partial_output(weights=w))

  def my_func(w, xs, ts):
    return cost(output(w, xs), ts)

  for i in range(max_iter):
    print(f"iteration {i+1}")
    train_errors[i] = cost(output(weights, x), t)

    print("now starting one-d optimization")
    res = minimize(my_func, x0=weights, args = (x, t), method="Nelder-Mead", options={"maxiter":1})
    gamma = res.x
    print(f"one-d optimization done, gamma found: {res}")
    d = sigmoid_grad_calc(output(weights, x), t, x)
    weights = weights + gamma*d
  
  test_error  = cost(output(weights, x_test), t_test)
  class_rate  = check_labels(weights, test)

  return i, weights, train_errors, test_error, class_rate

def line_search_analytics():

  return

def main():
  print("Hello :3")
  x_train, t_train, x_test, t_test = load_data()
  weights = np.random.rand(x_train.shape[1])
  idx, weights, train_errors, test_error, class_rate = line_search((x_train, t_train), (x_test, t_test), weights, 500)

  print(weights)

if __name__ == "__main__":
  main()