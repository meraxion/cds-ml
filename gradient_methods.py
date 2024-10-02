import numpy as np
from matplotlib import pyplot as plt
from scipy.special import expit
from scipy.optimize import minimize
from functools import partial
import time

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

def check_labels(w, data):
  x, labels = data
  preds = output(w, x)
  pred_labels = (preds >= 0.5).astype(int)
  comparison = pred_labels == labels
  correct = np.count_nonzero(comparison)
  return 1- (correct/len(labels))

# probability
def output(weights, data):
  return expit(data@weights)

# gradient

def sigmoid_grad_calc(output, target, data):
  N = len(output)
  gradient = ((output - target) @ data) / N

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
def early_stopping(vali_e, prev_vali_e):
  return vali_e > prev_vali_e

# gradient descent
def gradient_descent(input_train, input_vali, input_test, weights, learning_rate, max_iter):
  x,y = input_train
  x_vali, t_vali = input_vali
  x_test, t_test = input_test
  train_e = np.zeros(max_iter)
  validation_e = np.zeros(max_iter)
  test_e = np.zeros(max_iter)
  previous_validation_error = 1000

  classif_error = np.zeros(max_iter)
  for i in range(max_iter):
    weights = update_rule(weights, x, y, learning_rate)
    train_e[i] = cost(output(weights, x), y)

    validation_e[i] = cost(output(weights, x_vali), t_vali)
    test_e[i] = cost(output(weights, x_test), t_test)

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
  plt.figure(figsize=(10,6))
  plt.plot(train_e, label = "Training error")
  plt.plot(validation_e, label = "Validation error")
  plt.plot(test_e, label="Test error")
  plt.xlabel("Iterations")
  plt.ylabel("Error")
  plt.title(f"Gradient descent, learning rate: {learning_rate}")
  plt.legend()
  plt.show()

  print(f"Stopped after {i} iterations")
  print(f"E_train: {train_e[-1]}, E_test: {test_e[-1]}")
  print(f"Misclassified train set: {classif_error[-1]}") # this is added so now I am unsure also it says train and test


  return

def experiments_gradient_descent(input_train, input_vali, input_test, weights, max_iter):
  learning_rates = [1, 0.001, 0.01, 0.1, 0.5]
  for learning_rate in learning_rates:
    start = time.time()
    par = gradient_descent(input_train, input_vali, input_test, weights, learning_rate, max_iter)
    gradient_descent_analytics(par, learning_rate)
    end = time.time()
    print(f"CPU time: {end-start}")



# --- Momentum
def momentum_update_rule(current_weights, data, target, learning_rate, momentum, alpha):
  current_output = output(current_weights, data)
  momentum = - learning_rate*sigmoid_grad_calc(current_output, target, data) + alpha*momentum
  weights = current_weights + momentum
  return weights, momentum

def gradient_descent_momentum(input_train, input_vali, input_test, weights, learning_rate, max_iter, alpha):
  x,t = input_train
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
    train_e[i] = cost(output(weights, x), t)

    validation_e[i] = cost(output(weights, x_vali), t_vali)
    test_e[i] = cost(output(weights, x_test), t_test)
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
  plt.figure(figsize=(10,6))
  plt.plot(train_e, label = "Training error")
  plt.plot(validation_e, label = "Validation error")
  plt.plot(test_e, label="Test error")
  plt.xlabel("Iterations")
  plt.ylabel("Error")
  plt.title(f"Momentum learning rate: {learning_rate}, alpha: {alpha}")
  plt.legend()
  plt.show()

  print(f"Stopped after {i} iterations")
  print(f"E_train: {train_e[-1]}, E_test: {test_e[-1]}")
  print(f"Misclassified train set: {classif_error[-1]}") # this is added so now I am unsure also it says train and test


  return


def expirements_momentum(input_train, input_vali, input_test, initial_weights, learning_rates, max_iter):
  learning_rates = [0.001, 0.01, 0.1, 0.5]
  alphas = [0.1, 0.4, 0.6, 0.8]
  for learning_rate in learning_rates:
    for alpha in alphas:
      start = time.time()
      par = gradient_descent_momentum(input_train, input_vali, input_test, initial_weights, learning_rate, max_iter, alpha)
      momentum_analytics(par, learning_rate, alpha)
      end = time.time()
      print(f"CPU time: {end-start}")
  return

# --- Weight Decay
def cost_weight_decay(output, target, weights, data, lambda_):
  eps = 1e-15  # Small epsilon value
  output = np.clip(output, eps, 1 - eps)
  cost =  -np.mean(target * np.log(output) + (1 - target) * np.log(1 - output))
  weight_decay = (lambda_ / (2* data.shape[0])*np.sum(np.square(weights)))
  return cost + weight_decay

def sigmoid_weight_decay(output, target, data, lambda_, weights):
    N = len(output)
    gradient = ((output - target) @ data) / N + (lambda_ / N) * weights
    return gradient

def decay_update_rule(current_weights, data, target, learning_rate, momentum, alpha, lambda_):
    current_output = output(current_weights, data)
    momentum = -learning_rate * sigmoid_weight_decay(current_output, target, data, lambda_, current_weights) + alpha * momentum
    weights = current_weights + momentum
    return weights, momentum

def gradient_descent_weight_decay(input_train, input_vali, input_test, weights, learning_rate, lambda_, max_iter, alpha):
  x,t = input_train
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
    train_e[i] = cost_weight_decay(output(weights, x), t, weights, x, lambda_)

    validation_e[i] = cost_weight_decay(output(weights, x_vali), t_vali, weights, x_vali, lambda_)
    test_e[i] = cost_weight_decay(output(weights, x_test), t_test, weights, x_test, lambda_)
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
  plt.figure(figsize=(10,6))
  plt.plot(train_e, label = "Training error")
  plt.plot(validation_e, label = "Validation error")
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
  par = gradient_descent_weight_decay(input_train, input_vali, input_test, initial_weights, learning_rate, lambda_, max_iter, alpha)
  weight_decay_analytics(par, lambda_)
  end = time.time()
  print(f"CPU time: {end-start}")
  return

# --- Newton Method

# --- Line search

#from line_search import line_search, line_search_analytics

# --- Conjugate gradient descent

# --- Stochastic gradient descent

if __name__ == "__main__":
  weights = np.random.rand(X_train_norm_gd.shape[1])
  experiments_gradient_descent((X_train_norm_gd, Y_train_norm_gd), (X_train_norm_val, Y_train_norm_val), (X_test_norm, Y_test_norm), weights, max_iter=10000)
  expirements_momentum((X_train_norm_gd, Y_train_norm_gd), (X_train_norm_val, Y_train_norm_val), (X_test_norm, Y_test_norm), weights, 0.01, max_iter=10000)
  experiments_weight_decay((X_train_norm_gd, Y_train_norm_gd), (X_train_norm_val, Y_train_norm_val), (X_test_norm, Y_test_norm), weights, 0.01, max_iter=10000, alpha=0.8)
  #idx, weights, train_e, validation_e, test_e, classif_error = gradient_descent_momentum((X_train_norm_gd, Y_train_norm_gd), (X_train_norm_val, Y_train_norm_val), (X_test_norm, Y_test_norm), weights, 0.001, 100)
