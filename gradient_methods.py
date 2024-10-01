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
def early_stopping(vali_error, prev_vali_error):
  return vali_error > prev_vali_error

# gradient descent
def gradient_descent(val_train, val_vali, val_test, weights, learning_rate, max_iter):
  x,y = val_train
  x_vali, y_vali = val_vali
  x_test, y_test = val_test
  train_e = np.zeros(max_iter)
  validation_errors = np.zeros(max_iter)
  test_errors = np.zeros(max_iter)
  previous_validation_error = 1000

  classif_error = np.zeros(max_iter)
  for i in range(max_iter):
    weights = update_rule(weights, x, y, learning_rate)
    train_e[i] = cost(output(weights, x), y)

    validation_errors[i] = cost(output(weights, x_vali), y_vali)
    test_errors[i] = cost(output(weights, x_test), y_test)

    classif_error[i] = check_labels(weights, val_test)

    stop_check = early_stopping(validation_errors[i], previous_validation_error)
    if stop_check:
      break
    previous_validation_error = validation_errors[i]

  train_e = train_e[:i]
  validation_errors = validation_errors[:i]
  test_errors = test_errors[:i]
  return i, weights, train_e, validation_errors, test_errors, classif_error

def gradient_descent_analytics(par):
  i, weights, train_errors, validation_errors, test_errors, classif_error = par
  plt.figure(figsize=(10,6))
  plt.plot(train_errors, label = "Training error")
  plt.plot(validation_errors, label = "Validation error")
  plt.plot(test_errors, label="Test error")
  plt.xlabel("Iterations")
  plt.ylabel("Error")
  plt.legend()
  plt.show()

  print(f"Stopped after {i} iterations")
  print(f"E_train: {train_errors[-1]}, E_test: {test_errors[-1]}")
  print(f"Misclassified train set: {classif_error}") # this is added so now I am unsure also it says train and test


  return

def experiments_gradient_descent(val_train, val_vali, val_test, weights, max_iter):
  learning_rates = [1, 0.001, 0.01, 0.1, 0.5]
  for learning_rate in learning_rates:
    par = gradient_descent((X_train_norm_gd, Y_train_norm_gd), (X_train_norm_val, Y_train_norm_val), (X_test_norm, Y_test_norm), weights, learning_rate, 500)
    gradient_descent_analytics(par)



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
  classif_error = np.zeros(max_iter)
  previous_validation_error = 1000
  for i in range(max_iter):
    weights, momentum[i+1] = momentum_update_rule(weights, x, t, learning_rate, momentum[i], alpha = 0.8)
    train_errors[i] = cost(output(weights, x), t)

    validation_errors[i] = cost(output(weights, x_vali), y_vali)
    test_errors[i] = cost(output(weights, x_test), y_test)
    stop_check = early_stopping(validation_errors[i], previous_validation_error)
    classif_error[i] = check_labels(weights, val_test)
    if stop_check:
      break
    previous_validation_error = validation_errors[i]

  train_errors = train_errors[:i]
  validation_errors = validation_errors[:i]
  test_errors = test_errors[:i]

  return i, weights, train_errors, validation_errors, test_errors

def momentum_different_rates(val_train, val_vali, val_test, initial_weights, learning_rates, max_iter):
  return

def momentum_analytics(par):
  i, weights, train_errors, validation_errors, test_errors, classif_error = par
  plt.figure(figsize=(10,6))
  plt.plot(i, train_errors, label = "Training error")
  plt.plot(i, validation_errors, label = "Validation error")
  plt.plot(i, test_errors, label="Test error")
  plt.xlabel("Iterations")
  plt.ylabel("Error")
  plt.legend()
  plt.show()

  print(f"Stopped after {i} iterations")
  print(f"E_train: {train_errors[-1]}, E_test: {test_errors[-1]}")
  print(f"Misclassified train set: {classif_error[-1]}") # this is added so now I am unsure also it says train and test


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
  gradient = np.zeros_like(data[1, :])
  for i, data_i in enumerate(data.T):
    gradient[i] = 1/N * np.sum((output-target)*data_i) + (lambda_/data.shape[0])*weights[i]

  return gradient

def weight_decay_analytics(par):
  i, weights, train_errors, validation_errors, test_errors, classif_error = par
  plt.figure(figsize=(10,6))
  plt.plot(i, train_errors, label = "Training error")
  plt.plot(i, validation_errors, label = "Validation error")
  plt.plot(i, test_errors, label="Test error")
  plt.xlabel("Iterations")
  plt.ylabel("Error")
  plt.legend()
  plt.show()

  print(f"Stopped after {i} iterations")
  print(f"E_train: {train_errors[-1]}, E_test: {test_errors[-1]}")
  print(f"Misclassified train set: {classif_error[-1]}") 
  return

# --- Newton Method

# --- Line search

#from line_search import line_search, line_search_analytics

# --- Conjugate gradient descent

# --- Stochastic gradient descent

if __name__ == "__main__":
  weights = np.random.rand(X_train_norm_gd.shape[1])
  experiments_gradient_descent((X_train_norm_gd, Y_train_norm_gd), (X_train_norm_val, Y_train_norm_val), (X_test_norm, Y_test_norm), weights, 1000)
  #idx, weights, train_errors, validation_errors, test_errors, classif_error = gradient_descent_momentum((X_train_norm_gd, Y_train_norm_gd), (X_train_norm_val, Y_train_norm_val), (X_test_norm, Y_test_norm), weights, 0.001, 100)
