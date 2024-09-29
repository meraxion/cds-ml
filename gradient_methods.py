import numpy as np
import numpy as np
from scipy.special import expit

# --- Problem definition
"""
consider the the logistic regression problem
the model is given in the exercise pdf, and Bishop section 4.3
"""

# Input data

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

# update rule


# early stopping


# --- Momentum

# --- Weight Decay

# --- Newton Method

# --- Line search

# --- Conjugate gradient descent

# --- Stochastic gradient descent

if __name__ == "__main__":
  print("hello world")