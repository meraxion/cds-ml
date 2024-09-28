<<<<<<< HEAD
timport numpy as np
=======
import numpy as np
from scipy.special import expit

>>>>>>> 8ac37d6bf905bcdb4cf8f91ed4d5636fc4b20159
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
  return expit(np.sum(weights * data, axis=0))

# gradient

def sigmoid_grad_calc(output, target, data):
  N = len(output)
  gradient = np.zeros_like(data)
  for i, _ in enumerate(data):
    gradient[i] = 1/N * np.sum((output-target)*data[i])

  return gradient

# Hessian

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