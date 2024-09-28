timport numpy as np
# --- Problem definition
"""
consider the the logistic regression problem
the model is given in the exercise pdf, and Bishop section 4.3
"""

# Input data

# Targets

def cost(output, target):
  -1/len(target)*np.sum(target*np.log(output)+(1-target)*np.log(1-output))

# probability

# gradient

def sigmoid_grad_calc(output, target, data):
  N = len(output)
  gradient = np.zeros_like(data)
  for i, _ in enumerate(data):
    gradient[i] = 1/N * np.sum((output-target)*data[i])

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