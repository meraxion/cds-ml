import numpy as np
from matplotlib import pyplot as plt
from scipy.special import expit
from scipy.optimize import minimize
from functools import partial
import time

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
def update_rule(current_weights, data, target, learning_rate):
  current_output = output(current_weights, data)
  weight_update = current_weights - learning_rate * sigmoid_grad_calc(current_output, target, data)
  return weight_update