import numpy as np
from matplotlib import pyplot as plt
from scipy.special import expit
from scipy.optimize import minimize
from functools import partial

def sigmoid_grad_calc(output, target, data):
  N = len(output)
  gradient = np.zeros_like(data[1, :])
  for i, data_i in enumerate(data.T):
    gradient[i] = 1/N * np.sum((output-target)*data_i)

  return gradient


def sigmoid(weights, data):
  return expit(data@weights)
def update_rule_SGD(current_weights, data, target, learning_rate, batch_size):
  num_batches = int(np.ceil(data.shape[0] / batch_size))
  data_batched = np.array_split(data, num_batches)
  grad = np.zeros_like(current_weights)
  for batch in data_batched:
    current_output = sigmoid(current_weights, data)
    grad += sigmoid_grad_calc(current_output, target, batch)
  weight_update = current_weights - learning_rate * grad
  return weight_update