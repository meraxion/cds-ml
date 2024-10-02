import numpy as np
from gradient_methods import experiments_gradient_descent, expirements_momentum, experiments_weight_decay
from line_search import run_line_search
from SGD import experiments_SGD
from Newton_method import run_newton_method

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

if __name__ == "__main__":
  weights = np.random.rand(X_train_norm_gd.shape[1])
  experiments_gradient_descent((X_train_norm_gd, Y_train_norm_gd), (X_train_norm_val, Y_train_norm_val), (X_test_norm, Y_test_norm), weights, max_iter=10000)
  expirements_momentum((X_train_norm_gd, Y_train_norm_gd), (X_train_norm_val, Y_train_norm_val), (X_test_norm, Y_test_norm), weights, 0.01, max_iter=10000)
  experiments_weight_decay((X_train_norm_gd, Y_train_norm_gd), (X_train_norm_val, Y_train_norm_val), (X_test_norm, Y_test_norm), weights, 0.1, max_iter=5500, alpha=0.9)
  run_line_search()
  experiments_SGD((X_train_norm_gd, Y_train_norm_gd), (X_train_norm_val, Y_train_norm_val),
                               (X_test_norm, Y_test_norm), weights, 100)
  run_newton_method((X_train_norm_gd, Y_train_norm_gd), (X_train_norm_val, Y_train_norm_val), (X_test_norm, Y_test_norm), weights, max_iter=10)
