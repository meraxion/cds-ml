# from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd

"""
1. Initialize k jk random.
 2. For t = 12
 (a) For = 1
 (b) For k = 1
 (c) For k = 1
 N compute k = argmaxkp(x k)
 K compute Nk. Set k := NkN
 K j=1 dcomputemjk. Set jk =mjk
"""

class MixtureModel:
    def __init__(self, clusters, features):
        self.clusters = clusters
        self.features = features # do we need to specify features in a way?

        self.pi = np.random.rand(clusters)
        self.pi /= self.pi.sum()

        self.mu_jk = np.random.rand(clusters, features)


    def k_mu(self, x_mu, k):
        return np.argmax()
    
    def N_k(self, deltak_kmu):
        N_k = np.sum(deltak_kmu)
        return N_k
    
    def m_jk(self, deltak_kmu):
        N_k = self.N_k(deltak_kmu)

        return
    
    def run_model(self):
        return






data = np.load('train_data.npz')
X_train_norm = data['X_train_norm']