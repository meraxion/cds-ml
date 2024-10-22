# from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd
import scipy.stats as sps

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
    def __init__(self, clusters:int, features:int, data):
        self.clusters = clusters
        self.features = features # do we need to specify features in a way?
        self.data = data

        self.pi = np.random.rand(clusters)
        self.pi /= self.pi.sum()

        self.mu_jk = np.random.rand(clusters, features)

    def likelihood(self, mu_jk):

        pxsk = np.zeros((self.data.shape[0], self.clusters))
        for i in range(self.data.shape[0]):
            for k in range(self.clusters):
                product = 1
                for j in range(self.data.shape[1]):
                    product *= mu_jk[k, j]**data[i, j]*(1- mu_jk[k, j])**(1-data[i, j])
                pxsk[i, k] = product
        return pxsk
    
    def prior(self):
        pxst = np.zeros(self.data.shape[1])
        pxsk = self.likelihood(self.mu_jk)
        for i in range(self.data.shape[1]):
            for k in range(self.clusters):
                pxst[i] += self.pi[k]*pxsk[i, k]
        return pxst
            
        

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


noise = sps.multivariate_normal(cov=np.eye(2)*0.1)
dp1 = [1, 0] 
dp2 = [1, 0] 
dp3 = [0, 1] 
dp4 = [0, 1] 

data = np.array([dp1, dp2, dp3, dp4])
MM = MixtureModel(2, 2, data)
MM.likelihood()


data = np.load('train_data.npz')
X_train_norm = data['X_train_norm']