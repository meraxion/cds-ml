# from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd
import scipy.stats as sps
import matplotlib.pyplot as plt

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

    def pdata_given_clusters(self, mu_jk):

        pxsk = np.zeros((self.data.shape[0], self.clusters))
        for i in range(self.data.shape[0]):
            for k in range(self.clusters):
                product = 1
                for j in range(self.data.shape[1]):
                    product *= mu_jk[k, j]**data[i, j]*(1- mu_jk[k, j])**(1-data[i, j])
                pxsk[i, k] = product
        return pxsk
    
    def pdata_given_params(self):
        pxst = np.zeros(self.data.shape[1])
        pxsk = self.likelihood(self.mu_jk)
        for i in range(self.data.shape[1]):
            for k in range(self.clusters):
                pxst[i] += self.pi[k]*pxsk[i, k]
        return pxst
            
    def log_likelihood(self, k, lam):
        sum = 0
        lsum = 0
        for mu in range(self.data.shape[0]):
            sum += np.log(self.pi[k[mu]])

        for j in range(self.data.shape[1]):
            for mu in range(self.data.shape[0]):
                sum += self.data[mu, j]*np.log(self.mu_jk[j, k[mu]]) + (1-self.data[mu, j])*np.log(1- self.mu_jk[j, k[mu]])
                
        for k in range(self.clusters):
            lsum += self.pi[k]
        sum += lam*(lsum -1)
        return sum
        

    def k_mu(self):
        pxk = self.pi*self.pdata_given_clusters(self.mu_jk)
        return np.argmax(pxk, axis=1, keepdims=True)
    
    def delta_func(self, k, kmu):
        return np.where(k == kmu)
    
    def N_k(self, kmu):
        N_k = np.zeros((self.clusters, 1))
        for k in range(self.clusters):
            N_k[k] += self.delta_func(k, kmu)
        return N_k
    
    def pi_k(self, N_k):
        self.pi  = N_k/self.data.shape[0]
    
    def m_jk(self, N_k):
        m_jk = 0
        for j in range(self.data.shape[1]):
            for mu in range(self.data.shape[0]):
                m_jk += self.data[mu, j]
        m_jk /= N_k

        return m_jk
    
    def plot(self):
        plt.figure(figsize=(10, 10))
    
        for i in range(self.clusters):
            image = self.mu_jk[i].reshape((28,28))
            
            plt.subplot(1, self.clusters, i + 1)
            plt.imshow(image, cmap='gray')
            plt.title(f'Cluster {i+1}')
        
        plt.show()

    
    def run_model(self, max_iter = 1000):
        for t in range(max_iter):
            k_mu = self.k_mu()
            N_k = self.N_k(k_mu)
            self.pi_k(N_k)
            m_jk = self.m_jk()
            self.mu_jk = m_jk
        self.plot()
        return


# noise = sps.multivariate_normal(cov=np.eye(2)*0.1)
# dp1 = [1, 0] 
# dp2 = [1, 0] 
# dp3 = [0, 1] 
# dp4 = [0, 1] 

# data = np.array([dp1, dp2, dp3, dp4])

data = np.load('train_data.npz')
X_train_norm = data['X_train_norm']

MM = MixtureModel(10, X_train_norm.shape[1], X_train_norm)
MM.k_mu()


