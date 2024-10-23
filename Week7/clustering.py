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

    def logprob_data_given_clusters(self):

        # pxsk = np.zeros((self.data.shape[0], self.clusters))
        # for i in range(self.data.shape[0]):
        #     for k in range(self.clusters):
        #         sum = 0
        #         for j in range(self.data.shape[1]):
        #             sum += self.data[i,j]*np.log(self.mu_jk[k,j]) + (1-self.data[i,j])*np.log(1-self.mu_jk[k,j])
        #         pxsk[i, k] = sum
        pxsk = self.data@np.log(self.mu_jk).T + (1-self.data)@np.log(1-self.mu_jk).T
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
                sum += self.data[mu, j]*np.log(self.mu_jk[j, k[mu]]) + (1-self.data[mu, j])*np.log(1 - self.mu_jk[j, k[mu]])
                
        for k in range(self.clusters):
            lsum += self.pi[k]
        sum += lam*(lsum - 1)
        return sum
        
    def k_mu(self):
        pxk = np.log(self.pi) + np.exp(self.logprob_data_given_clusters())
        return np.argmax(pxk, axis=1, keepdims=True)
    
    def delta_func(self, k, kmu):
        ks = np.full_like(kmu, k)
        return np.where(ks == kmu, 1, 0)
    
    def N_k(self, kmu):
        N_k = np.zeros((self.clusters, 1))
        for k in range(self.clusters):
            N_k[k] = np.sum(self.delta_func(k, kmu))
        return N_k
    
    def get_pi_k(self):

        return np.exp(self.pi)
    
    def set_pi_k(self, N_k):
        self.pi  = N_k/self.data.shape[0]
    
    def m_jk(self, N_k, kmu):
        # m_jk = 0
        # for j in range(self.data.shape[1]):
        #     for mu in range(self.data.shape[0]):
        #         m_jk += self.data[mu, j]
        # m_jk /= N_k

        m = np.zeros_like(self.mu_jk)
        for j in range(self.features):
            for k in range(self.clusters):
                m[k, j] = np.sum(self.data[:,j]*self.delta_func(k, kmu))

        m = m/N_k
        return np.nan_to_num(m)
    
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
            # Diagnostic for argmax function:
            unique, counts = np.unique(k_mu, return_counts=True)
            print(dict(zip(unique, counts)))

            N_k = self.N_k(k_mu)
            m_jk = self.m_jk(N_k, k_mu)
            self.set_pi_k(N_k)
            self.mu_jk = m_jk
        self.plot()
        return


# noise = sps.multivariate_normal(cov=np.eye(2)*0.1)
# dp1 = [1, 0] 
# dp2 = [1, 0] 
# dp3 = [0, 1] 
# dp4 = [0, 1] 

# data = np.array([dp1, dp2, dp3, dp4])

df = np.load('train_data.npz')
X_train_norm = df['X_train_norm']
X_train_norm = X_train_norm[0:100]

MM = MixtureModel(10, X_train_norm.shape[1], X_train_norm)
MM.run_model()