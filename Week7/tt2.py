import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml

class MixtureModel:
    def __init__(self, clusters: int, features: int, data):
        self.clusters = clusters
        self.features = features
        self.data = data  # Data should be a NumPy array

        self.pi = np.ones(self.clusters) / self.clusters
        self.mu_jk = np.random.rand(clusters, features)
        # Normalize mu_jk to sum to 1 over features for each cluster
        self.mu_jk /= self.mu_jk.sum(axis=1, keepdims=True)

    def logprob_data_given_clusters(self):
        epsilon = 1e-10
        self.mu_jk = np.clip(self.mu_jk, epsilon, 1)
        log_mu_jk = np.log(self.mu_jk)
        pxsk = self.data @ log_mu_jk.T
        return pxsk

    def k_mu(self):
        epsilon = 1e-10
        self.pi = np.clip(self.pi, epsilon, 1)
        log_pi = np.log(self.pi)
        pxk = log_pi + self.logprob_data_given_clusters()
        return np.argmax(pxk, axis=1, keepdims=True)

    def N_k(self, kmu):
        N_k = np.zeros((self.clusters, 1))
        for k in range(self.clusters):
            N_k[k] = np.sum(kmu == k)
        return N_k

    def m_jk(self, N_k, kmu):
        m = np.zeros_like(self.mu_jk)
        for k in range(self.clusters):
            indices = np.where(kmu.flatten() == k)[0]
            if len(indices) == 0:
                continue  # Avoid division by zero
            total_counts = np.sum(self.data[indices], axis=0)
            m[k] = total_counts / total_counts.sum()
        return m

    def set_pi_k(self, N_k):
        self.pi = N_k.flatten() / N_k.sum()

    def reinitialize_empty_clusters(self, empty_clusters):
        for k in empty_clusters:
            self.mu_jk[k] = np.random.rand(self.features)
            self.mu_jk[k] /= self.mu_jk[k].sum()
            self.pi[k] = 1.0 / self.clusters
        self.pi /= self.pi.sum()
        print(f"Reinitialized empty clusters: {empty_clusters}")

    def plot(self):
        plt.figure(figsize=(10, 10))
        for i in range(self.clusters):
            image = self.mu_jk[i].reshape((28, 28))
            plt.subplot(1, self.clusters, i + 1)
            plt.imshow(image, cmap='gray')
            plt.title(f'Cluster {i+1}')
        plt.show()

    def run_model(self, max_iter=10):
        for t in range(max_iter):
            k_mu = self.k_mu()
            N_k = self.N_k(k_mu)
            empty_clusters = np.where(N_k == 0)[0]
            if len(empty_clusters) > 0:
                self.reinitialize_empty_clusters(empty_clusters)
                k_mu = self.k_mu()
                N_k = self.N_k(k_mu)
            self.set_pi_k(N_k)
            self.mu_jk = self.m_jk(N_k, k_mu)
        self.plot()

# Load the MNIST dataset and convert data to NumPy array
mnist = fetch_openml('mnist_784', version=1)
X_counts = mnist.data.astype(np.int32)[:1000].values  # Add .values to get NumPy array

MM = MixtureModel(10, X_counts.shape[1], X_counts)
MM.run_model()
