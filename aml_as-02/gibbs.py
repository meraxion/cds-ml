import numpy as np
import matplotlib.pyplot as plt

N = 1000

mu = np.zeros(N)
beta = np.zeros(N)

alpha_0 = 1
beta_0 = 1
mu_0 = 0
sigma2_0 = 10

data = np.random.normal(0, 1, size=N)
x_bar = np.mean(data)

# start with a random sample before going to loop?
mu[0] = np.random.normal(mu_0, sigma2_0)
beta[0] = np.random.gamma(alpha_0, 1 / beta_0)

for i in range(1, N):
    # update mu
    sigma2_mu = 1 / (beta[i - 1] * N + 1 / sigma2_0)
    mu_mean = sigma2_mu * (beta[i - 1] * N * x_bar + mu_0 / sigma2_0)
    mu[i] = np.random.normal(mu_mean, np.sqrt(sigma2_mu))

    # update beta
    alpha = N / 2
    beta_0 = (1 / 2) * np.sum((data - mu[i]) ** 2)
    beta[i] = np.random.gamma(alpha, 1 / beta_0)

plt.plot(data, 'b-', label='data')
sample = np.random.normal(mu[-1], 1 / np.sqrt(beta[-1]), size=N)
plt.plot(sample, 'r--', label='sample')
plt.xlabel('samples')
plt.ylabel('x value')
plt.title('Posterior distribution with sampled mu and sigma')
plt.show()

print(f"Sampled mu: {mu[-1]}")
print(f"Sampled sigma: {1 / np.sqrt(beta[-1])}")
