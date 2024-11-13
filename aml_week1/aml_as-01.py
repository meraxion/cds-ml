import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
def box_muller(mu = 0, sigma=1):
  # procedure:
  # sample from a uniform distribution x1 ~ [0,1], x2 ~ [0,1]
  x = sps.uniform()

  num_samples = 10_000

  ys = np.zeros((num_samples,2))

  for i in range(num_samples):
    x1, x2, = x.rvs(size=2)

    y1 = np.sqrt(-2*np.log(x1))*np.cos(2*np.pi*x2)
    y2 = np.sqrt(-2*np.log(x1))*np.sin(2*np.pi*x2)

    ys[i] = mu + sigma*y1,y2

  # plot histogram of y1
  plt.hist(ys[:,0], 100, density=True, label=r"Density histogram of $y$")
  # plot 1-d normal distribution over histogram
  xs = np.linspace(sps.norm.ppf(0.01),sps.norm.ppf(0.99), 100)

  plt.plot(xs, sps.norm.pdf(xs), "r--", label="Gaussian Probability Density")
  plt.xlabel("y")
  plt.ylabel(r"p(y)")
  plt.legend()
  plt.show()

box_muller()