import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

def plambda(lam, r):

  unnorm = np.exp(- lam)*lam**r
  norm = unnorm/np.sum(unnorm)
  return norm

def p1lambda(lam, r):

  unnorm = np.exp(1-r)*(r-1)**(r-1)*np.exp(-(1/(2*r-2))*(lam - (r-1))**2)
  norm = unnorm/np.sum(unnorm)
  return norm

def p2lambda(lam, r):

  unnorm = r**r * np.exp(-r) * np.exp(-(r/2)*(np.log(lam) - np.log(r))**2)
  norm = unnorm/np.sum(unnorm)

  return norm

def main():
  xs = np.linspace(0, 20, 1_000)

  ys0_2 = plambda(xs, 2)
  ys1_2 = p1lambda(xs, 2)
  ys2_2 = p2lambda(xs, 2)

  plt.plot(xs, ys0_2, label=r"$p(\lambda)$")
  plt.plot(xs, ys1_2, label=r"$p_1(\lambda)$")
  plt.plot(xs, ys2_2, label=r"$p_2(\lambda)$")
  plt.title("r = 2")
  plt.legend()
  plt.show()

  ys0_10 = plambda(xs, 10)
  ys1_10 = p1lambda(xs, 10)
  ys2_10 = p2lambda(xs, 10)

  plt.plot(xs, ys0_10, label = r"$p(\lambda)$")
  plt.plot(xs, ys1_10, label = r"$p_1(\lambda)$")
  plt.plot(xs, ys2_10, label = r"$p_2(\lambda)$")
  plt.title("r = 10")
  plt.legend()
  plt.show()


  return

if __name__ == "__main__":
  main()