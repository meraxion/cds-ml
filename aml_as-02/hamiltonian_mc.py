import numpy as np
import scipy.stats as sps
import jax
import jax.numpy as jnp

from typing import Callable
"""Pseudocode:
 - Choose initial x1
 - for t = 1 : T:
  1. Choose pt from N(0, a^-1), giving (xt,pt)
  2. Run Hamilton dynamics, giving (x', p')
  3. Metropolis step: accept (x(t+1), p(t+1) = (x',p') as new state with probability:
    - min(1,a), a = PH(x',p')/PH(x,p) = e^(-H(x',p'))/e^(-H(x,,p))
  4. On rejection, (xt+1, pt+1) = (xt, pt)
  """

x_init = np.asarray([0,0])

def hamiltonian(E:Callable, x, p):
  return E(x) + np.sum(np.pow(p,2))/2

def hmc():

  return

def main():
  """
  runs HMC for the elongated gaussian exercise
  """

  A = np.asarray([[250.25, -249.75], 
                  [-249.75, 250.25]])
  dims = A.shape[0]
  a = np.eye(dims)
  
  def find_E(x):
    def E(x):
      return 0.5 * x.T@A@x
    return E
  
  eps = 0.01
  Tau = 1_000

  times = np.arange(0, 100, eps)

  x = np.zeros((dims, times))
  x0 = [0, 0]
  x[0] = x0

  e = find_E(x0)
  g = jax.grad(hamiltonian, 1)(E, x[i], rho_new)

  for i, eps in range(times):
    rho_dist = sps.multivariate_normal(0, a)
    rho = rho_dist.rvs()
    # do some hamiltonian stuff here

    x_new = x[i]; rho_new = rho

    # start leapfrog
    for tau in range(Tau):
      rho_new = rho_new - 0.5 * eps * g
      x_new = x_new + eps*rho
      g_new = jax.grad(hamiltonian, 1)(e, x_new, rho_new)
      rho_new = rho_new - 0.5 * eps * g_new

    e_new = find_E(x_new)
    dH = np.exp(hamiltonian(e, x[i], rho) - hamiltonian(e_new, x_new, rho_new))
    accept_prob = np.min(1, dH)
    accept = 1 if accept_prob > np.random.rand() else 0

    if accept: 
      x[i+1] = x_new, g = g_new, e = e_new

  return x

if __name__ == "__main__":
  main()