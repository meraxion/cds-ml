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

def hamiltonian(e:float, p:float):
  return e + np.dot(p.T, p)/2

def main():
  """
  runs HMC for the elongated gaussian exercise
  """

  A = np.asarray([[250.25, -249.75], 
                  [-249.75, 250.25]])
  dims = A.shape[0]
  a = np.eye(dims)
  
  def E(x):
    return 0.5 * x.T@A@x
   
  eps = 0.01
  Tau = 1_000

  times = np.arange(0, 100, eps)
  accept_ratios = np.zeros_like(times)
  num_accepts   = 0

  x = np.zeros((len(times), dims), dtype=np.float32)
  x0 = np.asarray([5., 3.])
  x[0] = x0

  rho_dist = sps.multivariate_normal([0.,0.], a)
  rho = rho_dist.rvs()

  g = jax.grad(E)(x0)
  e = E(x0)

  for i in range(len(times)):
    x_new = x[i].copy()
    rho = rho_dist.rvs()
    g_new = g
    H = hamiltonian(e, rho)

    # start leapfrog
    for tau in range(Tau):
      rho = rho - 0.5 * eps * g_new
      x_new = x_new + eps*rho
      g_new = jax.grad(E)(x_new)
      rho = rho - 0.5 * eps * g_new

    e_new = E(x_new)
    H_new = hamiltonian(e_new, rho)
    dH = (H_new - H)
    if dH < 0:
      accept = 1
      num_accepts += 1
    elif np.random.rand() < np.exp(-dH):
      accept = 1
      num_accepts += 1
    else:
      accept = 0

    if accept: 
      x[i+1] = x_new
      g = g_new
    else:
      x[i+1] = x[i]
    accept_ratios[i] = num_accepts/(i+1)

  return x

if __name__ == "__main__":
  main()