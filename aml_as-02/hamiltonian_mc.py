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

def hmc(x0:np.ndarray,
        energy_fn:Callable[[np.ndarray], float],
        n_samples:int,
        eps:float = 0.01,
        tau:int   = 1000) -> tuple[np.ndarray, np.ndarray]:
  """
  Run Hamiltonian Monte Carlo sampling

  args:
    x0: initial position of the sampling process. array of shape (d,)
    energy_fn: function that computes potential energy at position x
    n_samples: number of samples to generate
    eps: step size for leapfrog integration
    tau: number of leapfrog steps per iteration

  returns:
    x: array of shape (n_samples, d), containing samples
    accept_ratios: array of shape (n_samples,) the acceptance ratio over time for this run
  """

  # Setup different arrays and counters
  accept_ratios = np.zeros((n_samples,))
  num_accepts   = 0
  x = np.zeros((n_samples, x0.shape[0]), dtype=np.float32)
  x[0] = x0

  rho_dist = sps.multivariate_normal([0.,0.], a)
  rho = rho_dist.rvs()

  g = jax.grad(energy_fn)(x0)
  e = energy_fn(x0)

  for i in range(n_samples):
    x_new = x[i].copy()
    rho = rho_dist.rvs()
    g_new = g
    H = hamiltonian(e, rho)

    # start leapfrog
    for t in range(tau):
      rho = rho - 0.5 * eps * g_new
      x_new = x_new + eps*rho
      g_new = jax.grad(energy_fn)(x_new)
      rho = rho - 0.5 * eps * g_new

    e_new = energy_fn(x_new)
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

  return x, accept_ratios

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
   
  n_samples = 20_000
  eps = 0.01
  Tau = 1_000

  accept_ratios = np.zeros((n_samples,))
  num_accepts   = 0

  x0 = np.asarray([5., 3.])

  x, accepts = hmc(x0, E, n_samples)

if __name__ == "__main__":
  main()