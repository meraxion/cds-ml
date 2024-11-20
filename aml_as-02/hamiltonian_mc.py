import numpy as np
import scipy.stats as sps
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from typing import Callable
from tqdm import tqdm
from jax.random import PRNGKey
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
  return e + jnp.dot(p.T, p)/2

def run_leapfrog(rho, g, x, eps, energy_fn:Callable, tau):
  """
  Doing this for closure reasons
  """
  def scan_step(carry, _):
    rho, g, x = carry

    rho = rho - 0.5 * eps * g
    x = x + eps*rho
    g = jax.grad(energy_fn)(x)
    rho = rho - 0.5 * eps * g

    return (rho, g, x), None

  return jax.lax.scan(scan_step, (rho, g, x), jnp.arange(tau))[0]

def hmc(x0:Array,
        energy_fn:Callable[[Array], float],
        n_samples:int,
        eps:float = 0.01,
        tau:int   = 1000,
        key:Array = PRNGKey(13)
        ) -> tuple[Array, Array]:
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
  accept_ratios = jnp.zeros((n_samples-1,))
  num_accepts   = 0
  x = jnp.zeros((n_samples, x0.shape[0]), dtype=jnp.float32)
  x = x.at[0].set(x0)
  # x[0] = x0


  g = jax.grad(energy_fn)(x0)
  e = energy_fn(x0)

  for i in tqdm(range(n_samples-1)):
    key, gausskey, unifkey = jax.random.split(key, 3)
    x_new = x.at[i].get()
    rho = jr.normal(gausskey, x.shape[1])
    g_new = g
    H = hamiltonian(e, rho)

    # run leapfrog
    rho, g_new, x_new = run_leapfrog(rho, g_new, x_new, eps, energy_fn, tau)
    
    e_new = energy_fn(x_new)
    H_new = hamiltonian(e_new, rho)
    dH = (H_new - H)
    if dH < 0:
      accept = 1
      num_accepts += 1
    elif jr.uniform(unifkey) < jnp.exp(-dH):
      accept = 1
      num_accepts += 1
    else:
      accept = 0

    if accept: 
      x = x.at[i+1].set(x_new)
      g = g_new
    else:
      x = x.at[i+1].set(x[i].copy())
    accept_ratios = accept_ratios.at[i].set(num_accepts/(i+1))

  return x, accept_ratios

def main():
  """
  runs HMC for the elongated gaussian exercise
  """
  A = jnp.asarray([[250.25, -249.75], 
                  [-249.75, 250.25]])
  dims = A.shape[0]
  a = jnp.eye(dims)
  
  @jax.jit
  def E(x):
    return 0.5 * x.T@A@x
   
  n_samples = 1000
  eps = 0.01
  Tau = 100

  x0 = jnp.asarray([5., 3.])

  print(f"Running Hamiltonian Monte Carlo sampling run with: {n_samples} samples, leapfrog step size {eps}, and leapfrog steps {Tau}")
  x, accepts = hmc(x0, E, n_samples, eps, Tau)

  print(f"""
        The mean (vector) of this Gaussian is: {jnp.mean(x, axis=0)}.
        The mean (vector) of this Gaussian, discarding 500 steps of burn-in is: {jnp.mean(x[int(len(x)/2):], axis=0)}
        The final acceptance ratio was: {accepts[-1]}.
        """)
  print("fin")

if __name__ == "__main__":
  main()