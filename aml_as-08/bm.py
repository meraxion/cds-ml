"""
Pseudocode:
1. Compute <s_i>_c, <s_is_j>_c from the data
2. start with a random initial state w_ij, theta_ij
3. for t = 1, 2, ... do:
  4. estimate <s_i>, <s_is_j>
  5. theta_i := theta_i + eta(<s_i>_c - <s_i>)
  6. w_ij := w_ij + eta(<s_is_j>_c - <s_is_j>)
"""
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array
from typing import Callable
from jax.random import PRNGKey
from tqdm import tqdm
import pandas as pd

# Load the data
# df = pd.read_csv("bint.txt", header=None, delimiter = " ")

# Initialize list to hold each row
rows = []

with open('bint.txt', 'r') as f:
    for line in f:
        # Parse each line into a numpy array (adjust 'sep' if needed)
        rows.append(jnp.fromstring(line, sep=' '))

# Combine rows into a 2D array
df = jnp.vstack(rows)
print(df.shape)

print("Data loaded")
df = jnp.array(df)
# restrict the dataset to a smaller one:
dfs = df[0:10, 0:953]

def data_statistics(df):
  mean = jnp.mean(df, axis = 1)
  cov  = jnp.cov(df)
 
  return mean, cov

"""
Exercise 1 Description:
- For small models (up to 20 spins) the computation can be done exactly. 
Make a toy problem by generating a random data set with 10-20 spins.
Define as convergence criterion that the change of the paramters of the BM is less than 10e-13. 
Demonstrate the convergence of the BM learning rule.
Show plot of the convergence of the likelihood over learning iterations.
"""

def random_small_dataset(key):

  key, subkey = jr.split(key)
  P = jr.uniform(subkey, minval=10, maxval=21)
  key, subkey = jr.split(key)
  df = jr.uniform(subkey, shape = (int(P.item()),))
  df = jnp.where(df>0.5, 1, 0)
  
  return jnp.atleast_2d(df)

def exact(df, key, eps:float = 1e-13):
  """
  For a small BM with no hidden units, solve the fixed point equations in the mean field and linear response approximation
  parameters:
  eps:float convergence criterion
  """
  key, subkey = jr.split(key)
  df = random_small_dataset(subkey)

  emp_mean, emp_cov = data_statistics(df)
  C = emp_cov - (emp_mean@emp_mean.T)
  delta = jnp.eye(len(emp_mean))
  w = delta/(1-jnp.pow(emp_mean, 2)) - jnp.linalg.inv(C)
  theta = jnp.arctanh(emp_mean) - w@emp_mean.T
   
  return w, theta

# Calculating Data Statistics

def main():
  key = PRNGKey(754273565)
  key, subkey = jr.split(key)
  w, theta = exact(df, subkey)

  
  return

if __name__ == "__main__":
   main()