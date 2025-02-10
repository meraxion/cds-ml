"""
Pseudocode:
1. Compute <s_i>_c, <s_is_j>_c from the data
2. start with a random initial state w_ij, theta_ij
3. for t = 1, 2, ... do:
  4. estimate <s_i>, <s_is_j> using MH sampling
  5. theta_i := theta_i + eta(<s_i>_c - <s_i>)
  6. w_ij := w_ij + eta(<s_is_j>_c - <s_is_j>)


estimate free expectations using Monte Carlo sampling
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

"""
Exercise 1 Description:
- For small models (up to 20 pins) the computation can be done exactly. 
Make a toy problem by generating a random data set with 10-20 spins.
Define as convergence criterion that the change of the paramters of the BM is less than 10e-13. Demonstrate the convergence of the BM learning rule.
Show plot of the convergence of the likelihood over learning iterations.
"""

def random_small_dataset():

  return

# Subsetting data to small toy problem:


# Calculating Data Statistics
