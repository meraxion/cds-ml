import time
import jax
import csv
import hashlib
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from jax import jit
from jax import Array
from functools import partial
from tqdm import tqdm
from typing import Callable

from makedata import make_data

def generator(cur_var):
    while cur_var > 0:
        yield

@partial(jit, static_argnums = 2)
def get_neighbour(key, state, R):
    """samples a state x' in R(x)"""

    positions = jr.choice(key, len(state), shape=(R,), replace=False)

    def flip_one(state, pos):
        return state.at[pos].set(-state[pos])
    
    new_state = jax.lax.fori_loop(0, R, 
                                  lambda i, s: flip_one(s, positions[i]),
                                  state)

    return new_state

@partial(jit, static_argnums = 1)
def reset(key, n:int):
    """"""
    """
    samples a starting state of shape (n,), where all values of the state are -1 or 1
    """
    unifs = jr.normal(key, (n,))
    state = unifs>0.5
    state = 2*state - 1
    return state

@jax.jit
def energy_calc(state, data):
    energy = - 1/2 *state.T@data@state
    return energy

@jax.jit
def compute_acceptance(state, new_state, data, beta):
    
    return jnp.exp(-beta*(energy_calc(new_state, data)-energy_calc(state, data)))

def sim_annealing(key, state:Array, data:Array, R:int, beta:float, chain_len:int):

    def body_fun(carry, tmp):
        state, key = carry
        key, sample_key, prob_key = jr.split(key, 3)

        current_state = state
        neighbour_state = get_neighbour(sample_key, state, R)
        
        acc = compute_acceptance(current_state, neighbour_state, data, beta)
        a = jnp.where(1 < acc, 1, acc)
        p = jr.uniform(prob_key)
        state = jnp.where(a > p, neighbour_state, current_state)
        return (state, key), state
    
    _, states = jax.lax.scan(body_fun, (state,key), (), length=chain_len)
    return states

jit_sim_annealing = jax.jit(sim_annealing, static_argnums=[3,5])

def exp_schedule(beta, f):
    return f*beta

def AK_schedule(beta, delta_beta, chain_var):
    return beta + delta_beta/jnp.sqrt(chain_var)

def ak_outer_fn():

    return

def main():
    save = False
    # save = True

    vec_len = 500
    chain_length = 500
    R = 1 # neighbourhood size

    data = jnp.array(np.loadtxt("w500"))
    key = jr.PRNGKey(42)

    # condition for while loops:
    def cond_fn(var_e):
        return var_e > 0

    # first, AK-schedule
    def ak_body_fn(key, state, beta, data, R, chain_length):
        states = jit_sim_annealing(key, state, data, R, beta, chain_length)
        energies = jax.vmap(energy_calc, (0, None))(states, data)
        mean_E = jnp.mean(energies)
        var_E  = jnp.var(energies)
        state = states[-1]
        beta   = AK_schedule(beta, delta_beta, var_E)
  
        return state, beta, mean_E, var_E
    
    means, vars, betas = [], [], []
    delta_beta = 0.1
    # 1 Run MH at high temperature to estimate beta_1:
    beta_0 = 0.0001 # start with high temp=low beta
    key, init_key, anneal_key = jr.split(key, 3)
    state = reset(init_key, vec_len)

    current_state, current_beta, mean_E, current_var = ak_body_fn(anneal_key, state, beta_0, data, R, chain_length)

    means.append(float(mean_E))
    vars.append(float(current_var))
    betas.append(float(current_beta))

    loop_i = 0

    for _ in tqdm(generator(current_var)):
        key, subkey = jr.split(key)
        current_state, current_beta, mean_E, current_var = ak_body_fn(subkey, current_state, current_beta, data, R, chain_length)

        means.append(float(mean_E))
        vars.append(float(current_var))
        betas.append(float(current_beta))   

        loop_i += 1

    xs = np.arange(loop_i)

    plt.plot(xs, means)
    plt.show()
    plt.plot(xs, vars)
    plt.show()
    plt.plot(xs, betas)
    plt.show()
  
    return means, vars, betas

if __name__ == "__main__":
    means, vars, beta = main()