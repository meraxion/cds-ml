import time
import jax
import csv
import hashlib
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from jax import jit
from jax import Array
from functools import partial
from tqdm import tqdm
from typing import Callable

from makedata import make_data


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

@jax.jit
def sim_annealing(key, state, data, vec_len, R, beta, num_iterations):

    def body_fun(carry, tmp):
        state, key = carry
        key, sample_key, prob_key = jr.split(key)

        current_state = state
        neighbour_state = get_neighbour(sample_key, state, R)
        
        a = jnp.min(1, compute_acceptance(current_state, neighbour_state, 
                                          data, beta))
        p = jr.uniform(prob_key)
        state = jnp.where(a > p, neighbour_state, current_state)
        return (state, key), state
    
    _, states = jax.lax.scan(body_fun, (state,key), (), length=num_iterations)
    return states

def exp_schedule(beta, f):
    return f*beta

def AK_schedule(beta, delta_beta, chain_var):
    return beta + delta_beta/jnp.sqrt(chain_var)

def main():
    save = False
    # save = True

    vec_len = 500
    chain_length = 1000
    R = 1 # neighbourhood size

    data = make_data()
    key = jr.PRNGKey(42)


    # first, AK-schedule
    delta_beta = 0.001
    var_chain = 1
    # 1 Run MH at high temperature to estimate beta_1:
    beta_0 = 0.00001 # start with high temp=low beta
    key, subkey = jr.split(key)
    states = sim_annealing(subkey, data, vec_len, R, beta_0, AK_schedule, chain_length)



    

    


    
    return