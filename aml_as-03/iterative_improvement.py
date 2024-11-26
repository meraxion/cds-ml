import time
import jax
import csv
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from jax import jit
from functools import partial

from makedata import make_data

# return neighbour set
@jax.jit
def get_neighbour(key, state):
    flip = jr.choice(key, len(state))
    v = state[flip]*-1
    state = state.at[flip].set(v)
    return state

@partial(jit, static_argnums = 1)
def sample_state(key, n:int):
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
def std_dev(mean, data):
    deviations = (data - mean)**2
    var = jnp.mean(deviations)

    return jnp.sqrt(var)

@partial(jit, static_argnums = 2)
def iterative_improvement(key, data, vec_len, num_iterations):
    state = sample_state(key, vec_len)

    def body_fun(i, carry):
        state, key = carry
        key, subkey = jr.split(key)
        current_state = state
        neighbour_state = get_neighbour(subkey, state)

        neighbour_energy = energy_calc(neighbour_state, data)
        current_energy   = energy_calc(current_state, data) 
        
        state = jnp.where(neighbour_energy < current_energy,
                          neighbour_state, current_state)
        return (state, key)

    final_state, _ = jax.lax.fori_loop(0, num_iterations-1, body_fun, (state,key))
    return final_state

@partial(jit, static_argnums = [1, 3, 4])
def K_num_restarts(key, K, data, vec_len, num_iterations):
    keys = jr.split(key, K)

    def single_run(run_key):
        state = iterative_improvement(run_key, data, vec_len, num_iterations)
        return energy_calc(state, data), state
    
    energies, states = jax.vmap(single_run)(keys)

    best_idx = jnp.argmin(energies)
    return energies[best_idx], states[best_idx]

@partial(jit, static_argnums = [1, 2, 4, 5])
def N_num_runs(key, num_runs, K, data, vec_len, num_iterations):
    keys = jr.split(key, num_runs)

    run_fn = lambda skey: K_num_restarts(skey, K, data, vec_len, num_iterations)
    energies, states = jax.vmap(run_fn)(keys)

    return energies, states

def main():
    save = False
    # save = True

    results = []

    vec_len = 500
    num_iterations = 100
    num_runs = 20

    # rng key
    key = jr.PRNGKey(42)

    # a) for both ferromagnetic and frustrated, find number of restarts K needed to obtain reproducible results

    # b) for frustrated, 

    # make frustrated data
    data = make_data().toarray()

    Ks = [20, 100, 200, 500, 1000, 2000, 4000]
    # Ks = [20, 100, 200, 500]
    # Ks = [20]

    for K in Ks:
        key, subkey = jr.split(key)
        start = time.time()
        Es, runtime = N_num_runs(subkey, num_runs, K, data, vec_len, num_iterations)
        end = time.time()
        runtime = end-start
        runtime /= num_runs

        mean_E = jnp.mean(Es)
        SD_E   = std_dev(mean_E, Es)

        results.append([K, runtime, mean_E, SD_E])

    if save:
        file_name = "iter_improvement.csv"
        with open(file_name, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)

            csv_writer.writerows(results)

    return results

if __name__ == "__main__":
    results = main()
    print(results)