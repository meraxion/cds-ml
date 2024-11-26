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

from makedata import make_data

# return neighbour set
@partial(jit, static_argnums = 2)
def get_neighbour(key, state, R):

    positions = jr.choice(key, len(state), shape=(R,), replace=False)

    def flip_one(state, pos):
        return state.at[pos].set(-state[pos])
    
    new_state = jax.lax.fori_loop(0, R, 
                                  lambda i, s: flip_one(s, positions[i]),
                                  state)

    return new_state

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

def hash_state(state):
    "converts a state to bytes and hashes it"

    return hashlib.sha256(state.tobytes()).hexdigest()

@jax.jit
def track_min_energy_state(curr_min_e, curr_min_state, new_energies, new_states):

    batch_min_idx = jnp.argmin(new_energies)
    batch_min_e   = new_energies[batch_min_idx]
    batch_min_state = new_states[batch_min_idx]

    min_e = jnp.where(batch_min_e < curr_min_e,
                      batch_min_e, curr_min_e)
    min_state = jnp.where(batch_min_e < curr_min_e,
                          batch_min_state, curr_min_state)
    
    return min_e, min_state

@partial(jit, static_argnums = [2, 3])
def iterative_improvement(key, data, vec_len, R, num_iterations):
    state = sample_state(key, vec_len)

    def body_fun(i, carry):
        state, key = carry
        key, subkey = jr.split(key)
        current_state = state
        neighbour_state = get_neighbour(subkey, state, R)

        neighbour_energy = energy_calc(neighbour_state, data)
        current_energy   = energy_calc(current_state, data) 
        
        state = jnp.where(neighbour_energy < current_energy,
                          neighbour_state, current_state)
        return (state, key)

    final_state, _ = jax.lax.fori_loop(0, num_iterations-1, body_fun, (state,key))
    return final_state

@partial(jit, static_argnums = [1, 2, 4, 5])
def K_num_restarts(key, K, R, data, vec_len, num_iterations):
    keys = jr.split(key, K)

    def single_run(run_key):
        state = iterative_improvement(run_key, data, vec_len, R, num_iterations)
        return energy_calc(state, data), state
    
    energies, states = jax.vmap(single_run)(keys)

    best_idx = jnp.argmin(energies)
    return energies[best_idx], states[best_idx]

@partial(jit, static_argnums = [1, 2, 3, 5, 6])
def N_num_runs(key, num_runs, K, R, data, vec_len, num_iterations):
    keys = jr.split(key, num_runs)

    run_fn = lambda skey: K_num_restarts(skey, K, R, data, vec_len, num_iterations)
    energies, states = jax.vmap(run_fn)(keys)

    return energies, states

def threshold(mean, std_dev, eps):
    return std_dev > jnp.abs(mean)*eps

def main():
    save = False
    # save = True
    vec_len = 500
    num_iterations = 7500
    num_runs = 20
    # neighbourhood
    R = 1

    # rng key
    key = jr.PRNGKey(35)

    # a) for both ferromagnetic and frustrated, find number of restarts K needed to obtain reproducible results
    max_K = 5000
    eps = 0.01 # 1%

    # ferro-magnetic
    data = jnp.array(make_data(frustrated=False).toarray())
     
    print(f"Starting a) for ferro-magnetic system, with N = {num_runs} runs")
    for K in range(2, max_K):
        key, subkey = jr.split(key)
        start = time.time()
        energies, states = N_num_runs(subkey, num_runs, K, R, data, vec_len, num_iterations)
        end = time.time()
        runtime = end-start
        runtime /= num_runs

        mean_E = jnp.mean(energies)
        SD_E   = std_dev(mean_E, energies)

        if threshold(mean_E, SD_E, eps):
            print(f"Reproduced state at K={K}")
            print(f"Energy of the final solution was: {mean_E}")
            print(f"Standard Deviation of the final solution was: {SD_E}")

    # frustrated
    data = jnp.array(make_data().toarray())

    print(f"Starting a) for frustrated system, with N = {num_runs} runs")
    for K in range(2, max_K):
        key, subkey = jr.split(key)
        start = time.time()
        energies, states = N_num_runs(subkey, num_runs, K, R, data, vec_len, num_iterations)
        end = time.time()
        runtime = end-start
        runtime /= num_runs

        mean_E = jnp.mean(energies)
        SD_E   = std_dev(mean_E, energies)

        if threshold(mean_E, SD_E, eps):
            print(f"Reproduced state at K={K}")
            print(f"Energy of the final solution was: {mean_E}")
            print(f"Standard Deviation of the final solution was: {SD_E}")   

    # b) for frustrated 
    results = []

    # make frustrated data
    data = jnp.array(make_data().toarray())

    # Ks = [20, 100, 200, 500, 1000, 2000, 4000]
    # Ks = [20, 100, 200, 500]
    Ks = [20]

    min_e = 0
    min_state = 0

    for K in Ks:
        key, subkey = jr.split(key)
        start = time.time()
        energies, states = N_num_runs(subkey, num_runs, K, R, data, vec_len, num_iterations)
        end = time.time()
        runtime = end-start
        runtime /= num_runs

        mean_E = jnp.mean(energies)
        SD_E   = std_dev(mean_E, energies)

        results.append([K, runtime, mean_E, SD_E])

        min_e, min_state = track_min_energy_state(min_e, min_state,
                                                  energies, states)
        
    if save:
        file_name = "iter_improvement.csv"
        with open(file_name, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)

            csv_writer.writerows(results)

    jax.debug.print("min_e: {e}", e=min_e)
    return results

if __name__ == "__main__":
    results = main()
    print(results)