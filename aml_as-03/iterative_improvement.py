import time
import jax
import csv
import numpy as np
import jax.numpy as jnp
import jax.random as jr

from makedata import make_data

# return neighbour set
def get_neighbour(key, state):
    flip = jr.choice(len(state))
    state[flip] *= -1
    # todo sample random flip in the state
    return state

def sample_state(key, n:int):
    """
    samples a starting state of shape (n,), where all values of the state are -1 or 1
    """
    unifs = jr.normal(key, (n,))
    state = unifs>0.5
    state = 2*state - 1
    return state

def energy_calc(state, data):
    energy = - 1/2 *state.T@data@state
    return energy

def std_dev(mean, data):
    deviations = data - mean
    var = jnp.mean(deviations)

    return jnp.sqrt(var)

def iterative_improvement(key, data, vec_len, num_iterations):
    states = np.zeros((num_iterations, vec_len))
    states[0] = sample_state(subkey, vec_len)

    for i in range(num_iterations):
        key, subkey = jr.split(key)
        current_state = states[i]
        neighbour_state = get_neighbour(key, states[i])
        if energy_calc(neighbour_state, data) < energy_calc(current_state, data):
            states[i+1] = neighbour_state
        else:
            states[i+1] = current_state
    return

def K_num_restarts(key, K, data, vec_len, num_iterations):

    e = 0

    for _ in range(K):
        key, subkey = jr.split(key)
        state = iterative_improvement(subkey, data, vec_len, num_iterations)
        
        new_e = (energy_calc(state, data))

        if new_e < e:
            e = new_e

    return e

def N_num_runs(subkey, num_runs, K, data, vec_len, num_iterations):

    Es = []

    for i in range(num_runs):
        key, subkey = jr.split(key)
        if i == 0:
            start = time.time()
            e = K_num_restarts(key, K, data, vec_len, num_iterations)
            end = time.time()
            runtime = end - start

        else:
            K_num_restarts(key, K, data, vec_len, num_iterations)
        Es.append(e)

    return Es, runtime

def main():
    save = False
    # save = True

    data = []

    vec_len = 500
    num_iterations = 100
    num_runs = 20

    # rng key
    key = jr.PRNGKey(42)

    data = make_data().toarray()

    # Ks = [20, 100, 200, 500, 1000, 2000, 4000]
    Ks = [20]

    for K in Ks:
        key, subkey = jr.split(key)
        Es, runtime = N_num_runs(subkey, num_runs, K, data, vec_len, num_iterations)

        mean_E = jnp.mean(Es)
        SD_E   = std_dev(mean_E, Es)

        data.append([K, runtime, mean_E, SD_E])

    if save:
        file_name = "iter_improvement.csv"
        with open(file_name, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)

            csv_writer.writerows(data)

    return data