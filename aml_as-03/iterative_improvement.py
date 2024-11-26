import jax
import numpy as np
import jax.numpy as jnp
import jax.random as jr

from makedata import make_data

# return neighbour set
def get_neighbour(state):
    # todo sample random flip in the state
    return state

def sample_state(key, n:int):
    """
    samples a starting state of shape (n,), where all values of the state are -1 or 1
    """
    unifs = jr.normal(key, (n,))
    state = unifs>0.5
    state = 2*state - 1
    # 500 random -1 and 1 (idk if I need 0)
    return state

def energy_calc(state, data):
    energy = - 1/2 *state.T@data@state
    return energy

vec_len = 500
num_iterations = 100

# rng key
key = jr.PRNGKey(42)
key, subkey = jr.split(key)

data = make_data().toarray()
states = np.zeros((num_iterations, vec_len))
states[0] = sample_state(subkey, vec_len)

for i in range(num_iterations):
    current_state = states[i]
    neighbour_state = get_neighbour(states[i])
    if energy_calc(neighbour_state, data) < energy_calc(current_state, data):
        states[i+1] = neighbour_state
    else:
        states[i+1] = current_state

print(states)
