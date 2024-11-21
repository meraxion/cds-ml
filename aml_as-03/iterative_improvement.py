import numpy as np

from makedata import make_data

# return neighbour set
def get_neighbour(state):
    # todo sample random flip in the state
    return state

def sample_state():
    # 500 random -1 and 1 (idk if I need 0)
    return state

def energy_calc(state):
    500,500 and x is 500
    energy = - 1/2 *x.T*w*x
    return energy
num_iterations = 100

data = make_data().toarray()
states = np.zeros(num_iterations)
states[0] = sample_state()

for i in range(num_iterations):
    current_state = states[i]
    neighbour_state = get_neighbour(states[i])
    if energy_calc(neighbour_state) < energy_calc(current_state):
        states[i+1] = neighbour_state
    else:
        states[i+1] = current_state

print(states)



