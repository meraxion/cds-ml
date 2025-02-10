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