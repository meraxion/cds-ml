import numpy as np
from scipy import sparse
import typing

def make_data(frustrated:bool=True):
    # Set the random seed
    np.random.seed(0)
    n = 500
    p = 0.5

    # Calculate the number of non-zero entries in the upper triangle (including diagonal)
    num_nonzero = int(p * n * (n + 1) / 2)

    # Get indices for the upper triangle (including diagonal)
    upper_triangle_indices = np.triu_indices(n)
    total_upper_entries = len(upper_triangle_indices[0])

    # Randomly select indices for non-zero entries
    selected_indices = np.random.choice(total_upper_entries, size=num_nonzero, replace=False)
    i_indices = upper_triangle_indices[0][selected_indices]
    j_indices = upper_triangle_indices[1][selected_indices]

    # Generate random data between -1 and 1 for the non-zero entries
    data = np.random.uniform(-1, 1, size=num_nonzero)

    # Create a sparse matrix for the upper triangle
    w_upper = sparse.coo_matrix((data, (i_indices, j_indices)), shape=(n, n))

    # Symmetrize the matrix to make it symmetric
    w = w_upper + w_upper.T - sparse.diags(w_upper.diagonal())

    if frustrated:
        #  Apply (w > 0) - (w < 0) to define a frustrated system
        w = (w > 0).astype(int) - (w < 0).astype(int)
    else:
        # this choice defines a ferro-magnetic (easy) system
        w = ( w > 0).astype(int)


    # Remove diagonal entries by setting them to zero
    w.setdiag(0)
    return w
