import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb

# Perceptron learning rule

# initial parameters and initial line
P = 50
N = 50
w = np.zeros(N)


#update step

def update_step(P, N, w, learning_rate=1, max_iter=1000):
    x = np.random.choice([0, 1], size=(P, N))
    y = np.random.choice([-1, 1], size=P)
    converged = False
    for _ in range(max_iter):
        converged = True
        for i in range(P):
            if y[i]*np.dot(x[i], w) <= 0:
                w += y[i]*x[i]*learning_rate
                converged = False
        if converged:
            break
        
    return converged

#a?
c = update_step(P, N, w)
print(c)

#b
Prange = np.arange(10, 200, 10)
converged = {}

for P in Prange:
    c = update_step(P, N, w)
    converged[P] = c

print(converged)

"""
Reconstruct the curve C(PN) for N = 50 as a function of P in the following way. For
 each P construct a number (nruns) of learning problems randomly and compute 1) the
 fraction of these problems for which the perceptron learning rule converges, 2) the mean
 and std of the classification error on the training set and 3) the mean and std of the number
 of iterations until convergence.
"""

def learning_rulec(P, N, w, nruns=100, max_iter = 1000):
    x = np.random.choice([-1, 1], size=(P, N))
    y = np.random.choice([-1, 1], size=P)
    converged = False
    # iterations_list = []
    # another for loop???
    # iterations = 0
    for _ in range(max_iter):
        converged = True
        for i in range(P):
            if y[i]*np.dot(x[i], w) <= 0:
                w += y[i]*x[i]
                converged = False
        if converged:
            # add to amount converged???
            break
        # iterations += 1; does not work cause we want it for every time we measure iterations
    # iterations store in list maybe


    # fraction =  amount converged/ nruns
    # I do not know what they mean with 2) 

    # mean_iterations = np.mean(iterations)
    return
N = 50
P = np.arange(10, 120, 10)

plt.figure(figsize=(16,6))
#plt.plot ()

# ---
def capacity(N:int, P):
    sum = 0
    for i in range(0, N):
        sum += comb(P-1, i)
    return 2*sum

def bound(N:int, P:int):
    return (np.e*P/N)**N

def three():
    """
    function for running all of exercise 3, getting plots etc
    can be thought of as "main"

    It should:
     - for N = 50, and P between 1 and 200,
     - numerically compute the capacity of C(N,P),
     - as well as the estimated bound,
     - plot them
    """

    fig = plt.figure(figsize=(10,10))

    Ps = np.linspace(1,200,200)
    N = 50
    ys = capacity(N, Ps)

    xs = np.linspace(1, 200, 200)
    bounds = bound(N, xs)

    plt.plot(Ps, ys, label="Computed")
    plt.plot(xs, bounds, label="Bound")
    plt.yscale("log")
    plt.legend()
    plt.show()

    return

if __name__ == "__main__":
    three()