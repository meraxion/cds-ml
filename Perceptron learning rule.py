import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb

# Perceptron learning rule

# initial parameters and initial line
P = 50
N = 50
w = np.zeros(N)

def step(output):
    if output > 0:
        return 1
    else:
        return -1

def update_step(P, N, w, learning_rate=1, max_iter=1000):
    x = np.random.choice([0, 1], size=(P, N))
    y = np.random.choice([-1, 1], size=P)
    converged = False
    for _ in range(max_iter):
        converged = True
        for i in range(P):
            output = np.dot(x[i], w)
            output = step(output)
            if y[i] != output:
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
    iterations_list = []
    error_list = [] 
    amount_converged = 0
    for _ in range(nruns):
        x = np.random.choice([0, 1], size=(P, N))
        y = np.random.choice([-1, 1], size=P)
        z = np.zeros_like(y)
        converged = False
        iterations = 0
        for _ in range(max_iter):
            converged = True
            for i in range(P):
                output = y[i]*np.dot(x[i], w)
                z[i] = output==y[i]
                if output <= 0:
                    w += y[i]*x[i]
                    converged = False
            iterations += 1 
            if converged:
                amount_converged += 1
                break
        iterations_list.append(iterations)
        error_list.append(np.sum(z)/iterations)


    fraction =  amount_converged/ nruns
    mean_error = np.mean(error_list)
    std_error = np.std(error_list)

    mean_iterations = np.mean(iterations_list)
    std_iterations = np.std(iterations_list)
    return fraction, mean_iterations, std_iterations , mean_error, std_error


N = 50
P_range = np.arange(10, 120, 10)
w = np.zeros(N)

fraction = np.zeros_like(P_range)
mean_iterations = np.zeros_like(P_range)
std_iterations = np.zeros_like(P_range)
mean_error = np.zeros_like(P_range)
std_error = np.zeros_like(P_range)

for P in P_range:
    fraction[P], mean_iterations[P], std_iterations[P], mean_error[P], std_error[P] = learning_rulec(P, N, w)

print(fraction, mean_iterations, std_iterations, mean_error, std_error)

plt.bar()
#plt.plot () I am also still struggling with how to plot these results?

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