import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb

# Perceptron learning rule

# initial parameters and initial line
P = 50
N = 50
w = np.zeros(N)

# for i in range(max):

#update step

def update_step(P, N, w, learning_rate=1, max_iter=1000):
    x = np.random.choice([-1, 1], size=(P, N))
    y = np.random.choice([-1, 1], size=P)
    converged = False
    for _ in range(max_iter):
        for i in range(P):
            if y[i]*np.dot(x[i], w) <= 0:
                w += y[i]*x[i]
        if y[i]*np.dot(x[i], w) > 0:
            converged = True
            break
        
    return converged

#a?
c = update_step(P, N, w)
print(c)

#b
Prange = np.arange(10, 200, 20)
converged = {}

for P in Prange:
    c = update_step(P, N, w)
    converged[P] = c

print(converged)

def learning_rulec(P, N, w)

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