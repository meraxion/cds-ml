import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb

# Perceptron learning rule

# initial parameters and initial line
# input vectors = {1, 2, 3, 4}
# corresponding labels = {a, b, b, a}

# sign(w0T*input vectors^mu) = labels0^mu
# weight vector0 = {0, 0, 0, 0}
w0 = [0.1, 0.1, 0.1]

# for i in range(max):

#update step
# weights at w^n-1; new (x^n, y0^n)
# if y0^n(w^n-1Tx^n) > 0:
# w[n] = w[n-1]
# elif y0[n](w[n-1]^T* x[n]) < 0
# w[n] = w[n-1] + eta*y0[m]*x[n]
# new w = old w + learning rate*d(1 if in upper, -1 in  lower) 
# *xvalue corresponding to update)

def update_step(x, y, w, learning_rate):
    if y*(w @ x) > 0:
        wnext = w
    else:
        wnext = w + learning_rate*y*x
    
    return wnext

#C(N,P) = 
# PN = P/N

plt.figure(figsize=(16,6))
#plt.plot ()

# ---
def capacity(N:int, P:int):
    sum = 0
    for i in range(0, N):
        sum += comb(P-1, i)
    return 2*sum

def bound(N:int, P:int):
    
    return np.e

def three():
    """
    function for running all of exercise 3, getting plots etc
    can be thought of as "main"

    It should:
     - for N = 50, and P between 1 and 200,
     - numerically compute the capacity of C(N,P),
     - as well as the estimated bound 
    """

    return