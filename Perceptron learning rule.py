import matplotlib.pyplot as plt
import numpy as np
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


# 