import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb

# Perceptron learning rule

# --- Exercise 2

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
def two_a():
    P = 50
    N = 50
    w = np.zeros(N)
    c = update_step(P, N, w)
    print(c)

#b
def two_b():
    N = 50
    w = np.zeros(N)
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
        count = 0
        error = 0
        iterations = 0
        for _ in range(max_iter):
            converged = True
            for i in range(P):
                count += 1
                output = np.dot(x[i], w)
                if y[i]* output <= 0:
                    w += y[i]*x[i]
                    error += 1
                    converged = False
            iterations += 1 
            if converged:
                amount_converged += 1
                break
        iterations_list.append(iterations)
        error_list.append(error/count)

    fraction =  amount_converged/nruns
    mean_error = np.mean(error_list)
    std_error = np.std(error_list)

    mean_iterations = np.mean(iterations_list)
    std_iterations = np.std(iterations_list)
    return fraction, mean_iterations, std_iterations , mean_error, std_error

def two_c():
    N = 50
    P_range = np.arange(10, 130, 10)
    w = np.zeros(N)

    fraction = np.zeros_like(P_range)
    mean_iterations = np.zeros_like(P_range)
    std_iterations = np.zeros_like(P_range)
    mean_error = np.zeros_like(P_range)
    std_error = np.zeros_like(P_range)

    for i, P in enumerate(P_range):
        intermediate = learning_rulec(P, N, w)
        print(f"P = {P}: \n fraction: {intermediate[0]}, \n mean_iterations: {intermediate[1]}, std_iterations: {intermediate[2]} \n mean_error: {intermediate[3]}, std_error: {intermediate[4]}")

        # fraction[i], mean_iterations[i], std_iterations[i], mean_error[i], std_error[i] = learning_rulec(P, N, w)


  

# --- Exercise 3
def capacity(N:int, P):
    """
    Computes the capacity of C(N,P) for the specified values of C and N
    due to numpy optimization, if P is a numpy array, will compute elementwise on the array (I think)
    """
    sum = 0
    for i in range(0, N):
        sum += comb(P-1, i)
    return 2*sum

def bound(N:int, P:int):
    """
    Computes the bound using the function provided in the exercise
    """
    return (np.e*P/N)**N

def three():
    """
    function for running all of exercise 3, getting plots etc
    automatically called when running the file as __main__


    the function:
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

def eps(N:int, P:int, delta:float = 0.01):
    """
    expression for epsilon in terms of N, P, and delta, with appropriate defaults
    """
    return np.sqrt(-8*np.log(delta/(4*bound(N, 2*P)))/P)

def four_a():
    """
    function for running exercise 4a
    automatically called when running the file as __main__

    the function:
     - uses the bound from exercies 3 to approximate m(P)
     - computes for N = 10, 20, ..., 50 the dependence of epsilon on P
     - keeps track of the number of patterns P required to reach an epsilon < 0.1
     - plots epsilon as a function of P for the different levels of N
     - plots the number of patterns P required to reach epsilon < 0.1 for different levels of N

    """
    delta = 0.01
    Ns = [10, 20, 30, 40, 50]
    fin_Ps = [] # a list of P-values required for epsilon < 0.1
    # Compute numerically for N = 10, 20, ..., 50 the dependence of epsilon on P
    for N in Ns:
        P = 1_000
        Ps = []
        epsilons = []
        epsilon = 1
        while epsilon > 0.1:
            Ps.append(P)
            epsilon = eps(N, P, delta)
            epsilons.append(epsilon)
            P += 100

        fin_Ps.append(P)
        # Plotting epsilon as a function of P for each N
        plt.plot(Ps, epsilons, label=f"N = {N}")
    
    plt.legend()
    plt.title(r"$\epsilon$ as a function of P, for different values of N, until $\epsilon < 0.1$")
    plt.show()

    # Plotting the P required to reach eps < 0.1 as a function of N
    plt.plot(Ns, fin_Ps, "r+")
    plt.plot(Ns, fin_Ps, "g--")
    plt.title("The P required to reach $\epsilon < 0.1$, as a function of N")
    plt.show()
    return fin_Ps

def get_labels_from_X_and_W(X, W):
    return np.sign(W @ X)


def generate_set(P, N, W):
    X = np.random.choice([-1, 1], size=(N, P))
    Y = get_labels_from_X_and_W(X, W)
    return X, np.array(Y)

def get_generalization_error(X_test, Y_test, W):
    Y_pred = get_labels_from_X_and_W(X_test, W)
    error = np.where(Y_test != Y_pred)
    return len(error[0]) / len(Y_test)

def learning_rule_four_b(P, x, y, w, nruns=100, max_iter=1000):
    amount_converged = 0
    for _ in range(nruns):
        z = np.zeros_like(y)
        iterations = 0
        for _ in range(max_iter):
            converged = True
            for i in range(P):
                output = y[i] * np.dot(x[i], w)
                z[i] = output == y[i]
                if output <= 0:
                    w += y[i] * x[i]
                    converged = False
            iterations += 1
            if converged:
                amount_converged += 1
                break
    return w
def four_b():
    P = [10, 50, 100, 500, 1000]
    N = 10
    delta = 0.01
    W_teacher = np.random.randn(N)

    # generate test data
    X_test, Y_test = generate_set(10000, N, W_teacher)
    results = []
    for p in P:
        # generate train set
        X_train, Y_train = generate_set(p, N, W_teacher)

        error_list = []
        for i in range(100):
            w_student = np.random.randn(N)
            learning_rule_four_b(p, X_train.T, Y_train, w_student, nruns=100, max_iter=100000)
            error_list.append(get_generalization_error(X_test, Y_test, w_student))

        epsilon = eps(N, p, delta)
        print(f'When P = {p}, generalized error : {np.mean(error_list)}, theoretical error = {epsilon}')
if __name__ == "__main__":
    two_a()
    two_b()
    two_c()
    three()
    four_a()
    four_b()