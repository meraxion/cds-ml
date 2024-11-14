# In this exercise you are asked to sample from the posterior of learning problem. The
# learning task is the perceptron/logistic regression classification problem as explained in
# Mackay chapter 39 and 41. The sampling methods are the Metropolis Hasting method
# and the Hamilton Monte Carlo method as described in MacKay chapter 29 and 30. The
# data are given by the files x.ext (input patterns) and t.ext (output label).
# Write a computer program to sample from the distribution p(w|D, α) as given by
# MacKay Eqs. 41.8-10 using the Metropolis Hasting algorithm. Do the same using the
# Hamilton Monte Carlo method. For both methods, reproduce plots similar to fig. 41.5
# and estimate the burn in time that is required before the sampler reaches the equilibrium
# distribution. Investigate the acceptance ratio for both methods and try to optimize this
# by varying the proposal distribution, the step size  in HMC and the number of leap
# frog steps τ.


# chapter 29, 30 (page 38 something look for A matrix) <- metropolis and hasting explained
# 39 and 41 learning task is the perceptron/logistic regression classification problem
#  weight decay rates α = 0.01, 0.1,
#  We modify the objective function to: M(w) = G(w) + αEW (w) (39.22)
# where the simplest choice of regularizer is the weight decay regularizer (39.23)