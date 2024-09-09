import numpy as np
from scipy.special import factorial

"""Consider 11 urns with u = 0, 1, ..., 10 balls. 
Urn u has u black balls and 10-u white balls.
Select one urn at random and draw N times with replacement.
Suppose that the outcome after N=10 draws is that the number of black balls that have been drawn is __even__. What is the probability that urn u was selected?"""
# Likelihood, P(Be | N)
# This is the sum of the probabilities of each even number
# For urn 0, that would be (using a binomial likelihood):
def combinations(n: int, k:int):
  return factorial(n) / (factorial(n-k)*factorial(k))

def binomial(N: int, k:int, p:float):
  return combinations(N, k) * p**(k)*(1-p)**(N-k)

def one_urn(p: int, num_urns:int, N:int):
  """Probability of an even number of urns for one urn
  p: probability in this urn of drawing a black ball"""
  my_sum = 0
  for k in range(0, num_urns, 2):
    my_sum += binomial(N, k, p)

  return my_sum

def all_urns(ind_probs:list, num_urns:int, N:int):
  likelihoods = []
  for j in range(0, num_urns):
    likelihoods.append(one_urn(ind_probs[j], num_urns, N))

  return likelihoods

def marginal_prob(likelihoods:list[int], priors:list[int]):
  likelihoods = np.asarray(likelihoods)
  priors = np.asarray(priors)

  marginal = np.sum(likelihoods*priors)
  return marginal

def main():
  # Bayes:
  # Be = "probabilty that an even number of black balls was pulled"
  # P(u | Be, N) = P(Be | N)P(u) / P(Be)

  num_urns = 11 # Number of urns
  N = 10 # number of draws
  # Prior on each urn, P(u)
  priors = [1/num_urns for n in range(num_urns)]
  # Likelihood of one black ball, in each urn:
  ind_probs = [i/10 for i in range(0, 11, 1)]

  likelihoods = all_urns(ind_probs, num_urns, N)

  marginal = marginal_prob(likelihoods, priors)

  likelihoods = np.asarray(likelihoods)
  priors = np.asarray(priors)
  posterior = (likelihoods*priors)/marginal
  print(posterior)

  print(f" This adds up to: {np.sum(posterior)}, and is therefore a real probability.")

  return posterior

if __name__ == "__main__":
  post = main()