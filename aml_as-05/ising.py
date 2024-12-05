import numpy as np
from scipy import sparse
import itertools
from tqdm import tqdm

def s_all(n):
    """Generate all possible spin configurations for n spins"""
    configs = list(itertools.product([-1, 1], repeat=n))
    return np.array(configs)

# Set random seeds for reproducibility
np.random.seed(0)

# Parameters
n = 20  # number of spins
Jth = 0.1  # sets the size of the random threshold values th

# Toggle between full and sparse Ising network
use_full_matrix = False

# Generate random thresholds
th = np.random.randn(n) * Jth

def generate_w_matrix(J=0.5, use_full_matrix=False):
  """
  J = temperature
  """
  if use_full_matrix:
      # Full weight matrix
      J0 = 0  # J0 and J are as defined for the SK model
      J = 0.5
      w = J0/n + J/np.sqrt(n) * np.random.randn(n, n)
      w = w - np.diag(np.diag(w))  # Zero out diagonal
      w = np.tril(w) + np.tril(w).T  # Make symmetric
      c = ~(w == 0)  # neighborhood graph fully connected
  else:
      # Sparse weight matrix
      c1 = 0.5  # connectivity is the approximate fraction of non-zero links
      k = c1 * n
      beta = 0.5
      
      # Create sparse symmetric random matrix
      # Note: sprandsym doesn't exist in scipy, so we create our own
      def sprandsym(n, density):
          nnz = int(density * n * n)
          # Create random coordinates for upper triangle
          coords = np.random.choice(n*n, size=nnz//2, replace=False)
          rows = coords // n
          cols = coords % n
          # Ensure we only take upper triangle elements
          mask = rows < cols
          rows = rows[mask]
          cols = cols[mask]
          data = np.random.randn(len(rows))
          # Create sparse matrix and make it symmetric
          w = sparse.coo_matrix((data, (rows, cols)), shape=(n, n))
          w = w + w.T
          return w.tocsr()
      
      w = sprandsym(n, c1)
      w = w - sparse.diags(w.diagonal())
      c = ~(w.toarray() == 0)  # sparse 0,1 neighborhood graph
      
      # Convert to +/- beta on the links
      w = beta * ((w > 0).astype(float) - (w < 0).astype(float))
      if sparse.issparse(w):
          w = w.toarray()
  return w, c

def exact(w, th, n):
  # EXACT calculations
  sa = s_all(n)  # all 2^n spin configurations
  # Calculate energies for all configurations
  Ea = 0.5 * np.sum(sa * (w @ sa.T).T, axis=1) + sa @ th
  Ea = np.exp(Ea)
  Z = np.sum(Ea)
  p_ex = Ea / Z  # probabilities of all 2^n configurations
  m_ex = sa.T @ p_ex  # exact mean values of n spins

  # Calculate exact connected correlations
  klad = (p_ex[:, np.newaxis] * np.ones((1, n))) * sa
  chi_ex = sa.T @ klad - np.outer(m_ex, m_ex)

  print("Exact magnetizations:", m_ex)
  print("\nCorrelation matrix shape:", chi_ex.shape)

  return m_ex, chi_ex

# Belief-Propagation Implementation
def belief_prop(w, th, n=20, max_iter = 100_000, tol=1e-12):
    """
    w is the coupling matrix
    th is the threshold 
    """
  
    # start with initial random nxn matrix a
    a = np.random.randn(n, n)
    da = 1

    for _ in tqdm(range(max_iter)):
      a_old = a
    
      # update all msgs
      mij_plus  = 2*np.cosh(w  + th + np.sum(a_old, axis=1)[:,np.newaxis]-a_old)
      mij_minus = 2*np.cosh(-w + th + np.sum(a_old, axis=1)[:,np.newaxis]-a_old)

      a = 0.5 * np.log(mij_plus/mij_minus)

      da = np.max(np.max(np.abs(a - a_old)))
      if da < tol:
          break

    m = np.tanh(th + np.sum(a, axis=0))
    return m

Js = [0.5]
# Js = np.linspace(0.2, 2, 15)

for J in Js:
  w = generate_w_matrix(J)
  m_exs, chi_exs = exact(w, th)
  m_ex = np.mean(m_exs)
  chi_ex = np.mean()
  m_bp = belief_prop(w, th)