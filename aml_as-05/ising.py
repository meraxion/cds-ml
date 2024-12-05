import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import itertools
from tqdm import tqdm

def s_all(n):
    """Generate all possible spin configurations for n spins"""
    configs = list(itertools.product([-1, 1], repeat=n))
    return np.array(configs)

def rms_mu(m_ex, m_approx):
   return np.sqrt(np.mean(m_ex-m_approx)**2)

# Set random seeds for reproducibility
np.random.seed(0)
def generate_w_matrix(J=0.5, c1=0.5, use_full_matrix=False):
  """
  J = temperature
  c1 = connectivity
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
      # c1 = 0.5  # connectivity is the approximate fraction of non-zero links
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

def exact(w, th, n=20):
  """
  w = matrix
  th = thresholds (theta_i)
  n  = matrix size
  """
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

  # print("Exact magnetizations:", m_ex)
  # print("\nCorrelation matrix shape:", chi_ex.shape)

  return m_ex, chi_ex, p_ex

# Belief-Propagation Implementation
def belief_prop(w, th, n=20, eta = 0.5, max_iter = 1000, tol=1e-13):
    """
    w is the coupling matrix
    th is the threshold 
    n is size of our lattice
    eta is smoothing factor
    """
  
    # start with initial random nxn matrix a
    a = np.random.randn(n, n)
    mij_plus  = np.ones((n,n))
    mij_minus = np.ones((n,n))
    da = 1

    print("")
    for i in range(max_iter):
      a_old = a
    
      # update all msgs
      mij_plus  = eta*mij_plus + (1-eta)*2*np.cosh( w + th + np.sum(a_old, axis=1)[:,np.newaxis]-a_old)
      mij_minus = eta*mij_minus + (1-eta)*2*np.cosh(-w + th + np.sum(a_old, axis=1)[:,np.newaxis]-a_old)

      a = 0.5 * np.log(mij_plus/mij_minus)

      da = np.max(np.max(np.abs(a - a_old)))
      if da < tol:
          break

    m = np.tanh(th + np.sum(a, axis=0))

    x_mat = np.asarray([[-1,-1],
                        [-1,1],
                        [1,-1],
                        [1,1]]).T
    
    a_term_1 = np.sum(a, axis=0)[:,np.newaxis]-a
    a_term_2 = np.sum(a, axis=1)[:,np.newaxis]-a
    
    Z = a_term_1@x_mat + a_term_2@x_mat

    one = w + 2*th + a_term_1 + a_term_2
    two = -w + a_term_1*1 + a_term_2*-1
    three = -w + a_term_1*-1 + a_term_2*1
    four = w - 2*th - a_term_1 + a_term_2

    b = np.exp([one, two, three, four])
    b = b/np.sum(b)

    xixj = np.array([1,-1,-1,1])
    sum_xi_xj_bij = np.sum(xixj * b)

    mi = np.sum(np.array([1,1,-1,-1])*b)
    mj = np.sum(np.array([1,-1,1,-1])*b)

    chi = sum_xi_xj_bij - mi*mj

    return m, chi, i

# Js = [0.5]
Js = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
Js = np.arange(0.2, 2.01, 0.2)
# Js = np.linspace(0.1, 2, 16)

mean_RMSs = [] # list for storing mean of RMS
std_RMSs  = [] # list for storing std dev of RMS
bp_iters  = []

M = 20 # multiple instances

# Parameters
n = 20  # number of spins
# Toggle between full and sparse Ising network
use_full_matrix = False
# Generate random thresholds

for J in Js:

  Rmss = []
  chi_exs = []
  p_exs = []

  m_bps = []
  i_bps = []

  print(f"Calculations for J = {J}")

  for m in tqdm(range(M)):
    th = np.random.randn(n) * J
    w, c = generate_w_matrix(J, 1, True)
    m_ex, chi_ex, p_ex = exact(w, th)
    m_bp, chi_bp, i_bp = belief_prop(w, th)

    Rmss.append(rms_mu(m_ex, m_bp))
    # bp_iters.append(i_bp)

  # A_bp = np.eye(n)/(1 - m_bp**2) - w
  # chi_bp = np.linalg.inv(A_bp)

  mean_RMSs.append(np.mean(Rmss))
  std_RMSs.append(np.std(Rmss))

mean_RMSs = np.array(mean_RMSs)
std_RMSs = np.array(std_RMSs)

plt.plot(Js, mean_RMSs, label="error bp")
plt.fill_between(Js, mean_RMSs+std_RMSs, mean_RMSs-std_RMSs, alpha=0.3)
plt.legend()
plt.show()