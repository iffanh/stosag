from sklearn.datasets import make_spd_matrix
from main import stosag
import numpy as np

def rosenbrock_uncertainty(x, c1:np.ndarray, c2:np.ndarray) -> np.ndarray:
    """Rosenbrock function."""
    return 100.0 * (c1*x[1] - x[0]**2.0)**2.0 + np.sin(c2)*(1 - x[0])**2.0

N = 2  # number of well parameters
M = 10  # number of ensemble realizations

Ct = make_spd_matrix(N, random_state=42) # there are many ways to generate a covariance matrix, this is just one example. See references on how to generate covariance matrices specific for this problem.

# Create ensemble members
c1 = np.random.rand(M)
c2 = np.random.rand(M)*2*np.pi

rosenbrocks = lambda x: np.array([rosenbrock_uncertainty(x, _c1, _c2) for _c1, _c2 in zip(c1, c2)]) # should also identify the number of ensemble realizations M

# initialize u
u0 = np.random.rand(N) # either mean value or current best iterate value
u = u0[:,np.newaxis] + 0.1*np.random.rand(N, M)

stosag_instance = stosag(u, u0, rosenbrocks, Ct)
