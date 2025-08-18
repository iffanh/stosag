from sklearn.datasets import make_spd_matrix
from main import stosag
import numpy as np

def rosenbrock_uncertainty(x, c1:np.ndarray, c2:np.ndarray) -> np.ndarray:
    """Rosenbrock function."""
    return 100.0 * (c1*x[1] - x[0]**2.0)**2.0 + np.sin(c2)*(1 - x[0])**2.0

N = 2  # number of well parameters
M = 100  # number of ensemble realizations

Ct = make_spd_matrix(N, random_state=42) # there are many ways to generate a covariance matrix, this is just one example. See references on how to generate covariance matrices specific for this problem.

# Create ensemble members
c1 = np.random.rand(M)
c2 = np.random.rand(M)*2*np.pi

rosenbrock = lambda x: rosenbrock_uncertainty(x, c1, c2)
robustness_measure = lambda x: np.mean(rosenbrock(x))

# initialize u
u0 = np.random.rand(M) # either mean value or current best iterate value
u = np.random.rand(N, M)

stosag_instance = stosag(u, u0, rosenbrock, Ct)


# j0 = rosenbrock(u0, c1, c2)  # calculate j0 based on the Rosenbrock function
# j = rosenbrock(u, c1, c2)  # calculate j based on the Rosenbrock function


# j = np.random.rand(M)
# j0 = 0.2

# U = value_shifted_array(u, u0) # Equation (5), (7), or (19). We use (19) as recommended in the paper
# J = j - j0 # Equation (6), (18), or (20). We use (20) as recommended in the paper