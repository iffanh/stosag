from sklearn.datasets import make_spd_matrix
from stosag import stosag
from scipy.stats import norm
import numpy as np

def rosenbrock_uncertainty(x, c1:np.ndarray, c2:np.ndarray) -> np.ndarray:
    """Rosenbrock function."""
    return 100.0 * (c1*x[1] - x[0]**2.0)**2.0 + np.sin(c2)*(1 - x[0])**2.0

N = 2  # number of well parameters
M = 100  # number of ensemble realizations
lb = np.array([-2.0, -2.0])  # lower bound
ub = np.array([2.0, 2.0])  # upper bound

Ct = make_spd_matrix(N, random_state=42)*0.1 # there are many ways to generate a covariance matrix, this is just one example. See references on how to generate covariance matrices specific for this problem.

# Create ensemble members
c1 = norm(1, 0.01).rvs(size=M)  # Randomly generated c1 values
c2 = norm(np.pi, 0.01*np.pi).rvs(size=M)  # Randomly generated c2 values

rosenbrocks = lambda x: np.array([rosenbrock_uncertainty(x, _c1, _c2) for _c1, _c2 in zip(c1, c2)]) # should also identify the number of ensemble realizations M

# initialize u
# u0 = np.random.rand(N) # either mean value or current best iterate value
u0 = np.array([-1.0, -1.0])  # Initial iterate value

stosag_instance = stosag(u0, 
                         rosenbrocks,
                         lb, 
                         ub, 
                         M, 
                         Ct,
                         constants={'mode': 4,
                                    'max_iter': 10000, 
                                    'line_search_max_iter': 20, 
                                    'line_search_alpha': 0.01})

stosag_instance.run()  # Run the optimization process

print("Number of evaluations:", stosag_instance.N_EVAL)
print("Best iterate values:", stosag_instance.x_list[-1])  # Get the last best iterate value
print("Function value at best iterate:", stosag_instance.j_list[-1])  
