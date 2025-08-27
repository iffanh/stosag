import unittest

from sklearn.datasets import make_spd_matrix
from stosag.stosag import stosag
from scipy.stats import norm
import numpy as np

class RosenbrockTest(unittest.TestCase):
    
    def test_rosenbrock(self):
        def rosenbrock_uncertainty(x, c1:np.ndarray, c2:np.ndarray) -> np.ndarray:
            """Rosenbrock function.
            In STOSAG, we must define the function two ways: one for single evaluation and one for ensemble evaluation.
            For single evaluation, we assume x is a 1D array and return a 1D vector of function values corresponding to all the realizations.
            For ensemble evaluation, we assume x is a 2D array where each row corresponds to an ensemble member and return a 1D vector of function values.
            """
            
            if x.ndim == 1:
                return 100.0 * (c1*x[1] - x[0]**2.0)**2.0 + np.sin(c2)*(1 - x[0])**2.0
            else:
                
                for i in range(x.shape[0]):
                    if i == 0:
                        j = 100.0 * (c1[i]*x[i, 1] - x[i, 0]**2.0)**2.0 + np.sin(c2[i])*(1 - x[i, 0])**2.0
                    else:
                        j = np.append(j, 100.0 * (c1[i]*x[i, 1] - x[i, 0]**2.0)**2.0 + np.sin(c2[i])*(1 - x[i, 0])**2.0)
                return j

        N = 2  # number of well parameters
        M = 10 # number of ensemble realizations
        lb = [-2.0, -2.0]  # lower bound
        ub = [2.0, 2.0]  # upper bound

        Ct = make_spd_matrix(N, random_state=42)*0.1 # there are many ways to generate a covariance matrix, this is just one example. See references on how to generate covariance matrices specific for this problem.

        # Create ensemble members
        c1 = norm(1, 0.001).rvs(size=M)  # Randomly generated c1 values
        c2 = norm(np.pi/2, 0.001*np.pi).rvs(size=M)  # Randomly generated c2 values

        # rosenbrocks = lambda x: np.array([rosenbrock_uncertainty(x, _c1, _c2) for _c1, _c2 in zip(c1, c2)]) # should also identify the number of ensemble realizations M
        rosenbrocks = lambda x: rosenbrock_uncertainty(x, c1, c2) # should also identify the number of ensemble realizations M
        # initialize u
        # u0 = np.random.rand(N) # either mean value or current best iterate value
        u0 = [-1.0, -1.0]  # Initial iterate value

        stosag_instance = stosag(u0, 
                                rosenbrocks,
                                lb, 
                                ub, 
                                M, 
                                Ct,
                                constants={'mode': 4,
                                            'max_iter': 10000, 
                                            'line_search_max_iter': 20,
                                            'line_search_max_attempts': 10,
                                            'line_search_alpha': 0.1})

        stosag_instance.run()  # Run the optimization process

        print("Number of evaluations:", stosag_instance.N_EVAL)
        print("Best iterate values:", stosag_instance.x_list[-1])  # Get the last best iterate value
        print("Function value at best iterate:", stosag_instance.j_list[-1])  

if __name__ == '__main__':
    unittest.main()  # Run unit tests if this script is executed directly