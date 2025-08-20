from utilities import value_shifted_array, calculate_cross_covariance, calculate_approximate_gradient_from_covariances
from scipy.stats import multivariate_normal

import numpy as np


class stosag:
    """Stochastic Optimization using Stochastic Gradient Descent (SPE-173236-MS)"""

    def __init__(self, u0, functions, Nens, Ct, mode=4):
        self.u0 = u0  # Initial iterate value
        self.functions = functions  # Function to evaluate
        self.Nens = Nens  # Number of ensemble realizations
        self.Ct = Ct  # Smoother covariance matrix
        self.mode = mode  # Mode for gradient calculation
        
        self.robustness_measure = lambda x: np.mean(x, axis=0)  # Robustness measure function
        
        self.u_list = []  # List to store best iterate values
        self.j_list = []  # List to store function values
        pass

    def run(self):
        
        self._main_loop(self.u0, self.functions)  # Start the main loop

    
    def _main_loop(self, uInit, functions:callable, alpha=0.01, maxIter=10000):
        
        UInit = multivariate_normal(uInit, self.Ct).rvs(size=self.Nens).T  # Ensemble members around the initial iterate   
        
        UCurr = UInit  # Current ensemble members
        JCurr = self.robustness_measure(functions(UCurr))  # Current function values
        
        uBest = uInit  # Best iterate value
        jBest = self.robustness_measure(functions(uBest))  # Best function value
        
        is_successful = True  # Flag to check if the bracktracking search is successful
        """Main loop for the optimization process."""
        for ii in range(maxIter):
            
            if is_successful:
                print(f"Iteration {ii}: uBest = {uBest}, jBest = {jBest}")
            
            dU = value_shifted_array(UCurr, uBest)  # Shift the ensemble members
            dJ = JCurr - jBest  # Shift the function values
            
            # Calculate covariance matrices
            Cuu = calculate_cross_covariance(dU, dU)
            Cuj = calculate_cross_covariance(dU, dJ)
            
            # Calculate gradients
            g = calculate_approximate_gradient_from_covariances(Cuu, 
                                                                Cuj, 
                                                                Ct=self.Ct, 
                                                                U=dU, 
                                                                j=dJ, 
                                                                mode=self.mode)
            
            # Perform backtracking line search to find the optimal step size
            uNext, jNext, _, is_successful = self._backtracking_line_search(uBest, 
                                                                            jBest, 
                                                                            functions, 
                                                                            g, 
                                                                            alpha)
            
            # Update ensemble members and function values
            UCurr = multivariate_normal(uNext, self.Ct).rvs(size=UInit.shape[1]).T  # Ensemble members around the next iterate
            
            uBest = uNext  # Update best iterate value
            jBest = jNext  # Update best function value
            
            # Store results
            self._write_results(uBest, jBest)
    
    def _backtracking_line_search(self, uInit, jInit, function, g, alpha, maxIter=10):
        """Perform backtracking line search to find the optimal step size.
        uInit: Initial ensemble members
        jInit: Initial function values
        g: Gradient array
        alpha: Initial step size
        maxIter: Maximum number of iterations for backtracking line search
        """
        for ii in range(maxIter):
            uNext = uInit - alpha * g  # Calculate the next best point based on the gradient
            
            jNext = self.robustness_measure(function(uNext)) # Evaluate function values at the new point

            if jNext <= jInit:
                # print(f"Backtracking line search successful at iteration {ii} with alpha = {alpha}, jNext = {jNext}, jInit = {jInit}")
                is_successful = True
                break
            else:
                alpha *= 0.5
                if ii == maxIter - 1:
                    is_successful = False
                    # print("Warning: Maximum iterations reached in backtracking line search.")
                    return uInit, jInit, alpha, is_successful
                
        return uNext, jNext, alpha, is_successful
    
    def _write_results(self, uNext, jNext):
        """Write results to a file or database."""
        
        self.u_list.append(uNext)
        self.j_list.append(jNext)
         
        pass