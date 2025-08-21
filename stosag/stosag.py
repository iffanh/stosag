from utilities import value_shifted_array, \
                        calculate_cross_covariance, \
                        calculate_approximate_gradient_from_covariances, \
                        normalize_variable, \
                        denormalize_variable,\
                        log_normalize_variable, \
                        log_denormalize_variable
                        
from scipy.stats import multivariate_normal

import numpy as np


class stosag:
    """Stochastic Optimization using Stochastic Gradient Descent (SPE-173236-MS)"""

    def __init__(self, x0, functions, lb, ub, Nens, Ct, mode=4):
        self.x0 = x0  # Initial iterate value
        self.functions = functions  # Function to evaluate
        self.Nens = Nens  # Number of ensemble realizations
        self.Ct = Ct  # Smoother covariance matrix
        self.mode = mode  # Mode for gradient calculation
        
        self.lb = lb  # Lower bound for the optimization
        self.ub = ub  # Upper bound for the optimization
        
        self.robustness_measure = lambda x: np.mean(x, axis=0)  # Robustness measure function
        
        self.u_list = []  # List to store best iterate values
        self.j_list = []  # List to store function values
        pass
        
    def modify_function_to_use_transformed_variables(self, functions):
        """Modify the function to use transformed variables."""
        def modified_function(u):
            if u.ndim == 1:
                x = log_denormalize_variable(u, self.lb, self.ub)  # Assuming denormalize is a function that transforms u to the original variable space
            else:
                x = np.zeros((u.shape[0], u.shape[1]))  # Initialize x with the correct shape
                for i in range(u.shape[1]):
                    x[:,i] = log_denormalize_variable(u[:,i], self.lb, self.ub)
            return functions(x)
        
        return modified_function

    def run(self):
        
        self.u0, self.mfunctions = self.preparation()
        self._main_loop(self.u0, self.mfunctions)  # Start the main loop

    def preparation(self):
        """Preparation step before the main loop.
        1. Normalize the initial iterate value
        2. Modify the function to use transformed variables
        """
        
        # Normalize the initial iterate value
        u0 = log_normalize_variable(self.x0, self.lb, self.ub)
        
        # Modify the function to use transformed variables
        mfunctions = self.modify_function_to_use_transformed_variables(self.functions)
        
        return u0, mfunctions
    
    def _main_loop(self, uInit, mfunctions:callable, alpha=0.01, maxIter=10000):
        
        UInit = multivariate_normal(uInit, self.Ct).rvs(size=self.Nens).T  # Ensemble members around the initial iterate   
        
        UCurr = UInit  # Current ensemble members
        JCurr = self.robustness_measure(mfunctions(UCurr))  # Current function values
        
        uBest = uInit  # Best iterate value
        jBest = self.robustness_measure(mfunctions(uBest))  # Best function value
        
        is_successful = True  # Flag to check if the bracktracking search is successful
        """Main loop for the optimization process."""
        for ii in range(maxIter):
            
            if is_successful:
                print(f"Iteration {ii}: uBest = {log_denormalize_variable(uBest, self.lb, self.ub)}, jBest = {jBest}")
                # Store results
                self._write_results(log_denormalize_variable(uBest, self.lb, self.ub), jBest)
            
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
                                                                            mfunctions, 
                                                                            g, 
                                                                            alpha)
            
            # Update ensemble members and function values
            UCurr = multivariate_normal(uNext, self.Ct).rvs(size=UInit.shape[1]).T  # Ensemble members around the next iterate
            
            uBest = uNext  # Update best iterate value
            jBest = jNext  # Update best function value
            
            
    
    def _backtracking_line_search(self, uInit, jInit, mfunctions, g, alpha, maxIter=10):
        """Perform backtracking line search to find the optimal step size.
        uInit: Initial ensemble members
        jInit: Initial function values
        g: Gradient array
        alpha: Initial step size
        maxIter: Maximum number of iterations for backtracking line search
        """
        for ii in range(maxIter):
            uNext = uInit - alpha * g  # Calculate the next best point based on the gradient
            
            jNext = self.robustness_measure(mfunctions(uNext)) # Evaluate function values at the new point

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