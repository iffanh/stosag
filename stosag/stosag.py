from stosag.utilities import value_shifted_array, \
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

    def __init__(self, x0, functions, lb, ub, Nens, Ct, constants):
        self.x0 = x0  # Initial iterate value
        self.functions = functions  # Function to evaluate
        self.Nens = Nens  # Number of ensemble realizations
        self.Ct = Ct  # Smoother covariance matrix
        
        self.mode = constants['mode']  # Mode for gradient calculation
        self.max_iter = constants['max_iter']  # Maximum number of iterations
        self.line_search_max_iter = constants['line_search_max_iter']  # Maximum iterations for line search
        self.line_search_alpha = constants['line_search_alpha']  # Initial step size
        
        self.lb = lb  # Lower bound for the optimization
        self.ub = ub  # Upper bound for the optimization
        
        self.robustness_measure = lambda x: np.mean(x, axis=0)  # Robustness measure function
        
        self.x_list = []  # List to store best iterate values
        self.j_list = []  # List to store function values
        
        self.N_EVAL = 0  # Number of evaluations
        pass
        
    def modify_function_to_use_transformed_variables(self, functions):
        """Modify the function to use transformed variables."""
        def modified_function(u):
            if u.ndim == 1:
                x = log_denormalize_variable(u, self.lb, self.ub)  # Assuming denormalize is a function that transforms u to the original variable space
                self.N_EVAL += 1
            else:
                x = np.zeros(u.shape)  # Initialize x with the correct shape
                for i in range(u.shape[1]):
                    x[:,i] = log_denormalize_variable(u[:,i], self.lb, self.ub)
                    
                self.N_EVAL += u.shape[1]  # Increment the number of evaluations by the number of ensemble realizations
                
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
    
    def _main_loop(self, uInit, mfunctions:callable):
        
        UInit = multivariate_normal(uInit, self.Ct).rvs(size=self.Nens).T  # Ensemble members around the initial iterate   
        
        UCurr = UInit  # Current ensemble members
        # JCurr = self.robustness_measure(mfunctions(UCurr))  # Current function values
        
        uBest = uInit  # Best iterate value
        jBest = self.robustness_measure(mfunctions(uBest))  # Best function value
        
        is_successful = True  # Flag to check if the bracktracking search is successful
        line_ii = 0 # Line search iteration index
        """Main loop for the optimization process."""
        for ii in range(self.max_iter):
            
            JCurr = self.robustness_measure(mfunctions(UCurr))  # Current function values
            
            if is_successful:
                print(f"Iteration {ii}: xBest = {log_denormalize_variable(uBest, self.lb, self.ub)}, jBest = {jBest}, line_ii = {line_ii}, N_EVAL = {self.N_EVAL}")
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
            uNext, jNext, line_ii, is_successful = self._backtracking_line_search(uBest, 
                                                                            jBest, 
                                                                            mfunctions, 
                                                                            g, 
                                                                            self.line_search_alpha)
            
            # Update ensemble members and function values
            UCurr = multivariate_normal(uNext, self.Ct).rvs(size=UInit.shape[1]).T  # Ensemble members around the next iterate
            
            uBest = uNext  # Update best iterate value
            jBest = jNext  # Update best function value
            
            
    
    def _backtracking_line_search(self, uInit, jInit, mfunctions, g, alpha):
        """Perform backtracking line search to find the optimal step size.
        uInit: Initial ensemble members
        jInit: Initial function values
        g: Gradient array
        alpha: Initial step size
        maxIter: Maximum number of iterations for backtracking line search
        """
        for ii in range(self.line_search_max_iter):
            uNext = uInit - alpha * g  # Calculate the next best point based on the gradient
            
            jNext = self.robustness_measure(mfunctions(uNext)) # Evaluate function values at the new point

            if jNext <= jInit:
                # print(f"Backtracking line search successful at iteration {ii} with alpha = {alpha}, jNext = {jNext}, jInit = {jInit}")
                is_successful = True
                break
            else:
                alpha *= 0.5
                if ii == self.line_search_max_iter - 1:
                    is_successful = False
                    # print("Warning: Maximum iterations reached in backtracking line search.")
                    return uInit, jInit, alpha, is_successful
                
        return uNext, jNext, ii, is_successful
    
    def _write_results(self, x, j):
        """Write results to a file or database."""
        
        self.x_list.append(x)
        self.j_list.append(j)
         
        pass