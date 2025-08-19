from utilities import value_shifted_array, calculate_cross_covariance, calculate_approximate_gradient_from_covariances

import numpy as np


class stosag:
    """Stochastic Optimization using Stochastic Gradient Descent (SPE-173236-MS)"""

    def __init__(self, U, u0, functions, Ct, mode=4):
        self.U = U  # Ensemble members
        self.u0 = u0  # Initial iterate value
        self.functions = functions  # Function to evaluate
        self.Ct = Ct  # Smoother covariance matrix
        self.mode = mode  # Mode for gradient calculation
        
        self.robustness_measure = lambda x: np.mean(x, axis=0)  # Robustness measure function
        
        self.u_list = []  # List to store best iterate values
        self.j_list = []  # List to store function values
        pass

    def run(self):
        
        self._main_loop(self.u0, self.U, self.functions)  # Start the main loop

    
    def _main_loop(self, uInit, UInit, function:callable, alpha=0.01, maxIter=10):
        
        UCurr = UInit  # Current ensemble members
        JCurr = self.robustness_measure(function(UCurr))  # Current function values
        
        uBest = uInit  # Best iterate value
        jBest = self.robustness_measure(function(uBest))  # Best function value
        
        """Main loop for the optimization process."""
        for ii in range(maxIter):
            print(f"Iteration {ii}: uBest = {uBest}, jBest = {jBest}, alpha = {alpha}")
            
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
            uNext, jNext, _ = self._backtracking_line_search(uBest, jBest, function, g, alpha)
            
            # Update ensemble members and function values
            UCurr = uNext[:,np.newaxis] + 0.1 * np.random.rand(*UInit.shape)  # Add some noise
            
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
            uNext = uInit - alpha * g  # Update ensemble members based on the gradient
            
            UNext = uNext + 0.1 * np.random.rand(*uInit.shape)  # Add some noise
            
            jNext = self.robustness_measure(function(UNext))
            
            # print(f"Iteration {ii}: UNext = {UNext}, jNext = {jNext}, jInit = {jInit}, alpha = {alpha}, g = {g}"
            #       )
            # if np.mean(jNext) <= np.mean(jInit) + 0.1 * alpha * np.sum(g**2):
            if np.mean(jNext) <= np.mean(jInit):
                print(f"Backtracking line search successful at iteration {ii} with alpha = {alpha}, jNext = {jNext}, jInit = {jInit}")
                break
            else:
                alpha *= 0.5
                if ii == maxIter - 1:
                    print("Warning: Maximum iterations reached in backtracking line search.")
                    return uInit, jInit, alpha
                
        return uNext, jNext, alpha
    
    def _write_results(self, uNext, jNext):
        """Write results to a file or database."""
        
        self.u_list.append(uNext)
        self.j_list.append(jNext)
         
        pass