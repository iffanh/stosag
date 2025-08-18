from utilities import value_shifted_array, calculate_cross_covariance, calculate_approximate_gradient_from_covariances

import numpy as np


class stosag:
    """Stochastic Optimization using Stochastic Gradient Descent (SPE-173236-MS)"""

    def __init__(self, u, u0, function, Ct, mode=4):
        self.u = u  # Ensemble members
        self.u0 = u0  # Initial iterate value
        self.function = function  # Function to evaluate
        self.Ct = Ct  # Smoother covariance matrix
        self.mode = mode  # Mode for gradient calculation
        pass

    def run(self):
        
        self.j0 = self.function(self.u0)  # calculate j0 based on the Rosenbrock function
        self.j = self.function(self.u)  # calculate j based on the Rosenbrock function
            
        self.U = value_shifted_array(self.u, self.u0)  # Equation (19) as recommended in the paper
        self.J = value_shifted_array(self.j, self.j0)  # Equation (20) as recommended in the paper
        
        # Precalculate covariance matrices
        self.Cuu = calculate_cross_covariance(self.U, self.U)
        self.Cuj = calculate_cross_covariance(self.U, self.J)

        # Calculate gradients using different modes
        self.g = calculate_approximate_gradient_from_covariances(self.Cuu, 
                                                                 self.Cuj, 
                                                                 Ct=self.Ct, 
                                                                 U=self.U, 
                                                                 j=self.j, 
                                                                 mode=self.mode) # We use mode 4 (Equation (15)) as recommended in the paper

        # Update the values in U based on the calculated gradients using backtracking line search
        UNext = np.zeros(self.U.shape)
        uNext = np.zeros(self.u.shape)
        
        alpha = 0.01  # step size
        
        # backtracking inner loop
        maxIter = 10
        for ii in range(maxIter):
            
            for i in range(self.u0.shape[0]):
                UNext[:, i] = self.U[:, i] - alpha * self.g  # Update U using the gradient from mode 4 
            
            # Update u based on the shifted values
            uNext = value_shifted_array(UNext, -self.u0)
            
            # Calculate the new function values
            jNext = self.function(uNext)
            
            # Check if the new function values are better than the old ones using Armijo's rule
            # if np.all(jNext <= self.j + 0.1 * alpha * np.sum(self.g**2)):
            if np.mean(jNext) <= np.mean(self.j) + 0.1 * alpha * np.sum(self.g**2): # Use mean for robustness measure, might change later
                break
            else:
                alpha *= 0.5
        
        print(ii, uNext, jNext)
        
        return uNext  # Return the updated ensemble members