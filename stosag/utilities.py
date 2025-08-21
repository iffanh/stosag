import numpy as np
from typing import List

def generate_random_array(size:int) -> np.ndarray: 
    """Generates a random array of given size."""
    return np.random.rand(size)

def value_shifted_array(array:np.ndarray, shift_value:np.ndarray) -> np.ndarray:
    """Shifts the values in the array by a given value."""
    return array - shift_value[:, np.newaxis]  # Ensure the shift_value is broadcasted correctly


## Covariance functions to calculate approximate gradients
# based on paper SPE-173236-MS
def calculate_cross_covariance(array1:np.ndarray, array2:np.ndarray) -> float:
    """Calculates the covariance between two arrays.
    
    N = number of well parameters
    M = number of ensemble realizations
    array1: First input array. array1 size (N, M)
    array2: Second input array. array2 size (N, M) or (M,)
    """
    
    N = array1.shape[0]
    M = array1.shape[1]
    if array2.ndim == 1 and array2.shape[0] != M:
        raise ValueError("If array2 is a 1D array, it must have the same number of elements as the number of ensemble realizations (M).")
    
    return 1/(M - 1) * np.matmul(array1, array2.T)

def calculate_approximate_gradient_from_covariances(Cuu, Cuj, Ct=None, U=None, j=None, mode:int=0):
    """_summary_

    Args:
        mode (int, optional): different modes of calculating gradient based on SPE-173236-MS. Defaults to 0.
        mode 0: calculate gradient using Equation (8)
        mode 1: calculate gradient using Equation (11)
        mode 2: calculate gradient using Equation (12)
        mode 3: calculate gradient using Equation (14)
        mode 4: calculate gradient using Equation (15) (Recommended approach in the paper)
    Returns:
    
        np.ndarray: gradient array

    """
    
    if mode == 0:
        g = np.matmul(np.linalg.inv(Cuu), Cuj)
    elif mode == 1:
        g = Cuj[:,0]
    elif mode == 2:
        g = np.matmul(Cuu, Cuj)[:,0]
    elif mode == 4:
        # Recommended approach in the paper
        UU = np.matmul(U, U.T)
        Uj = np.matmul(U, j)
        g = np.matmul(np.matmul(Ct, np.linalg.pinv(UU)), Uj)
    else:
        raise ValueError("Invalid mode. Choose from 0, 1, 2, or 4.")
    
    return g


## Well related covariance functions
def create_spherical_covariance_function(Nw:int, std:List[float], Ns:List[int], Nt:List[int]) -> np.ndarray:
    """From the paper originally by Chen et al. (2015). "Ensemble-Based Optimization of the
    Water-Alternating-Gas-Injection Process" SPE Journal
    
    or by Ru et al. (2017). "An efficient adaptive algorithm 
    for robust control optimization using Stosag" Journal of Petroleum Science and Engineering, 159, 314-330
    
    Nw = number of wells
    std = standard deviations for each well
    Ns = number of correlated control steps for each well
    Nt = number of time steps for each well
    
    """
    
    for k in range(Nw):
        Cd = create_block_matrix(std[k], Ns[k], Nt[k])
        if k == 0:
            C = Cd
        else:
            # Create a block diagonal matrix with the current covariance matrix
            # and the previous covariance matrices
        
            C = np.block([[C, np.zeros((C.shape[0], Cd.shape[1]))],[np.zeros((Cd.shape[0], C.shape[1])), Cd]])
        
    return C

def create_block_matrix(std:float, Ns:int, Nt:int) -> np.ndarray:
    """Creates a block diagonal matrix with the given standard deviation and number of correlated steps."""
    
    if Nt < Ns:
        raise ValueError("Nt must be greater than or equal to Ns.")
    
    
    Cd = np.zeros((Nt,Nt))
    for i in range(Nt):
        for j in range(Nt):
            if abs(i - j) > Ns:
                Cd[i,j] = 0
            else:
                # Using a cubic decay function for the covariance
                Cd[i,j] = std**2 * (1 - 1.5*abs(i - j)/Ns + 0.5*(abs(i - j)/Ns)**3)
    
    return Cd

## variable transformation functions
@np.vectorize
def normalize_variable(x, lb, ub):
    """Normalizes a variable x between the lower bound lb and upper bound ub."""
    return (x - lb) / (ub - lb)

@np.vectorize
def denormalize_variable(x, lb, ub):
    """Denormalizes a variable x between the lower bound lb and upper bound ub."""
    return x * (ub - lb) + lb  # Ensure the denormalization is done correctly
    # return np.add(np.multiply(x,(ub - lb), axis=0), lb, axis=0)

@np.vectorize
def log_normalize_variable(x, lb, ub):
    """Log-normalizes a variable x between the lower bound lb and upper bound ub."""
    return np.log((x - lb) / (ub - x))

@np.vectorize
def log_denormalize_variable(x, lb, ub):
    """Log-denormalizes a variable x between the lower bound lb and upper bound ub."""
    return (ub * np.exp(x) + lb) / (1 + np.exp(x))