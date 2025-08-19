import numpy as np

def generate_random_array(size:int) -> np.ndarray: 
    """Generates a random array of given size."""
    return np.random.rand(size)

def value_shifted_array(array:np.ndarray, shift_value:np.ndarray) -> np.ndarray:
    """Shifts the values in the array by a given value."""
    return array - shift_value[:, np.newaxis]  # Ensure the shift_value is broadcasted correctly


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
    
    # if array2.shape[0] == 1:
    #     # If array2 is a 1D array, reshape it to 1 x M
    #     array2 = array2.reshape(1, M)
    # else:
    #     # check if array1 and array2 have the same number of rows
    #     if N != array2.shape[0]:
    #         raise ValueError("Both arrays must have the same number of rows (N).")
    
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
        raise ValueError("Invalid mode. Choose from 0, 1, 2, or 3.")
    
    return g


