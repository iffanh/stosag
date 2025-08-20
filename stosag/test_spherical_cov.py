from utilities import create_spherical_covariance_function

Nw = 3
std = [0.1, 0.2, 0.3]  # Standard deviations for each well
Ns = [1, 2, 1]  # Number of correlated control steps for each well
Nt = [2, 4, 1]  # Number of time steps for each

C = create_spherical_covariance_function(Nw=Nw, std=std, Ns=Ns, Nt=Nt)