from stosag.utilities import value_shifted_array, \
                        calculate_cross_covariance, \
                        calculate_approximate_gradient_from_covariances, \
                        normalize_variable, \
                        denormalize_variable,\
                        log_normalize_variable, \
                        log_denormalize_variable
                        
from scipy.stats import multivariate_normal

from typing import Callable, List

import numpy as np


class stosag:
    """Stochastic Optimization using Stochastic Gradient Descent (SPE-173236-MS)"""

    def __init__(self, x0:List, functions:Callable, lb:List[float], ub:List[float], ineqs:List[Callable], Nens:int, Ct:np.ndarray, constants:dict):
        self.x0 = x0  # Initial iterate value
        self.functions = functions  # Function to evaluate
        self.Nens = Nens  # Number of ensemble realizations
        self.Ct = Ct  # Smoother covariance matrix
        
        self.mode = constants['mode'] if 'mode' in constants else 4 # Mode for gradient calculation
        self.max_iter = constants['max_iter'] if 'max_iter' in constants else 1000 # Maximum number of iterations
        self.line_search_max_iter = constants['line_search_max_iter'] if 'line_search_max_iter' in constants else 20 # Maximum iterations for line search
        self.line_search_max_attempts = constants['line_search_max_attempts'] if 'line_search_max_attempts' in constants else 5 # Maximum attempts for line search
        self.line_search_alpha = constants['line_search_alpha'] if 'line_search_alpha' in constants else 0.1 # Initial step size
        self.improvement_threshold = constants['improvement_threshold'] if 'improvement_threshold' in constants else 1e-6  # Improvement threshold
        
        self.lb = lb  # Lower bound for the optimization
        self.ub = ub  # Upper bound for the optimization
        
        self.ineqs = ineqs  # List of inequality constraint functions 
        
        self.robustness_measure = lambda x: np.mean(x, axis=0)  # Robustness measure function
        
        self.x_list = []  # List to store best iterate values
        self.j_list = []  # List to store function values
        
        self.N_EVAL = 0  # Number of evaluations
        pass
        
    def modify_function_to_use_transformed_variables(self, functions):
        """Modify the function to use transformed variables."""
        def modified_function(u:list):
            if isinstance(u[0], float) or isinstance(u[0], int):
            # if u.ndim == 1:
                x = log_denormalize_variable(u, self.lb, self.ub)  # Assuming denormalize is a function that transforms u to the original variable space
                self.N_EVAL += 1
            # else:
            elif isinstance(u[0], list):
                # x = np.zeros(u.shape)
                x = [] 
                for i in range(len(u)):
                    x.append(log_denormalize_variable(u[i], self.lb, self.ub).tolist())
                    
                self.N_EVAL += i  # Increment the number of evaluations by the number of ensemble realizations
                
            x = np.array(x)
            return functions(x)
        
        return modified_function

    def run(self):
        
        self.u0, self.mcf, self.mineqs, self.mfunctions = self.preparation()
        self._main_loop(self.u0, self.mfunctions)  # Start the main loop

    def preparation(self):
        """Preparation step before the main loop.
        1. Normalize the initial iterate value
        2. Modify the function to use transformed variables
        """
        
        # Normalize the initial iterate value
        u0 = log_normalize_variable(self.x0, self.lb, self.ub)
    
        # Modify the function to use transformed variables
        mcf = self.modify_function_to_use_transformed_variables(self.functions)
        
        mineqs = [self.modify_function_to_use_transformed_variables(ineq) for ineq in self.ineqs]
        
        def eval_mineqs(u, mineqs):
            cineqs = []
            for mineq in mineqs:
                cineq = mineq(u)
                cineq[cineq > 0.0] = 0.0
                cineqs.append(cineq)
                
            return cineqs
        
        # both cf and ineq should be simulated just one time
        mfunctions = lambda u : [mcf(u), eval_mineqs(u, mineqs)]
        
        return u0, mcf, mineqs, mfunctions
        
    def varphi(self, cineqs:List[float], rho:List, mu:np.ndarray):
        
        """Build the augmented Lagrangian function.
        Based on the paper "Minimizing the Risk in the robust life-cycle production optimization using stochastic simplex approximate gradient"
        Equation (16)
        """

        var = np.zeros(len(rho))
        for i, cineq in enumerate(cineqs):
            for j in range(len(rho)):
                
                if cineq[j] <= mu[i]*rho[i]:
                    var[j] += rho[i] * cineq[j] - 0.5 * cineq[j]**2 / mu[i]
                else:
                    var[j] += 0.5 * mu[i] * rho[i]**2
                
        return np.sum(var)
    
    def lagrangian_form(self, cfunctions, rho:np.ndarray, mu:np.ndarray):
        
        cf, cineqs = cfunctions
        v = self.varphi(cineqs, rho, mu)
        
        # print(cf, v)
        return cf - v
    
    def update_penalty_parameters(self, eta:float, rho:list, mu:list, eps_u:float, eps_f:float, cmax:float):
        """Update the penalty parameters.
        Based on the paper "Minimizing the Risk in the robust life-cycle production optimization using stochastic simplex approximate gradient"
        Equation (26), (29) and (30)
        
        eta: threshold of the constraint violation
        
        """
        
        a = 0.1
        b = 0.2
        tau = 0.25
        
        # tolerance at the final convergence
        eta_fin = 1.0
        eps_u_fin = 10E-3
        eps_f_fin = 10E-4
        
        sigma = np.sqrt(np.min([0, cmax])**2)
        
        rho_next = np.zeros(len(rho))
        mu_next = np.zeros(len(mu))
        
        # print(sigma, eta)
        # if sigma < eta: # step 2d
            
        #     for i in range(len(rho)):
        #         rho_next[i] = np.max([0, rho[i] - cmax/mu[i]])
                
        #     mu_next = mu    
        #     eta_next = np.max([eta*np.min([np.max(mu, initial=0)**b,0.5]), eta_fin])
        #     eps_u_next = np.max([eps_u*np.min([np.max(mu, initial=0)**b,0.5]), eps_u_fin])
        #     eps_f_next = np.max([eps_f*np.min([np.max(mu, initial=0)**b,0.5]), eps_f_fin])
            
        # else: # step 2e
            
        #     for i in range(len(rho)):
        #         mu_next[i] = tau*mu[i]

        #     rho_next = rho
        #     eta_next = np.max([eta*np.min([np.max(mu, initial=0)**a,0.5]), eta_fin])
        #     eps_u_next = np.max([eps_u*np.min([np.max(mu, initial=0)**a,0.5]), eps_u_fin])
        #     eps_f_next = np.max([eps_f*np.min([np.max(mu, initial=0)**a,0.5]), eps_f_fin])
        
        for i in range(len(rho)):
            rho_next[i] = rho[i] - mu[i]*cmax
            
        mu_next = mu
        eta_next = eta
        eps_u_next = eps_u
        eps_f_next = eps_f
        
        return eta_next, rho_next, mu_next, eps_u_next, eps_f_next
    
    def _main_loop(self, uInit, mfunctions:Callable):
        
        UInit = multivariate_normal(uInit, self.Ct).rvs(size=self.Nens).T  # Ensemble members around the initial iterate   
        
        UCurr = UInit  # Current ensemble members
        uBest = uInit  # Best iterate value
        
        _jBest, _cBest = mfunctions(uBest)
        jBest = self.robustness_measure(_jBest)
        cBest = [self.robustness_measure(_c) for _c in _cBest]
        
        rho = [0.0 for _ in cBest]
        mu = [1/(2000*np.abs(c)) if c < 0 else 0.1 for c in cBest]
        
        _lBest = self.lagrangian_form([_jBest, _cBest], rho=rho, mu=mu)
        lBest = self.robustness_measure(_lBest) 
                
        # jBest = self.robustness_measure(mfunctions(uBest))  # Best function value
        jInit = jBest  # Initial function value
        lInit = lBest
        
        
        # Initialize augmented lagrangian parameters
        eta = 1.0
        eps_u = 0.1
        eps_f = 0.1
        
        
        N_unsuccessful = 0  # Counter for unsuccessful line searches
        is_successful = True  # Flag to check if the bracktracking search is successful
        line_ii = 0 # Line search iteration index
        """Main loop for the optimization process."""
        for ii in range(self.max_iter):
            
            JCurr, CCurr = mfunctions(UCurr.T.tolist())
            LCurr = self.lagrangian_form([JCurr, CCurr], rho=rho, mu=mu)
            
            # stopping conditions:
            # 1. if the function value does not improve after N line search iterations
            # 2. if the improvement in function value is less than a threshold
            # 3. if the maximum number of iterations is reached
            
            if is_successful:
                print(f"Iteration {ii}: " 
                      f"xBest = {log_denormalize_variable(uBest, self.lb, self.ub)}, " 
                      f"jBest = {jBest}, " 
                      f"cBest = {np.min(cBest, initial=0)}, " 
                      f"lBest = {lBest}, " 
                      f"rho = {rho}, "
                      f"mu = {mu}, "
                      f"line_ii = {line_ii}, "  
                      f"N_EVAL = {self.N_EVAL}")
                self._write_results(log_denormalize_variable(uBest, self.lb, self.ub), lBest)
                N_unsuccessful = 0
                
                if self.robustness_measure(np.abs(LCurr - lBest)/np.abs(lInit)) < self.improvement_threshold:
                    print(f"Stopping criteria met: Improvement in function value is less than {self.improvement_threshold}.")
                    self.stopping_criteria = "Improvement threshold reached"
                    break
                
            else:
                N_unsuccessful += 1
                if N_unsuccessful >= self.line_search_max_attempts:
                    print(f"Stopping criteria met: No sufficient improvement in function value after {N_unsuccessful} unsuccessful line searches.")
                    self.stopping_criteria = "Maximum unsuccessful line searches attempts reached"
                    break
                
            dU = value_shifted_array(UCurr, uBest)  # Shift the ensemble members
            dL = LCurr - lBest  # Shift the function values
        
            # Calculate covariance matrices
            Cuu = calculate_cross_covariance(dU, dU)
            Cul = calculate_cross_covariance(dU, dL)
            
            # Calculate gradients
            g = calculate_approximate_gradient_from_covariances(Cuu, 
                                                                Cul, 
                                                                Ct=self.Ct, 
                                                                U=dU, 
                                                                j=dL, 
                                                                mode=self.mode)
            
            # Perform backtracking line search to find the optimal step size
            uNext, jNext, cNext, lNext, line_ii, is_successful = self._backtracking_line_search(uBest, 
                                                                            jBest,
                                                                            cBest,
                                                                            lBest, 
                                                                            mfunctions, 
                                                                            g, 
                                                                            self.line_search_alpha,
                                                                            rho,
                                                                            mu)
            
            # Update ensemble members and function values
            UCurr = multivariate_normal(uNext, self.Ct).rvs(size=UInit.shape[1]).T  # Ensemble members around the next iterate
            
            uBest = uNext  # Update best iterate value
            cBest = cNext
            jBest = jNext
            lBest = lNext  # Update best function value
            
            # update penalty parameters
            if not is_successful:
                pass
            else:
                eta, rho, mu, eps_u, eps_f = self.update_penalty_parameters(eta, 
                                                                        rho, 
                                                                        mu, 
                                                                        eps_u, 
                                                                        eps_f, 
                                                                        np.min(cNext, initial=0))
            
        if ii == self.max_iter - 1:
            print(f"Warning: Maximum number of iterations {self.max_iter} reached.")
            self.stopping_criteria = "Maximum iterations reached"
            
    
    def _backtracking_line_search(self, uInit, jInit, cInit, lInit, mfunctions, g, alpha, rho, mu):
        """Perform backtracking line search to find the optimal step size.
        uInit: Initial ensemble members
        jInit: Initial function values
        g: Gradient array
        alpha: Initial step size
        maxIter: Maximum number of iterations for backtracking line search
        """

        gamma = 0.001  # Step size reduction factor
        for ii in range(self.line_search_max_iter):
            uNext = uInit - alpha * g  # Calculate the next best point based on the gradient
            

            jNext, cNext = mfunctions(uNext)
            lNext = self.lagrangian_form([jNext, cNext], rho=rho, mu=mu)
            jNext = self.robustness_measure(jNext)
            cNext = [self.robustness_measure(c) for c in cNext]
            lNext = self.robustness_measure(lNext)
            # jNext = self.robustness_measure(mfunctions(uNext)) # Evaluate function values at the new point
            
            # apply armijo condition
            if lNext <= lInit - gamma * alpha * np.dot(g, g): 
            # if jNext < jInit:
                # print(f"Backtracking line search successful at iteration {ii} with alpha = {alpha}, jNext = {jNext}, jInit = {jInit}")
                is_successful = True
                break
            else:
                alpha *= 0.5
                if ii == self.line_search_max_iter - 1:
                    is_successful = False
                    return uInit, jInit, cInit, lInit, ii, is_successful
                
        return uNext, jNext, cNext, lNext, ii, is_successful
    
    def _write_results(self, x:np.ndarray, j:float):
        """Write results to a file or database."""
        
        self.x_list.append(x.tolist())
        self.j_list.append(j)
         
        pass