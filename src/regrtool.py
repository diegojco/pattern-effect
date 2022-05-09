""" Module containing functions to perform regression and other statistical tests
"""
from __future__ import division
import numpy as np
import scipy.odr as odr
import scipy.interpolate as interpol
import randtool as rnt
import disttool as dst

""" Empirical Mode Decomposition of a one-dimensional time series
"""

def EMD(x, S_tol=10, it_lim=1000):
    """ Decomposes a 1D time series in IMFs: an Empirical Mode Decomposition
    """
    t = np.array(range(x.size))
    modes = []
    
    data = x[0:]
    monotonous = np.all(data == np.sort(data)) or np.all(data[::-1] == np.sort(data))
    while not monotonous:
        old_data = data[0:]
        mode = IMF_by_index(old_data, S_tol=S_tol, it_lim=it_lim)
        modes.append(mode)
        data = old_data - mode(t)
        monotonous = np.all(data == np.sort(data)) or np.all(data[::-1] == np.sort(data))
    return modes

def IMF_by_index(x, S_tol=10, it_lim=1000):
    """ Process a 1D time series to get an IMF by index
    """
    t = np.array(range(x.size))
    
    n_it = 1
    S = 0
    
    it = preIMF_by_index(x)
    it_data = it(t)
    n_max = identify_local_maxima_by_index(it_data).size - 2
    n_min = identify_local_minima_by_index(it_data).size - 2
    while (S < S_tol) or (n_it < it_lim):
        n_it += 1
        n_max_old = n_max
        n_min_old = n_min
        it = preIMF_by_index(it_data)
        it_data = it(t)
        n_max = identify_local_maxima_by_index(it_data).size - 2
        n_min = identify_local_minima_by_index(it_data).size - 2
        test_1 = abs(n_max - n_max_old) <= 1
        test_2 = abs(n_min - n_min_old) <= 1
        if test_1 and test_2:
            S += 1
        else:
            S = 0
    return it            

def preIMF_by_index(x):
    """ Get a candidate of Intrinsic Mode Function from data by index
    """
    t = np.array(range(x.size))
    avg = average_by_index(x)(t)
    candidate = x - avg
    return interpol.CubicSpline(t, candidate, bc_type="natural", extrapolate=True)

def average_by_index(x):
    """ Get a cubic spline interpolator for the average between envelopes by index
    """
    t = np.array(range(x.size))
    min_envelope = minima_envelope_by_index(x)
    max_envelope = maxima_envelope_by_index(x)
    avg = (max_envelope(t) + min_envelope(t)) / 2.0
    return interpol.CubicSpline(t, avg, bc_type="natural", extrapolate=True)

def minima_envelope_by_index(x):
    """ Get a cubic spline interpolator for the minima envelope by index
    """
    minima = identify_local_minima_by_index(x)
    return interpol.CubicSpline(minima, x[minima], bc_type="natural", extrapolate=True)

def maxima_envelope_by_index(x):
    """ Get a cubic spline interpolator for the maxima envelope by index
    """
    maxima = identify_local_maxima_by_index(x)
    return interpol.CubicSpline(maxima, x[maxima], bc_type="natural", extrapolate=True)

def identify_local_minima_by_index(x):
    """ Get the local minima of the signal by index
    """
    temp = np.sign(np.diff(x)[:-1]) + np.sign(np.diff(x[::-1])[:-1][::-1])
    temp = np.nonzero(temp < 0)[0] + 1
    minima = []
    if x[0] < x[1]:
        minima.append(0)
    [minima.append(i) for i in temp]
    if x[-1] < x[-2]:
        minima.append(x.size - 1)
    return np.array(minima)

def identify_local_maxima_by_index(x):
    """ Get the local maxima of the signal by index
    """
    temp = np.sign(np.diff(x)[:-1]) + np.sign(np.diff(x[::-1])[:-1][::-1])
    temp = np.nonzero(temp > 0)[0] + 1
    maxima = []
    if x[0] > x[1]:
        maxima.append(0)
    [maxima.append(i) for i in temp]
    if x[-1] > x[-2]:
        maxima.append(x.size - 1)
    return np.array(maxima)

""" Pearson's correlation coefficient for two variables
"""

def dist_corrcoef(x, y, size=1000000, H0=True):
    """ Pearson's correlation coefficient for two variables
    """
    m = x.shape[0]
    
    results = []
    pcc = corrcoef(x, y)
    results.append(pcc)
    
    if H0:
        results_H0 = []
        results_H0.append(pcc)
    
    for r in range(3, m):
        sets = rnt.random_combinations(
            range(m), r, rnt.number_nCr_geq2(m, r, total=size)
        )
        
        new_x = None
        new_y = None
        
        for indexes in sets:
            
            new_x = np.array([x[index] for index in indexes])
            new_y = np.array([y[index] for index in indexes])
            
            pccp = corrcoef(new_x, new_y)
            results.append(pccp)
            
            if H0:
                
                indexes_shuffled = rnt.rn.sample(indexes, len(indexes))
                new_y = np.array([y[index] for index in indexes_shuffled])
                
                pccp = corrcoef(new_x, new_y)
                results_H0.append(pccp)
                
    dist_param = dst.random_variates(np.array(results))
    
    if H0:
        
        dist_param_H0 = dst.random_variates(np.array(results_H0))
        
        return pcc, dist_param, dist_param_H0
    
    else:
        
        return pcc, dist_param

def corrcoef(x, y):
    """ Pearson's correlation coefficient for two variables
    """
    return np.corrcoef(x, y)[0, 1]

""" Convenience functions for specific cases using orthogonal distance regression (ODR)
"""

""" 1-dimensional linear functions with 2-dimensional parameter space
"""

def dist_ODR_lin2d(x, y, size=1000000, multi=False, H0=False):
    """ Uses ODR to get the parameter distribution of the 1-dimensional linear model
        with 2-dimensional parameter space
    """
    beta0 = OLS_lin2d(x, y).beta
    
    if H0:
        b0, dist_param, dist_param_H0 = ODR_dist(
            flin2d, x, y, beta0, dfdbeta=dflin2ddb, dfdx=dflin2ddx,
            job="00020",
            size=size, multi=multi, H0=H0
        )
        
        return b0, dist_param, dist_param_H0
    
    else:
        b0, dist_param = ODR_dist(
            flin2d, x, y, beta0, dfdbeta=dflin2ddb, dfdx=dflin2ddx,
            job="00020",
            size=size, multi=multi, H0=H0
        )
        
        return b0, dist_param

def ODR_lin2d(x, y, iprint=None):
    """ Uses ODR to fit a 1-dimensional linear model with 2-dimensional parameter space
    """
    beta0 = OLS_lin2d(x, y).beta
    solution = ODR(
        flin2d, x, y, beta0, dfdbeta=dflin2ddb, dfdx=dflin2ddx,
        job="00020", iprint=iprint
    )
    
    return solution

""" 1-dimensional linear functions with a 1-dimensional parameter space (only slope and
    with x-intercept)
"""

def dist_ODR_lin1d_xi(x, y, x_intercept=0, size=1000000, multi=False, H0=False):
    """ Uses ODR to get the parameter distribution of the 1-dimensional linear model
        with 1-dimensional parameter space (only slope and with x-intercept)
    """
    beta0 = OLS_lin1d_xi(x, y, x_intercept=x_intercept).beta
    if H0:
        b0, dist_param, dist_param_H0 = ODR_dist(
            flin1d_xi(x_intercept=x_intercept),
            x, y, beta0,
            dfdbeta=dflin1ddb_xi(x_intercept=x_intercept),
            dfdx=dflin1ddx_xi(x_intercept=x_intercept),
            job="00020",
            size=size, multi=multi, H0=H0
        )
        
        return b0, dist_param, dist_param_H0
    
    else:
        b0, dist_param = ODR_dist(
            flin1d_xi(x_intercept=x_intercept),
            x, y, beta0,
            dfdbeta=dflin1ddb_xi(x_intercept=x_intercept),
            dfdx=dflin1ddx_xi(x_intercept=x_intercept),
            job="00020",
            size=size, multi=multi, H0=H0
        )
        
        return b0, dist_param

def ODR_lin1d_xi(x, y, x_intercept=0, iprint=None):
    """ Uses ODR to fit a 1-dimensional linear model with 1-dimensional parameter space
        (only slope and with x-intercept)
    """
    beta0 = OLS_lin1d_xi(x, y, x_intercept=x_intercept).beta
    solution = ODR(
        flin1d_xi(x_intercept=x_intercept),
        x, y, beta0,
        dfdbeta=dflin1ddb_xi(x_intercept=x_intercept),
        dfdx=dflin1ddx_xi(x_intercept=x_intercept),
        job="00020", iprint=iprint
    )
    
    return solution

""" 1-dimensional linear functions with a 1-dimensional parameter space (only slope and
    with y-intercept)
"""

def dist_ODR_lin1d_yi(x, y, y_intercept=0, size=1000000, multi=False, H0=False):
    """ Uses ODR to get the parameter distribution of the 1-dimensional linear model
        with 1-dimensional parameter space (only slope and with y-intercept)
    """
    beta0 = OLS_lin1d_yi(x, y, y_intercept=y_intercept).beta
    if H0:
        b0, dist_param, dist_param_H0 = ODR_dist(
            flin1d_yi(y_intercept=y_intercept),
            x, y, beta0,
            dfdbeta=dflin1ddb_yi(y_intercept=y_intercept),
            dfdx=dflin1ddx_yi(y_intercept=y_intercept),
            job="00020",
            size=size, multi=multi, H0=H0
        )
        
        return b0, dist_param, dist_param_H0
    
    else:
        b0, dist_param = ODR_dist(
            flin1d_yi(y_intercept=y_intercept),
            x, y, beta0,
            dfdbeta=dflin1ddb_yi(y_intercept=y_intercept),
            dfdx=dflin1ddx_yi(y_intercept=y_intercept),
            job="00020",
            size=size, multi=multi, H0=H0
        )
        
        return b0, dist_param

def ODR_lin1d_yi(x, y, y_intercept=0, iprint=None):
    """ Uses ODR to fit a 1-dimensional linear model with 1-dimensional parameter space
        (only slope and with y-intercept)
    """
    beta0 = OLS_lin1d_yi(x, y, y_intercept=y_intercept).beta
    solution = ODR(
        flin1d_yi(y_intercept=y_intercept),
        x, y, beta0,
        dfdbeta=dflin1ddb_yi(y_intercept=y_intercept),
        dfdx=dflin1ddx_yi(y_intercept=y_intercept),
        job="00020", iprint=iprint
    )
    
    return solution

""" 2-dimensional linear functions with 3-dimensional parameter space
"""

def dist_ODR_lin3d(x, y, size=1000000, multi=False, H0=False):
    """ Uses ODR to get the parameter distribution of the 2-dimensional linear model
        with 3-dimensional parameter space
    """
    beta0 = OLS_lin3d(x, y).beta
    
    if H0:
        b0, dist_param, dist_param_H0 = ODR_dist(
            flin3d, x, y, beta0, dfdbeta=dflin3ddb, dfdx=dflin3ddx,
            job="00020",
            size=size, multi=multi, H0=H0
        )
        
        return b0, dist_param, dist_param_H0
    
    else:
        b0, dist_param = ODR_dist(
            flin3d, x, y, beta0, dfdbeta=dflin3ddb, dfdx=dflin3ddx,
            job="00020",
            size=size, multi=multi, H0=H0
        )
        
        return b0, dist_param

def ODR_lin3d(x, y, iprint=None):
    """ Uses ODR to fit a 2-dimensional linear model with 3-dimensional parameter space
    """
    beta0 = OLS_lin3d(x, y).beta
    solution = ODR(
        flin3d, x, y, beta0, dfdbeta=dflin3ddb, dfdx=dflin3ddx,
        job="00020", iprint=iprint
    )
    
    return solution

""" Equilibrium climate sensitivity (ECS) model from Jimenez-de-la-Cuesta and Mauritsen
    (2019)
"""

def dist_ODR_ECS(x, y, size=1000000, multi=False, H0=False):
    """ Uses ODR to get the parameter distribution of the ECS model from
        Jimenez-de-la-Cuesta and Mauritsen (2019)
    """
    beta0 = OLS_ECS(x, y).beta
    
    if H0:
        b0, dist_param, dist_param_H0 = ODR_dist(
            fECS, x, y, beta0, dfdbeta=dfECSdb, dfdx=dfECSdx,
            job="00020",
            size=size, multi=multi, H0=H0
        )
        
        return b0, dist_param, dist_param_H0
    
    else:
        b0, dist_param = ODR_dist(
            fECS, x, y, beta0, dfdbeta=dfECSdb, dfdx=dfECSdx,
            job="00020",
            size=size, multi=multi, H0=H0
        )
        
        return b0, dist_param

def ODR_ECS(x, y, iprint=None):
    """ Uses ODR to fit the ECS model from Jimenez-de-la-Cuesta and Mauritsen (2019)
    """
    beta0 = OLS_ECS(x, y).beta
    solution = ODR(
        fECS, x, y, beta0, dfdbeta=dfECSdb, dfdx=dfECSdx,
        job="00020", iprint=iprint
    )
    
    return solution

""" Convenience functions for specific cases using ordinary least squares (OLS)
"""

""" 1-dimensional linear functions with 2-dimensional parameter space
"""

def dist_OLS_lin2d(x, y, size=1000000, multi=False, H0=False):
    """ Uses OLS to get the parameter distribution of the 1-dimensional linear model
        with 2-dimensional parameter space
    """
    if H0:
        b0, dist_param, dist_param_H0 = ODR_dist(
            flin2d, x, y, np.array([0, 0]), dfdbeta=dflin2ddb, dfdx=dflin2ddx,
            job="00022",
            size=size, multi=multi, H0=H0
        )
        
        return b0, dist_param, dist_param_H0
    
    else:
        b0, dist_param = ODR_dist(
            flin2d, x, y, np.array([0, 0]), dfdbeta=dflin2ddb, dfdx=dflin2ddx,
            job="00022",
            size=size, multi=multi, H0=H0
        )
        
        return b0, dist_param

def OLS_lin2d(x, y, iprint=None):
    """ Uses OLS to fit a 1-dimensional linear model with 2-dimensional parameter space
    """
    solution = ODR(
        flin2d, x, y, np.array([0, 0]), dfdbeta=dflin2ddb, dfdx=dflin2ddx,
        job="00022", iprint=iprint
    )
    
    return solution

""" 1-dimensional linear functions with a 1-dimensional parameter space (only slope and
    with x-intercept)
"""

def dist_OLS_lin1d_xi(x, y, x_intercept=0, size=1000000, multi=False, H0=False):
    """ Uses OLS to get the parameter distribution of the 1-dimensional linear model
        with 1-dimensional parameter space (only slope and with x-intercept)
    """
    if H0:
        b0, dist_param, dist_param_H0 = ODR_dist(
            flin1d_xi(x_intercept=x_intercept),
            x, y, np.array([0]),
            dfdbeta=dflin1ddb_xi(x_intercept=x_intercept),
            dfdx=dflin1ddx_xi(x_intercept=x_intercept),
            job="00022",
            size=size, multi=multi, H0=H0
        )
        
        return b0, dist_param, dist_param_H0
    
    else:
        b0, dist_param = ODR_dist(
            flin1d_xi(x_intercept=x_intercept),
            x, y, np.array([0]),
            dfdbeta=dflin1ddb_xi(x_intercept=x_intercept),
            dfdx=dflin1ddx_xi(x_intercept=x_intercept),
            job="00022",
            size=size, multi=multi, H0=H0
        )
        
        return b0, dist_param

def OLS_lin1d_xi(x, y, x_intercept=0, iprint=None):
    """ Uses OLS to fit a 1-dimensional linear model with 1-dimensional parameter space
        (only slope and with x-intercept)
    """
    solution = ODR(
        flin1d_xi(x_intercept=x_intercept),
        x, y, np.array([0]),
        dfdbeta=dflin1ddb_xi(x_intercept=x_intercept),
        dfdx=dflin1ddx_xi(x_intercept=x_intercept),
        job="00022", iprint=iprint
    )
    
    return solution

""" 1-dimensional linear functions with a 1-dimensional parameter space (only slope and
    with y-intercept)
"""

def dist_OLS_lin1d_yi(x, y, y_intercept=0, size=1000000, multi=False, H0=False):
    """ Uses OLS to get the parameter distribution of the 1-dimensional linear model
        with 1-dimensional parameter space (only slope and with y-intercept)
    """
    if H0:
        b0, dist_param, dist_param_H0 = ODR_dist(
            flin1d_yi(y_intercept=y_intercept),
            x, y, np.array([0]),
            dfdbeta=dflin1ddb_yi(y_intercept=y_intercept),
            dfdx=dflin1ddx_yi(y_intercept=y_intercept),
            job="00022",
            size=size, multi=multi, H0=H0
        )
        
        return b0, dist_param, dist_param_H0
    
    else:
        b0, dist_param = ODR_dist(
            flin1d_yi(y_intercept=y_intercept),
            x, y, np.array([0]),
            dfdbeta=dflin1ddb_yi(y_intercept=y_intercept),
            dfdx=dflin1ddx_yi(y_intercept=y_intercept),
            job="00022",
            size=size, multi=multi, H0=H0
        )
        
        return b0, dist_param

def OLS_lin1d_yi(x, y, y_intercept=0, iprint=None):
    """ Uses OLS to fit a 1-dimensional linear model with 1-dimensional parameter space
        (only slope and with y-intercept)
    """
    solution = ODR(
        flin1d_yi(y_intercept=y_intercept),
        x, y, np.array([0]),
        dfdbeta=dflin1ddb_yi(y_intercept=y_intercept),
        dfdx=dflin1ddx_yi(y_intercept=y_intercept),
        job="00022", iprint=iprint
    )
    
    return solution

""" 2-dimensional linear functions with 3-dimensional parameter space
"""

def dist_OLS_lin3d(x, y, size=1000000, multi=False, H0=False):
    """ Uses OLS to get the parameter distribution of the 2-dimensional linear model
        with 3-dimensional parameter space
    """
    if H0:
        b0, dist_param, dist_param_H0 = ODR_dist(
            flin3d, x, y, np.array([0, 0, 0]), dfdbeta=dflin3ddb, dfdx=dflin3ddx,
            job="00022",
            size=size, multi=multi, H0=H0
        )
        
        return b0, dist_param, dist_param_H0
    
    else:
        b0, dist_param = ODR_dist(
            flin3d, x, y, np.array([0, 0, 0]), dfdbeta=dflin3ddb, dfdx=dflin3ddx,
            job="00022",
            size=size, multi=multi, H0=H0
        )
        
        return b0, dist_param

def OLS_lin3d(x, y, iprint=None):
    """ Uses OLS to fit a 2-dimensional linear model with 3-dimensional parameter space
    """
    solution = ODR(
        flin3d, x, y, np.array([0, 0, 0]), dfdbeta=dflin3ddb, dfdx=dflin3ddx,
        job="00022", iprint=iprint
    )
    
    return solution

""" Equilibrium climate sensitivity (ECS) model from Jimenez-de-la-Cuesta and Mauritsen
    (2019)
"""

def dist_OLS_ECS(x, y, size=1000000, multi=False, H0=False):
    """ Uses OLS to get the parameter distribution of the ECS model from
        Jimenez-de-la-Cuesta and Mauritsen (2019)
    """
    if H0:
        b0, dist_param, dist_param_H0 = ODR_dist(
            fECS, x, y, np.array([0, 0]), dfdbeta=dfECSdb, dfdx=dfECSdx,
            job="00022",
            size=size, multi=multi, H0=H0
        )
        
        return b0, dist_param, dist_param_H0
    
    else:
        b0, dist_param = ODR_dist(
            fECS, x, y, np.array([0, 0]), dfdbeta=dfECSdb, dfdx=dfECSdx,
            job="00022",
            size=size, multi=multi, H0=H0
        )
        
        return b0, dist_param

def OLS_ECS(x, y, iprint=None):
    """ Uses OLS to fit the ECS model from Jimenez-de-la-Cuesta and Mauritsen (2019)
    """
    solution = ODR(
        fECS, x, y, np.array([0, 0]), dfdbeta=dfECSdb, dfdx=dfECSdx,
        job="00022", iprint=iprint
    )
    
    return solution

""" ODR basic functions
"""

def ODR_dist(f, x, y, beta0, dfdbeta=None, dfdx=None, job=None, iprint=None,
             size=1000000, multi=False, H0=False):
    """ Obtains distributions for the parameters of the regression. Uses resampling
        without replacement to make a bootstrap-like process and find the distribution
        of the regression parameters
        
        Parameters:
            
            f (function): User-defined function that can take ndarrays as input. This
                is the function representing the general model
            
            x (ndarray): Values of the independent variable at m points. Shape (m, )
                for a 1-dimensional variable. Shape (q, m) for a q-dimensional variable
            
            y (ndarray): Values of the dependent variable at m points. Shape (m, ) for
                a 1-dimensional variable. Shape (q', m) for a q'-dimensional variable
            
            beta0 (ndarray): Initial guess of the model's parameters. Shape (p, ) for
                p-dimensional parameter space
            
            dfdbeta (function,optional): User-defined function implementing the
                Jacobian matrix of f wrt parameters. The function should return an
                array of shape (q',p,m) containing the values of the Jacobian matrix at
                the m points
            
            dfdx (function,optional): User-defined function implementing the Jacobian
                matrix of f wrt the independent variable. The function should return an
                array of shape (q',q,m) containing the values of the Jacobian matrix at
                the m points
            
            job (str,optional): Determine job settings. See ODRPACK manual page 31
            
            iprint (str,optional): Determine report settings. See ODRPACK manual page
                33-34
            
            size (int,optional): Number of drawn samples for the bootstrap-like process
            
            multi (bool,optional): `True` to get the joint distribution of the
                parameters. Default: `False`
            
            H0 (bool,optional): `True` to get distributions for the null hypothesis of
                non-significant parameters. Default: `False`
            
        Returns:
            
            b0 (ndarray): Full-sample estimate in a (p,) array.
            
            dist_param (tuple): Functions to draw from the individual parameter
                distributions. If `multi=True` the joint distribution is obtained
            
            dist_param_H0 (tuple): Functions to draw from the individual parameter
                distributions of the null hypothesis. If `multi=True` the joint
                distribution is obtained
    """
    if len(x.shape) == 1:
        q = 1
        m = x.shape[0]
    else:
        q = x.shape[0]
        m = x.shape[1]
    
    if len(y.shape) == 1:
        qp = 1
    else:
        qp = y.shape[0]
    p = beta0.shape[0]
    
    results = []
    
    b0 = ODR(f, x, y, beta0, dfdbeta=dfdbeta, dfdx=dfdx, job=job, iprint=iprint)
    b0 = b0.beta
    results.append(b0)
    if H0:
        results_H0 = []
        results_H0.append(b0)
    
    for r in range(p + 1, m):
        sets = rnt.random_combinations(
            range(m), r, rnt.number_nCr_geq2(m, r, total=size)
        )
        
        new_x = None
        new_y = None
        for indexes in sets:
            
            if q == 1:
                new_x = np.array([x[index] for index in indexes])
            else:
                new_x = np.array([x[:, index] for index in indexes]).transpose()
            
            if qp == 1:
                new_y = np.array([y[index] for index in indexes])
            else:
                new_y = np.array([y[:, index] for index in indexes]).transpose()
                
            solution = ODR(
                f, new_x, new_y, b0, dfdbeta=dfdbeta, dfdx=dfdx, job=job, iprint=iprint
            )
            results.append(solution.beta)
            
            if H0:
                
                idxs_shuf = rnt.rn.sample(indexes, len(indexes))
                if qp == 1:
                    new_y = np.array([y[index] for index in idxs_shuf])
                else:
                    new_y = np.array([y[:, index] for index in idxs_shuf]).transpose()
                
                solution = solution = ODR(
                    f, new_x, new_y, b0, dfdbeta=dfdbeta, dfdx=dfdx, job=job,
                    iprint=iprint
                )
                results_H0.append(solution.beta)
            
    dist_param = dst.random_variates(np.array(results), multi=multi)
    
    if H0:
        
        dist_param_H0 = dst.random_variates(np.array(results_H0), multi=multi)
        
        return b0, dist_param, dist_param_H0
    
    else:
        
        return b0, dist_param

def ODR(f, x, y, beta0, dfdbeta=None, dfdx=None, job=None, iprint=None):
    """ Uses the ODR algorithm to fit a general model to a set of data
        
        Parameters:
            
            f (function): User-defined function that can take ndarrays as input. This
                is the function representing the general model
            
            x (ndarray): Values of the independent variable at m points. Shape (m, )
                for a 1-dimensional variable. Shape (q, m) for a q-dimensional variable
            
            y (ndarray): Values of the dependent variable at m points. Shape (m, ) for
                a 1-dimensional variable. Shape (q', m) for a q'-dimensional variable
            
            beta0 (ndarray): Initial guess of the model's parameters. Shape (p, ) for
                p-dimensional parameter space
            
            dfdbeta (function,optional): User-defined function implementing the
                Jacobian matrix of f wrt parameters. The function should return an
                array of shape (q',p,m) containing the values of the Jacobian matrix at
                the m points
            
            dfdx (function,optional): User-defined function implementing the Jacobian
                matrix of f wrt the independent variable. The function should return an
                array of shape (q',q,m) containing the values of the Jacobian matrix at
                the m points
            
            job (str,optional): Determine job settings. See ODRPACK manual page 31
            
            iprint (str,optional): Determine report settings. See ODRPACK manual page
                33-34
        
        Returns:
            
            result (object): An instance of the scipy.odr.Output class with attributes
                that contain the results
    """
    mod = odr.Model(f, fjacb=dfdbeta, fjacd=dfdx)
    
    dat = odr.Data(x, y)
    
    if iprint != None:
        
        if job != None:
            
            problem = odr.ODR(dat, mod, beta0=beta0, iprint=iprint)
            problem.set_job(
                fit_type=int(job[-1]),
                deriv=int(job[-2]),
                var_calc=int(job[-3]),
                del_init=int(job[-4]),
                restart=int(job[-5])
            )
            
        else:
            
            problem = odr.ODR(dat, mod, beta0=beta0, iprint=iprint)
            
    else:
        
        if job != None:
            
            problem = odr.ODR(dat, mod, beta0=beta0)
            problem.set_job(
                fit_type=int(job[-1]),
                deriv=int(job[-2]),
                var_calc=int(job[-3]),del_init=int(job[-4]),
                restart=int(job[-5])
            )
            
        else:
            
            problem = odr.ODR(dat, mod, beta0=beta0)
    
    return problem.run()

""" Special models
"""

""" 1-dimensional linear functions with 2-dimensional parameter space
"""

def flin2d(beta, x):
    """ Defines a 1-dimensional linear function with a 2-dimensional parameter space
    """
    return beta[0] + (beta[1] * x)

def dflin2ddb(beta, x):
    """ Defines the Jacobian wrt the parameters of the 1-dimensional linear function
        with a 2-dimensional parameter space
    """
    if isinstance(x, np.ndarray) == True:
        dfdbval = np.ones((2, x.size))
        dfdbval[0, :] = 1
        dfdbval[1, :] = x
        
    else:
        dfdbval = np.array([1, x])

    return dfdbval

def dflin2ddx(beta, x):
    """ Defines the derivative of the 1-dimensional linear function with a
        2-dimensional parameter space
    """
    if isinstance(x, np.ndarray) == True:
        dfdxval = beta[1] * np.ones(x.size)
        
    else:
        dfdxval = beta[1]
    
    return dfdxval

""" 1-dimensional linear functions with a 1-dimensional parameter space (only slope and
    with x-intercept)
"""

def flin1d_xi(x_intercept=0):
    """ Defines a 1-dimensional linear function with a 1-dimensional parameter space
        (only slope and with x-intercept)
    """
    def flin1d(beta, x):
        return beta[0] * (x - x_intercept)
    return flin1d

def dflin1ddb_xi(x_intercept=0):
    """ Defines the Jacobian wrt the parameters of the 1-dimensional linear function
        with a 1-dimensional parameter space (only slope and with x-intercept)
    """
    def dflin1ddb(beta, x):
        if isinstance(x, np.ndarray) == True:
            dfdbval = np.ones((1, x.size))
            dfdbval[0, :] = x - x_intercept
            
        else:
            dfdbval = np.array([x - x_intercept])
        
        return dfdbval
        
    return dflin1ddb

def dflin1ddx_xi(x_intercept=0):
    """ Defines the derivative of the 1-dimensional linear function with a
        2-dimensional parameter space (only slope and with x-intercept)
    """
    def dflin1ddx(beta, x):
        if isinstance(x, np.ndarray) == True:
            dfdxval = beta[0] * np.ones(x.size)
            
        else:
            dfdxval = beta[0]
            
        return dfdxval
        
    return dflin1ddx

""" 1-dimensional linear functions with a 1-dimensional parameter space (only slope and
    with y-intercept)
"""

def flin1d_yi(y_intercept=0):
    """ Defines a 1-dimensional linear function with a 1-dimensional parameter space
        (only slope and with y-intercept)
    """
    def flin1d(beta, x):
        return y_intercept + beta[0] * x
    return flin1d

def dflin1ddb_yi(y_intercept=0):
    """ Defines the Jacobian wrt the parameters of the 1-dimensional linear function
        with a 1-dimensional parameter space (only slope and with y-intercept)
    """
    def dflin1ddb(beta, x):
        if isinstance(x, np.ndarray) == True:
            dfdbval = np.ones((1, x.size))
            dfdbval[0, :] = x
            
        else:
            dfdbval = np.array([x])
            
        return dfdbval
        
    return dflin1ddb

def dflin1ddx_yi(y_intercept=0):
    """ Defines the derivative of the 1-dimensional linear function with a
        2-dimensional parameter space (only slope and with y-intercept)
    """
    def dflin1ddx(beta, x):
        if isinstance(x, np.ndarray) == True:
            dfdxval = beta[0] * np.ones(x.size)
            
        else:
            dfdxval = beta[0]
            
        return dfdxval
        
    return dflin1ddx

""" 2-dimensional linear functions with 3-dimensional parameter space
"""

def flin3d(beta, x):
    """ Defines a 2-dimensional linear function with a 3-dimensional parameter space
    """
    return beta[0] + (beta[1] * x[0]) + (beta[2] * x[1])

def dflin3ddb(beta, x):
    """ Defines the Jacobian wrt the parameters of the 2-dimensional linear function
        with a 3-dimensional parameter space
    """
    if len(x.shape) == 2:
        dfdbval = np.ones((3, x.size // 2))
        dfdbval[0, :] = 1
        
        if x.shape[0] == 2:
            dfdbval[1, :] = x[0, :]
            dfdbval[2, :] = x[1, :]
            
        else:
            dfdbval[1, :] = x[:, 0]
            dfdbval[2, :] = x[:, 1]
    
    if len(x.shape) == 1:
        dfdbval = np.array([1, x[0], x[1]])
    
    return dfdbval

def dflin3ddx(beta, x):
    """ Defines the gradient of the 2-d linear function with a 3-dimensional parameter
        space
    """
    if len(x.shape) == 2:
        dfdxval = np.ones((2, x.size // 2))
        dfdxval[0, :] = beta[1] * dfdxval[0, :]
        dfdxval[1, :] = beta[2] * dfdxval[1, :]
        
    if len(x.shape) == 1:
        dfdxval = np.array([beta[1], beta[2]])
    
    return dfdxval

""" Equilibrium climate sensitivity (ECS) model from Jimenez-de-la-Cuesta and Mauritsen
    (2019)
"""

def fECS(beta, x):
    """ Defines the equilibrium climate sensitivity (ECS) model from
        Jimenez-de-la-Cuesta and Mauritsen (2019)
    """
    return x / (beta[0] - (beta[1] * x))

def dfECSdb(beta, x):
    """ Defines the Jacobian wrt the parameters of the ECS model from
        Jimenez-de-la-Cuesta and Mauritsen (2019)
    """
    if isinstance(x,np.ndarray) == True:
        dfdbval = np.ones((2, x.size))
        dfdbval[0,:] = -x / ((beta[0] - (beta[1] * x)) ** 2)
        dfdbval[1,:] = (x ** 2) / ((beta[0] - (beta[1] * x)) ** 2)
        
    else:
        dfdbval = np.array([
            -x / ((beta[0] - (beta[2] * x)) ** 2),
            (x ** 2) / ((beta[0] - (beta[2] * x)) ** 2)
        ])
    
    return dfdbval

def dfECSdx(beta, x):
    """ Defines the derivative of the ECS model from Jimenez-de-la-Cuesta and Mauritsen
        (2019)
    """
    if isinstance(x, np.ndarray) == True:
        dfdxval = beta[0] / ((beta[0] - (beta[1] * x)) ** 2)
        
    else:
        dfdxval = beta[0] / ((beta[0] - (beta[1] * x)) ** 2)
        
    return dfdxval