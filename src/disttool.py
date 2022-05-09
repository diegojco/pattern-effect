""" Module containing functions to sample random variables
"""
from __future__ import division
import random as rn
import numpy as np
import scipy.stats as st

def random_variates(data, multi=False):
    """ Provides generators of random variates that follow the distribution of a random
        variable described by its data histogram.
        
        Parameters:
            
            data (ndarray): An array of shape (m,) or (m,n), where m is the number of
                samples of the random variable and n is its dimensionality
            
            multi (boolean,optional): `True` for providing a generator that considers
                the covariance in the n-dimensional random variable
            
        Returns:
            
            dist,multidist (function): For 1-d random variables and `multi=False`, the
                function only returns univariate generators using the `rvs` method of a
                `scipy.stats.rv_histogram` instance. If `multi=True` then it returns
                the univariate generators for each random variable as well as a
                multivariate generator that takes into account covariance in the
                n-dimensional random variable
    """
    if len(data.shape) == 1:                                 # CASE: 1-d random variable
        
        if not multi:
            
            bins = np.histogram_bin_edges(data, bins="auto")            # Histogram bins
            binc = bins[:-1] + (np.diff(bins) / 2)                         # Bin centers
            dist = np.histogram(data, bins)                   # Histogram using the bins
            dist = st.rv_histogram(dist).rvs             # From random variate generator
            
            return dist
        
        else:
            
            print("`multi=True` for 1-d variable is a nonsense")
        
    else:                                                    # CASE: n-d random variable
        bins = [
            np.histogram_bin_edges(data[:, i], bins="auto")
            for i in range(data.shape[1])
        ]
        binc = [
            b[:-1] + (np.diff(b) / 2)
            for b in bins
        ]
        dist = [
            np.histogram(data[:, i], bins[i])
            for i in range(data.shape[1])
        ]
        dist = [
            st.rv_histogram(d).rvs
            for d in dist
        ]
        
        if multi:                  # SUBCASE: n-d random variable considering covariance
            multidist = np.histogramdd(data, bins=bins)[0] # Multi-dimensional histogram
            
            j_cdf = np.cumsum(multidist.ravel())                             # Joint CDF
            j_cdf /= j_cdf[-1]                                  # Normalised from 0 to 1
            
            multidist = lambda size: multi_rvs(size, j_cdf, binc) # Random generator
            
            return dist, multidist
            
        else:
            
            return dist

def multi_rvs(samples, raveled_cdf, bin_centers):
    """ Samples a n-dimensional random variable using its cumulative distribution
        function obtained from a data histogram
        
        Parameters:
            
            samples (int): The number of the drawn samples
            
            raveled_cdf (ndarray): Cumulative distribution function (CDF) obtained from
                the flattened histogram of the n-dimensional random variable. Usually
                obtained with `np.cumsum(np.histogramdd(data)[0].ravel())`
            
            bin_centers (list of arrays): List that contains n arrays with the
                histogram bin centers
            
        Returns:
            
            random (ndarray): Array of shape (samples, n) that contains the random
                samples
    """
    vals_bins = [
        np.searchsorted(raveled_cdf, rn.random())
        for i in range(samples)
    ]
    
    indexes = np.unravel_index(
        vals_bins, tuple( len(b) for b in bin_centers )
    )
    
    random = np.array(
        [
            [
                bin_centers[i][index]
                for index in indexes[i]
            ]
            for i in range(len(bin_centers))
        ]
    ).transpose()
    
    return random

def percentile_range_print(samples, p=0.05, dec=2):
    """ Function that prints the p to (1 - p) percentile range and the median from a
        sample distribution
    """
    prange = percentile_range(samples, p=p)
    return f"{prange[1]:0.{dec}f} ({prange[0]:0.{dec}f}—{prange[2]:0.{dec}f})"

def percentile_range(samples, p=0.05):
    """ Function that obtains the p to (1 - p) percentile range and the median from a
        sample distribution
    """
    return [
        np.percentile(samples, p * 100),
        np.percentile(samples, 50),
        np.percentile(samples, (1 - p) * 100)
    ]

def significance(dist_H0, dist, epsilon=0.00001):
    """ Function that compares a distribution with the distribution of a null hypothesis
        and returns the corresponding p-value of the distribution's median
    """
    return find_pvalue(dist_H0, np.percentile(dist, 50), epsilon=epsilon)

def find_pvalue(sample, value, epsilon=0.00001):
    """ Function that obtains the p-value of a certain value wrt a distribution
    """
    pvalue = find_percentile(sample, value, epsilon=epsilon)
    
    if pvalue > 0.5:
        pvalue = 1 - pvalue
        
    return pvalue

def find_percentile(samples, value, epsilon=0.00001):
    """ Function that obtains the percentile at a certain value of the random variable
    """
    pi = 0
    pf = 1
    
    pc = pi + ((pf - pi) / 2)
    
    val = np.percentile(samples, pc * 100)
    
    test = value - val
    
    while np.absolute(test) > epsilon:
        
        if test < 0:
            pf = pc
        else:
            pi = pc
        
        pc = pi + ((pf - pi) / 2)
        
        val = np.percentile(samples, pc * 100)
        
        test = value - val
    
    return pc