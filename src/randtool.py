""" Module containing functions to generate random combinations
"""
from __future__ import division
import math as mt
import random as rn

def random_combinations(iterable, r, l):
    """ Calculates l random combinations of r elements from an iterable and returns a
        light-weight iterator
    """
    pool = tuple(iterable)
    n = len(pool)
    
    n_combinations = nCr(n, r)
    if l > n_combinations:
        l = n_combinations
    
    combinations = set()
    while len(combinations) < l:
        combinations.add(tuple(sorted(rn.sample(range(n), r))))
    
    def filtro(combi):
        return tuple(pool[index] for index in combi)
    
    combinations = map(filtro, combinations)
    
    return combinations

def number_nCr_geq2(n, r, total=1000000):
    """ Calculates the number of random samples drawn to fill approximately the ratio
        given by ratio_nCr_geq2 function, given a total number of samples
    """
    return int(mt.ceil(ratio_nCr_geq2(n, r) * total))

def ratio_nCr_geq2(n, r):
    """ Calculates the ratio of the number of the subsets of cardinality r of a set to
        the number of subsets of cardinality greater than one
    """
    return nCr(n, r) / total_nCr_geq2(n)

def ratio_nCr(n, r):
    """ Calculates the ratio of the number of the subsets of cardinality r of a set to
        the number of non-empty subsets
    """
    return nCr(n, r) / total_nCr(n)

def total_nCr_geq2(n):
    """ Calculates the cardinality of the power set minus the empty set and the subsets
        of cardinality one
    """
    return total_nCr(n) - n

def total_nCr(n):
    """ Calculates the cardinality of the power set excluding the empty set
        This number is the cardinality of the power set minus one
    """
    return card_powerset(n) - 1

def card_powerset(n):
    """ Calculates the number of subsets from a set of cardinality n
        This number is the cardinality of the power set
    """
    return 2 ** n

def nCr(n, r):
    """ Function that calculates the number of subsets of cardinality r that exist in a
        set of cardinality n
    """
    f = mt.factorial
    return f(n) // (f(r) * f(n - r))