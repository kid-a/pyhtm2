import numpy as np
import random
import pytest

import inference
import network
import utils

def make_random_array(length, limit_max=1000):
    a = np.random.random_integers(1, limit_max, length)
    #a = np.random.random((1, length))
    return a

def make_random_coincidences(size, length, limit_max=1000):    
    c = make_random_array(size, limit_max)

    for i in range(length - 1):
        c = np.vstack((c, make_random_array(size, limit_max)))

    return c

def make_random_lambda(size, length):
    c = [make_random_array(size, size)]

    for i in range(length - 1):
        c.append(make_random_array(size, size))
        
    return c
    


EPSILON = 0.00000001
        

def test_dens_over_coinc():
    length = 16
    how_many_coinc = 200
    c = make_random_coincidences(length, how_many_coinc)
    i = make_random_array(length)

    ## test for network.ENTRY
    ##
    y = inference.dens_over_coinc(c, i, network.ENTRY)
    
    ## size of y equals the number of coinc
    assert y.shape[0] == how_many_coinc

    ## assert on each element of y
    for j in range(how_many_coinc):
        expected = np.linalg.norm(c[j,:] - i)
        expected = np.exp(- np.power(expected, 2) / 1.0)
        
        assert expected - y[j] < EPSILON

    ## test for network.INTERMEDIATE and network.OUTPUT
    ##
    c = make_random_coincidences(length, how_many_coinc, length - 1)
    i = make_random_lambda(length, how_many_coinc - 1)
    y = inference.dens_over_coinc(c, i, network.INTERMEDIATE)
    
    ## size of y equals the number of coinc
    assert y.shape[0] == how_many_coinc

    ## assert on each element of y
    for j in range(how_many_coinc):
        expected = np.array([])

        for k in range(length):
            expected = np.append(expected, i[k][c[j][k]])
                    
        assert np.prod(expected) - y[j] < EPSILON    
