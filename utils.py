## a module for array/matrix manipulation

import numpy as np


def inc_rows_cols(array):
    """ Adds one row and one column to a given array.
    Fills new fields with zeros."""
    (rows, cols) = array.shape
    array = np.hstack((array, np.zeros((rows, 1))))
    array = np.vstack((array, np.zeros((1, cols + 1))))

    return array
    

def transpose(array):
    """ Transpose an array."""
    return np.transpose(array)


def normalize_over_rows(array):
    """Normalize a vector such that the elements of each row sum up to 1."""
    return np.transpose(
        np.transpose(array) / np.sum(array, axis=1, dtype=np.double))


def normalize_over_cols(array):
    """Normalize a vector such that the elements of each column sum up to 1."""
    return array / np.sum(array, axis=0, dtype=np.double)


def make_symmetric(array):
    """Make an array symmetric summing up its lower and its upper parts."""
    (size, size) = array.shape
    return \
        np.tril( np.tril(array, -1) + np.transpose(np.triu(array, 1)), -1) + \
        np.triu( np.transpose(np.tril(array, -1)) + np.triu(array, 1),  1) + \
        (array * np.eye(size))
    




    
    
    
