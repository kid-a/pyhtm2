## global import ---------------------------------------------------------------
import numpy as np


def inc_rows_cols(array):
    """ Adds one row and one column to a given array.
    Fills new fields with zeros."""
    (rows, cols) = array.shape
    new_array = np.zeros((rows + 1, cols + 1), dtype=array.dtype)
    new_array[:rows,:cols] = array

    del array
    return new_array
    

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

def resize(array, new_dims):
    """Resize an array, guaranteeing that elements stays in the same position."""
    (rows, cols) = array.shape
    new_array = np.zeros(new_dims)
    new_array[0:rows, 0:cols] = array
    return new_array


if __name__ == "__main__":
    ## profile inc_rows_cols
    import profile

    def enlarge_matrix():
        a = np.random.random((12000, 4))
        for i in range(400):
            a = inc_rows_cols(a)
    
    profile.run("enlarge_matrix()")
    
    
    




    
    
    
