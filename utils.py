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
        np.transpose(array) / np.sum(array, axis=1, dtype=np.float32))


def normalize_over_cols(array):
    """Normalize a vector such that the elements of each column sum up to 1."""
    return array / np.sum(array, axis=0, dtype=np.float32)


def make_symmetric(array):
    """Make an array symmetric summing up its lower and its upper parts."""
    return array + array.T - np.diag(array.diagonal())

    # return \
    #     np.tril( np.tril(array, -1) + np.transpose(np.triu(array, 1)), -1) + \
    #     np.triu( np.transpose(np.tril(array, -1)) + np.triu(array, 1),  1) + \
    #     (array * np.eye(size))


def resize(array, new_dims):
    """Resize an array, guaranteeing that elements stays in the same position."""
    (rows, cols) = array.shape
    new_array = np.zeros(new_dims)
    new_array[0:rows, 0:cols] = array
    return new_array


if __name__ == "__main__":
    import profile
    import gc
    import time

    N = 15000

    a = np.random.randint(100, size=(N, N))
    a = np.array(a, dtype=np.uint16)

    a = make_symmetric(np.array(a))
    a = normalize_over_rows(np.array(a, dtype=np.float32))
    b = np.nan_to_num(a)

    seen = np.random.randint(600, size=(1, N))
    seen = np.array(seen, dtype=np.float32)
    coinc_priors = seen / float(seen.sum())
        
    TC = (np.dot(coinc_priors, b)).flatten()
    
    import temporal_clustering
    
    t0 = time.time()

    p = temporal_clustering.TemporalPooler()
    # profile.runctx("p.greedy_temporal_clustering(TC, b, params)",
    #                globals(),
    #                {'p' : p,
    #                 'TC' : TC,
    #                 'b' : b,
    #                 'params' : {'max_group_size' : 10,
    #                             'top_neighbours' : 3}
    #                 })

    p.greedy_temporal_clustering(TC, b, {'max_group_size' : 10,
                                         'top_neighbours' : 3})
    
    print t0 - time.time()
    

    # profile.runctx("make_symmetric(np.array(a, dtype=np.double))",
    #                globals(),
    #                {'make_symmetric' : make_symmetric,
    #                 'np' : np,
    #                 'a' : a })
    
    



    ## profile inc_rows_cols
    # import profile

    # def enlarge_matrix():
    #     a = np.random.random((12000, 4))
    #     for i in range(400):
    #         a = inc_rows_cols(a)
    
    # profile.run("enlarge_matrix()")
