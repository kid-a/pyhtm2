## global import ---------------------------------------------------------------
import numpy as np
import pytest
import utils


EPSILON = 0.00000001

##------------------------------------------------------------------------------
## global import ---------------------------------------------------------------
##------------------------------------------------------------------------------
@pytest.mark.randomize(("size1", int), min_num=1, max_num=10)
@pytest.mark.randomize(("size2", int), min_num=1, max_num=10, ncalls=5)
def test_inc_rows_cols(size1, size2):
    array = np.random.random((size1, size2))
    (rows, cols) = array.shape
    new_array = utils.inc_rows_cols(array)

    ## assert that size has been incremented both
    ## on rows and cols
    assert rows + 1 == new_array.shape[0]
    assert cols + 1 == new_array.shape[1]

    ## assert that elements have not changed
    for i in range(rows):
        for j in range(cols):
            assert array[i,j] == new_array[i,j]

    ## assert that the new elements are all zeros
    for i in range(cols + 1):
        assert new_array[-1, i] == 0
        
    for i in range(rows + 1):
        assert new_array[i, -1] == 0


# def test_tranpose():
#     (rows, cols) = array.shape
#     t_array = utils.transpose(array)
    
#     ## assert on array shape
#     assert rows == t_array.shape[1]
#     assert cols == t_array.shape[0]

#     ## assert that the array has been transposed
#     for i in range(rows):
#         for j in range(cols):
#             assert array[i,j] == t_array[j,i]
    
    
# def test_normalize_over_rows():
#     (rows, cols) = array.shape
    
#     normalized_array = utils.normalize_over_rows(array)
    
#     ## assert on array shape
#     assert rows == normalized_array.shape[0]
#     assert cols == normalized_array.shape[1]

#     ## assert that each row must sum up to 1
#     for i in range(rows):
#         assert np.sum(normalized_array[i,:]) - 1 < EPSILON
    

# def test_normalize_over_cols():
#     (rows, cols) = array.shape
    
#     normalized_array = utils.normalize_over_cols(array)
    
#     ## assert on array shape
#     assert rows == normalized_array.shape[0]
#     assert cols == normalized_array.shape[1]

#     ## assert that each row must sum up to 1
#     for i in range(cols):
#         assert np.sum(normalized_array[:,i]) - 1 < EPSILON


# # if __name__ == "__main__":
# #     test_inc_rows_cols
