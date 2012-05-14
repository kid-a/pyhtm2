## global import ---------------------------------------------------------------
import numpy as np
import pytest


## local import ----------------------------------------------------------------
import spatial_clustering
from spatial_clustering import EntrySpatialPooler
from network import NodeFactory
import config


## test the Entry Temporal Pooler
@pytest.mark.randomize(("size1", int), min_num=2, max_num=10)
@pytest.mark.randomize(("size2", int), min_num=2, max_num=10, ncalls=3)
def test_entry_closest_coinc(size1, size2):
    p = EntrySpatialPooler()
    
    input_msg = np.random.randint(255, size=size1)
    input_msg.dtype = np.uint8

    c = []

    for i in range(size2):
        r = np.random.randint(255, size=size1)
        r.dtype = np.uint8
        c.append(r)
        
        
    min_ = np.sqrt(np.sum(np.power(c[0] - input_msg, 2)))
        
    k = 0
    print 0, min_

    for i in range(1, len(c)):
        dist = np.sqrt(np.sum(np.power(c[i] - input_msg, 2)))
        print i, dist
        
        if dist < min_:
            min_ = dist
            k = i

    coinc_matrix = c[0]
    for i in range(1, len(c)):
        coinc_matrix = np.vstack((coinc_matrix, c[i]))
                         
    res = p.closest_coincidence(coinc_matrix, input_msg)
        
    assert res[0] == k
    assert res[1] == min_
    assert res[1].dtype == np.float64

## test the select_active_coinc when len(coincidences) == 0
def select_active_coinc1():
    n = make_node((0,0), network.ENTRY, config.usps_net)
    s = n.state
    
    input_msg = np.random.randint(255, size=size1)
    input_msg.dtype = np.uint8

    s.state['input_msg'] = input_msg
    s.strategy['trainer'].select_active_coinc(s.state)
    
    assert s.state['coincidences'][0] == input_msg
    assert s.state['coincidences'].dtype == np.uint8
    assert s.state['TAM'].shape == (1,1)
    assert s.state['seen'] == 1

## test the select_active_coinc when there is more than one coincidence
@pytest.mark.randomize(("coinc_size", int), min_num=2, max_num=10)
def select_active_coinc2():
    n = make_node((0,0), network.ENTRY, config.usps_net)
    s = n.state
    
    input_msg = np.random.randint(255, size=size1)
    input_msg.dtype = np.uint8

    s.state['input_msg'] = input_msg
    s.state['coincidences'] = np.random.randint(255, shape(coinc_size, 4))
    s.state['seen'] = np.random.randint(10, size=coinc_size)
    s.state['TAM'] = np.random.randint(10, size=(coinc_size, coinc_size))

    s.strategy['trainer'].select_active_coinc(s.state)
    
    assert s.state['coincidences'].dtype == np.uint8


## test the closest_coincidence method
@pytest.mark.randomize(("size1", int), min_num=1, max_num=10)
@pytest.mark.randomize(("size2", int), min_num=1, max_num=10, ncalls=3)
def test_wdix_distance2(size1, size2):
    array = np.random.randint(0, 100, size=(12000, 4))
    distances = spatial_clustering.widx_distance2(array)
    
    print distances

    (rows, cols) = array.shape

    ## assert on the number of non-zero elements in each row
    for i in range(rows):
        non_zeros_count = 0
        for j in range(cols):
            if array[i,j] != 0: non_zeros_count += 1
            
        assert non_zeros_count == distances[i]
            
        
        
    
    
    
    

