usps_net = [
    ## layer 0
    {'shape' : (4,4),
     'sigma' : 150.0,
     'distance_thr' : 224.0,
     'node_sharing' : True,
     'transition_memory_size' : 2,
     'top_neighbours' : 3,
     'max_group_size' : 10,
     'min_group_size' : 4},
    
    ## layer 1
    {'shape' : (2,2),
     'distance_thr' : 0.0,
     'node_sharing' : False,
     'transition_memory_size' : 5,
     'top_neighbours' : 2,
     'max_group_size' : 12,
     'min_group_size' : 2},
    
    ## layer 2
    {'shape' : (1,1),
     'distance_thr' : 0.0}]

test_net = [
    ## layer 0
    {'shape' : (4,4),
     'sigma' : 150.0,
     'distance_thr' : 0.0,
     'node_sharing' : True,
     'transition_memory_size' : 9,
     'top_neighbours' : 3,
     'max_group_size' : 10,
     'min_group_size' : 4},
    
    ## layer 1
    {'shape' : (2,2),
     'distance_thr' : 0.0,
     'node_sharing' : False,
     'transition_memory_size' : 5,
     'top_neighbours' : 2,
     'max_group_size' : 12,
     'min_group_size' : 2},
    
    ## layer 2
    {'shape' : (1,1),
     'distance_thr' : 0.0}]
