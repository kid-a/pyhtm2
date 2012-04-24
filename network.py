import numpy as np

## layer type definitions
ENTRY = 0
INTERMEDIATE = 1
OUTPUT = 2


class Network(object):
    def __init__(self, *args, **kwargs):
        self.layers = []
        
    def train(self):
        pass


class Layer(object):
    def __init__(self, *args, **kwargs):
        self.nodes = []
        self.spatial_pooler = None

        ## params
        self.sigma = None
        self.distance_thr = None
        self.node_sharing = False
        self.transition_memory_size = None
        self.group_max_size = None
        self.group_min_size = None
        
    def inference(self):
        for i in range(self.nodes):
            for j in range(self.nodes[i]):
                spatial_pooler.train_node(self.nodes[i][j], uClass, uTemporalGap)

        
    def train(self, uClass, uTemporalGap=False):
        if self.node_sharing:  
            ## train just the first node
            spatial_pooler.train_node(self.nodes[0][0], uClass, uTemporalGap)
            
        else: ## train all nodes in the layer
            for i in range(self.nodes):
                for j in range(self.nodes[i]):
                    spatial_pooler.train_node(self.nodes[i][j], uClass, uTemporalGap)
                    
    def finalize_training(self):
        if self.node_sharing:
            ## finalize just the first node
            temporal_pooler.finalize_training(self.nodes[0][0])

            ## then copy its state
            ## clone state
            ## !FIXME implement cloning
            
        else:
            for i in range(self.nodes):
                for j in range(self.nodes[i]):
                    temporal_pooler.finalize_training(self.nodes[i][j])


class Node(object):
    def __init__(self, *args, **kwargs):
        self.coincidences = np.array([[]])
        self.temporal_groups = None
        self.input_msg = np.array([])
        self.output_msg = np.array([])
        self.seen = np.array([])
        self.TAM = np.array([[]])
        self.PCG = np.array([[]])
        self.k = None
        self.k_prev = []
        
        ## !FIXME a clone method here?
        

class OutputNode(object):
    def __init__(self, *args, **kwargs):
        self.coincidences = np.array([[]])
        self.input_msg = np.array([])
        self.output_msg = np.array([])
        self.k = None
        self.cls_prior_prob = np.array([])
        self.PCW = np.array([[]])

        ## !FIXME a clone method here?


class HTMBuilder(object):
    def __init__(self, *args, **kwargs):
        self.network = Network()
        
    def make_layer(self, params):
        pass
