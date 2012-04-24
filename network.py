import numpy as np
import math

## layer type definitions
ENTRY = 0
INTERMEDIATE = 1
OUTPUT = 2


class Network(object):
    def __init__(self, *args, **kwargs):
        self.layers = []
        
    def propagate(self, uFrom, uTo):
        f = self.layers[uFrom]
        t = self.layers[uTo]
        
        ## reset target layer input messages
        for i in range(len(t.nodes)):
            for j in range(len(t.nodes[i])):
                t.nodes[i][j].input_msg = []
            
        if len(t.nodes[0]) == 1: ## if last node
            for i in range(len(f.nodes)):
                for j in range(len(f.nodes[i])):
                    t.nodes[0][0].input_msg.append(f.nodes[i][j].output_msg)
                    
        else:
            for i in range(len(f.nodes)):
                upper_i = math.floor(i / float(len(t.nodes)))
            
                for j in range(len(f.nodes[i])):
                    upper_j = math.floor(j / float(len(t.nodes[0])))
                    msg = f.nodes[i][j].output_msg
                    t.nodes[int(upper_i)][int(upper_j)].input_msg.append(msg)

    def train(self):
        pass


class Layer(object):
    def __init__(self, *args, **kwargs):
        self.nodes = []
        self.spatial_pooler = None
        self.temporal_pooler = None

        ## params
        self.sigma = None
        self.distance_thr = None
        self.node_sharing = False
        self.transition_memory_size = None
        self.group_max_size = None
        self.group_min_size = None
        
    def train(self, uTemporalGap=False):
        if self.node_sharing:  
            ## train just the first node
            spatial_pooler.train_node(self.nodes[0][0], uTemporalGap=uTemporalGap)
            
        else: ## train all nodes in the layer
            for i in range(self.nodes):
                for j in range(self.nodes[i]):
                    spatial_pooler.train_node(self.nodes[i][j], uTemporalGap=uTemporalGap)
                    
    def finalize_training(self):
        if self.node_sharing:
            ## finalize just the first node
            temporal_pooler.finalize_training(self.nodes[0][0])

            ## then copy its state, i.e. coincidences, temporal_groups and PCG
            for i in range(self.nodes):
                for j in range(self.nodes[i]):
                    if i == 0 and j == 0: continue
                    self.nodes[i][j].coincidences = self.nodes[0][0].coincidences
                    self.nodes[i][j].temporal_groups = self.nodes[0][0].temporal_groups
                    self.nodes[i][j].PCG = self.nodes[0][0].PCG
            
        else:
            for i in range(self.nodes):
                for j in range(self.nodes[i]):
                    temporal_pooler.finalize_training(self.nodes[i][j])


class OutputLayer(object):
    def __init__(self, *args, **kwargs):
        self.nodes = []
        self.spatial_pooler = None

        ## params
        self.distance_thr = None
        
    def inference(self): pass
    
    def train(self, uClass):
        spatial_pooler.train_node(self.nodes[0][0], uClass)
        
    def finalize_training(self):
        ## compute class priors
        s = self.nodes[0][0].PCW.sum(axis=0)
        total = self.nodes[0][0].PCW.sum()
        self.cls_prior_prob = s / float(total)
        
        ## normalize the PCW matrix
        self.nodes[0][0].PCW = utils.normalize_over_columns(self.nodes[0][0].PCW)


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
    def __init__(self, params):
        self.network_spec = params
        self.network = Network()

    def build(self):
        """Build the network."""
        for i in range(len(self.network_spec)):
            if i == 0:
                l = self.make_layer(self.network_spec[i], ENTRY)
            elif i == (len(self.network_spec) - 1):
                l = self.make_layer(self.network_spec[i], OUTPUT)
            else:
                l = self.make_layer(self.network_spec[i], INTERMEDIATE)
                
            self.network.layers.append(l)
        
    def make_layer(self, uParams, uType):
        l = Layer()
        
        if uType == ENTRY:
            for i in range(uParams['shape'][0]):
                l.nodes.append([])
                for j in range(uParams['shape'][1]):
                    l.nodes[i].append(Node())
                                
            l.sigma = uParams['sigma']
            l.distance_thr = uParams['distance_thr']
            l.node_sharing = uParams['node_sharing']
            l.transition_memory_size = uParams['transition_memory_size']
            l.group_max_size = uParams['group_max_size']
            l.group_min_size = uParams['group_min_size']

        elif uType == INTERMEDIATE:
            for i in range(uParams['shape'][0]):
                l.nodes.append([])
                for j in range(uParams['shape'][1]):
                    l.nodes[i].append(Node())

            l.distance_thr = uParams['distance_thr']
            l.node_sharing = uParams['node_sharing']
            l.transition_memory_size = uParams['transition_memory_size']
            l.group_max_size = uParams['group_max_size']
            l.group_min_size = uParams['group_min_size']
            
        else: ## uType == OUTPUT
            l.nodes.append(OutputNode())
            l.distance_thr = uParams['distance_thr']
            
        return l


if __name__ == "__main__":
    network_spec = [
        {'shape' : (4,4),
         'sigma' : 25.0,
         'distance_thr' : 55.0,
         'node_sharing' : True,
         'transition_memory_size' : 5,
         'group_max_size' : 5,
         'group_min_size' : 2},

        {'shape' : (2,2),
         'distance_thr' : 0.0,
         'node_sharing' : False,
         'transition_memory_size' : 5,
         'group_max_size' : 5,
         'group_min_size' : 2},

        {'distance_thr' : 0.0}]
    
    b = HTMBuilder(network_spec)
    b.build()
    
    n = b.network
    #print n.layers[0].nodes[3][3]

    for i in range(len(n.layers[0].nodes)):
        for j in range(len(n.layers[0].nodes[i])):
            n.layers[0].nodes[i][j].output_msg = np.array([1,2,3,4])
            
    n.propagate(0, 1)
    print n.layers[1].nodes[0][0].input_msg
