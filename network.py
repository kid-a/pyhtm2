## layer type definitions
ENTRY = 0
INTERMEDIATE = 1
OUTPUT = 2


class Network(object):
    def __init__(self, *args, **kwargs):
        self.layers = []


class Layer(object):
    def __init__(self, *args, **kwargs):
        self.sigma = None
        self.distance_thr = None 
        self.node_sharing = False
        self.transition_memory_size = None
        self.group_max_size = None
        self.group_min_size = None


class Node(object):
    def __init__(self, *args, **kwargs):
        self.coincidences = None ## np.array
        self.input_msg = None    ## np.array
        self.seen = None         ## np.array
        self.TAM = None          ## np.array
        self.k = None            ## integer
        self.k_prev = []         ## list of integers
    
    
class EntryNode(object):
    pass


class HTMBuilder(object):
    def __init__(self, *args, **kwargs):
        self.network = Network()
        
    def make_layer(self, params):
        pass
