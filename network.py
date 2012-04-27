from multiprocessing import Pool
import numpy as np
import math

from spatial_clustering import EntrySpatialPooler
from spatial_clustering import IntermediateSpatialPooler
from spatial_clustering import OutputSpatialPooler
from temporal_clustering import TemporalPooler
import inference
from inference import EntryInferenceMaker
from inference import IntermediateInferenceMaker
from inference import OutputInferenceMaker

import utils

## layer type definitions
ENTRY = 0
INTERMEDIATE = 1
OUTPUT = 2


class Network(object):
    def __init__(self, *args, **kwargs):
        self.layers = []
        
    def expose(self, uInput):
        (input_height, input_width) = uInput.shape
        (layer_height, layer_width) = (len(self.layers[0].nodes), len(self.layers[0].nodes))
        
        patch_height = input_height / layer_height
        patch_width = input_width / layer_width
    
        starting_point_h = 0;
        starting_point_w = 0;
        
        for i in range(layer_height):
            for j in range(layer_width):
                node = self.layers[0].nodes[i][j]
                patch = uInput[starting_point_h : starting_point_h + patch_height,
                               starting_point_w : starting_point_w + patch_width]
                
                node.input_msg = np.reshape(patch, (1, patch.size))
                
                starting_point_w += patch_width
                
            starting_point_w = 0
            starting_point_h += patch_height

        
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

    def train(self, sequences):
        
        for i in range(len(self.layers)):
            if i == 0: ## entry
                for cls in sequences['entry']:
                    sequence = sequences['entry'][cls]

                    for (s,t) in sequence:
                        for j in range(len(s)):
                            
                            if j in t: temporal_gap = True
                            else: temporal_gap = False

                            if self.layers[i].node_sharing:
                                self.layers[i].nodes[0][0].input_msg = \
                                    s[j].reshape((1, s[j].size))
                        
                                self.layers[0].train(temporal_gap)

                            ## !FIXME node sharing set to false won't work
                            # else:
                            #     for l in range(len(self.layers[i].nodes)):
                            #         for k in range(len(self.layers[i].nodes[l])):
                            #             self.layers[i].nodes[l][k].input_msg = \
                            #                 e[0].reshape((1, e[0].size))
                                
                            #             self.layers[0].train()

                self.layers[0].finalize_training()

            elif i == (len(self.layers) - 1): ## output
                for cls in sequences['output']:
                    sequence = sequences['output'][cls]
                    
                    for (s,t) in sequence:
                        for j in range(len(s)):
                            
                            if j in t: temporal_gap = True
                            else: temporal_gap = False
                            
                            self.expose(s[j])

                            for m in range(i):
                                self.layers[m].inference()
                                self.propagate(m, m + 1)

                            self.layers[i].train(cls)
                            
                self.layers[i].finalize_training()
                                    
            else: ## intermediate
                for cls in sequences['intermediate']:
                    sequence = sequences['intermediate'][cls]

                    for (s,t) in sequence:
                        for j in range(len(s)):
                            
                            if j in t: temporal_gap = True
                            else: temporal_gap = False
                            
                            self.expose(s[j])

                            for m in range(i):
                                self.layers[m].inference()
                                self.propagate(m, m + 1)

                            self.layers[i].train(temporal_gap)
                            
                self.layers[i].finalize_training()

    def inference(self):
        pass


class Layer(object):
    def __init__(self, *args, **kwargs):
        self.type = None
        self.nodes = []
        self.spatial_pooler = None
        self.temporal_pooler = None
        self.pool = Pool(processes=20)
        
        ## params
        self.sigma = None
        self.distance_thr = None
        self.node_sharing = False
        self.transition_memory_size = None
        self.max_group_size = None
        self.min_group_size = None
        
    def inference(self):
        results = []
        for i in range(len(self.nodes)):
            results.append([])
            for j in range(len(self.nodes[i])):
                if self.sigma != None:
                    #self.inference_maker.inference(self.nodes[i][j], self.sigma)
                    results[i].append(
                        self.pool.apply_async(
                            inference.inference,
                            [self.nodes[i][j], self.type, self.sigma]))
                else:
                    results[i].append(
                        self.pool.apply_async(
                            inference.inference,
                            [self.nodes[i][j], self.type]))
                    #self.inference_maker.inference(self.nodes[i][j])
                    
        for i in range(len(results)):
            for j in range(len(results[i])):
                results[i][j].wait()
                self.nodes[i][j].output_msg = \
                    results[i][j].get()
                    
        
    def train(self, uTemporalGap=False):
        if self.node_sharing:  
            ## train just the first node
            self.spatial_pooler.train_node(self.nodes[0][0], uTemporalGap=uTemporalGap)
            
        else: ## train all nodes in the layer
            for i in range(len(self.nodes)):
                for j in range(len(self.nodes[i])):
                    self.spatial_pooler.train_node(self.nodes[i][j], uTemporalGap=uTemporalGap)
                    
    def finalize_training(self):
        if self.node_sharing:
            ## finalize just the first node
            self.temporal_pooler.finalize_training(self.nodes[0][0])

            ## then copy its state, i.e. coincidences, temporal_groups and PCG
            for i in range(len(self.nodes)):
                for j in range(len(self.nodes[i])):
                    if i == 0 and j == 0: continue
                    self.nodes[i][j].coincidences = self.nodes[0][0].coincidences
                    self.nodes[i][j].temporal_groups = self.nodes[0][0].temporal_groups
                    self.nodes[i][j].PCG = self.nodes[0][0].PCG
            
        else:
            for i in range(len(self.nodes)):
                for j in range(len(self.nodes[i])):
                    self.temporal_pooler.finalize_training(self.nodes[i][j])


class OutputLayer(object):
    def __init__(self, *args, **kwargs):
        self.nodes = []
        self.spatial_pooler = None
        self.inference_maker = None

        ## params
        self.distance_thr = None
        
    def inference(self):
        self.inference_maker.inference(self.nodes[0][0])
    
    def train(self, uClass):
        self.spatial_pooler.train_node(self.nodes[0][0], uClass)
        
    def finalize_training(self):
        ## compute class priors
        s = self.nodes[0][0].PCW.sum(axis=0)
        total = self.nodes[0][0].PCW.sum()
        self.cls_prior_prob = s / float(total)
        
        ## normalize the PCW matrix
        self.nodes[0][0].PCW = utils.normalize_over_cols(self.nodes[0][0].PCW)


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

                l.spatial_pooler = EntrySpatialPooler(
                    self.network_spec[i]['distance_thr'],
                    self.network_spec[i]['transition_memory_size'])
                
                l.temporal_pooler = TemporalPooler(
                    self.network_spec[i]['max_group_size'],
                    self.network_spec[i]['min_group_size'],
                    self.network_spec[i]['top_neighbours'])

                l.type = ENTRY
                                                   
            elif i == (len(self.network_spec) - 1):
                l = self.make_layer(self.network_spec[i], OUTPUT)

                l.spatial_pooler = OutputSpatialPooler(
                    self.network_spec[i]['distance_thr'])

                l.type = OUTPUT

            else:
                l = self.make_layer(self.network_spec[i], INTERMEDIATE)

                l.spatial_pooler = IntermediateSpatialPooler(
                    self.network_spec[i]['distance_thr'],
                    self.network_spec[i]['transition_memory_size'])

                l.temporal_pooler = TemporalPooler(
                    self.network_spec[i]['max_group_size'],
                    self.network_spec[i]['min_group_size'],
                    self.network_spec[i]['top_neighbours'])

                l.type = INTERMEDIATE
                
            self.network.layers.append(l)
        
    def make_layer(self, uParams, uType):        
        ## !FIXME lots of unuseful params here
        if uType == ENTRY:
            l = Layer()
            for i in range(uParams['shape'][0]):
                l.nodes.append([])
                for j in range(uParams['shape'][1]):
                    l.nodes[i].append(Node())
                                
            l.sigma = uParams['sigma']
            #l.distance_thr = uParams['distance_thr']
            l.node_sharing = uParams['node_sharing']
            # l.transition_memory_size = uParams['transition_memory_size']
            # l.max_group_size = uParams['max_group_size']
            # l.min_group_size = uParams['min_group_size']

        elif uType == INTERMEDIATE:
            l = Layer()
            for i in range(uParams['shape'][0]):
                l.nodes.append([])
                for j in range(uParams['shape'][1]):
                    l.nodes[i].append(Node())

            #l.distance_thr = uParams['distance_thr']
            l.node_sharing = uParams['node_sharing']
            # l.transition_memory_size = uParams['transition_memory_size']
            # l.max_group_size = uParams['max_group_size']
            # l.min_group_size = uParams['min_group_size']
            
        else: ## uType == OUTPUT
            l = OutputLayer()
            l.nodes.append([OutputNode()])
            #l.distance_thr = uParams['distance_thr']
            
        return l


if __name__ == "__main__":
    network_spec = [
        {'shape' : (4,4),
         'sigma' : 25.0,
         'distance_thr' : 55.0,
         'node_sharing' : True,
         'transition_memory_size' : 5,
         'max_group_size' : 5,
         'min_group_size' : 2},

        {'shape' : (2,2),
         'distance_thr' : 0.0,
         'node_sharing' : False,
         'transition_memory_size' : 5,
         'max_group_size' : 5,
         'min_group_size' : 2},

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
