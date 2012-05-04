## global import ---------------------------------------------------------------
from multiprocessing import Process
from multiprocessing import Queue
import numpy as np
import math
import time


## local import ----------------------------------------------------------------
from spatial_clustering import EntrySpatialPooler
from spatial_clustering import IntermediateSpatialPooler
from spatial_clustering import OutputSpatialPooler
from temporal_clustering import TemporalPooler
from inference import EntryInferenceMaker
from inference import IntermediateInferenceMaker
from inference import OutputInferenceMaker
from debug import debug_print
import utils


## layer/node type definitions -------------------------------------------------
ENTRY = 0
INTERMEDIATE = 1
OUTPUT = 2


## -----------------------------------------------------------------------------
## save function
## -----------------------------------------------------------------------------
def save(uNetwork, uPath):
    """Saves a network on file, in the speficied path."""
    ## save the network spec
    out = open(uPath + "saved_net.py", "w")
    out.write("spec = " + str(uNetwork.spec))
    out.write("\n")
    out.write("\n")

    
    ## for each layer, save the node's configuration
    layers = uNetwork.layers

    out.write("layers = [")
    for l in range(len(layers) - 1):
        out.write("[")
        for i in range(len(layers[l].nodes)):
            out.write("[")
            for j in range(len(layers[l].nodes[i])):
                layers[l].nodes[i][j].input_channel.put("clone_state")
                state = layers[l].nodes[i][j].output_channel.get()
        
                out.write("{")
                out.write("'coincidences' : " + str(state['coincidences'].tolist()) + ",")
                out.write("'temporal_groups' : " + str(state['temporal_groups']) + ",")
                out.write("'PCG' : " + str(state['PCG'].tolist()))
                out.write("},")
            out.write("],")
        out.write("],")
    
    ## save the last node, as well
    layers[-1].nodes[0][0].input_channel.put("clone_state")
    state = layers[-1].nodes[0][0].output_channel.get()
    out.write("[[")
    out.write("{")
    out.write("'coincidences' : " + str(state['coincidences'].tolist()) + ",")
    out.write("'PCW' : " + str(state['PCW'].tolist()) + ",")
    out.write("'cls_prior_prob' : " + str(state['cls_prior_prob'].tolist()))
    out.write("}")
    out.write("]]]")
    out.close()


## -----------------------------------------------------------------------------
## load function
## -----------------------------------------------------------------------------
def load(uPath):
    """Loads a network from file."""
    import sys
    sys.path.append(uPath)
    import saved_net
        
    builder = NetworkBuilder(saved_net.spec)
    htm = builder.build()
    htm.start()
    
    ## restore each node state
    layers = htm.layers
    layers_spec = saved_net.layers
    
    for l in range(len(layers_spec)):
        for i in range(len(layers_spec[l])):
            for j in range(len(layers_spec[l][i])):
                ## !FIXME lists must be converted to np.arrays

                layers_spec[l][i][j]['coincidences'] = \
                    np.array(layers_spec[l][i][j]['coincidences'])
                
                try: 
                    layers_spec[l][i][j]['PCG'] = \
                        np.array(layers_spec[l][i][j]['PCG'])
                    
                except:
                    layers_spec[l][i][j]['cls_prior_prob'] = \
                        np.array(layers_spec[l][i][j]['cls_prior_prob'])

                    layers_spec[l][i][j]['PCW'] = \
                        np.array(layers_spec[l][i][j]['PCW'])

                layers[l].nodes[i][j].input_channel.put(("set_state", 
                                                         layers_spec[l][i][j]))
                
    return htm


## -----------------------------------------------------------------------------
## Node Class
## -----------------------------------------------------------------------------
class Node(Process):
    """Implements an HTM node."""
    def __init__(self, uState, uStrategy):
        super(Node, self).__init__()
        self.state = uState
        self.strategy = uStrategy
        self.input_channel = Queue()
        self.output_channel = Queue()
        
    def set_state(self, uState):
        """Set the state of a node. This operation makes sense only for all nodes
        except the output."""
        self.strategy['state_handler'].set_state(self.state, uState)

    def clone_state(self):
        """Clone a node state. This operation makes sense only for all nodes
        except the output."""
        return self.strategy['state_handler'].clone(self.state)
        
    def run(self):
        while True:
            msg = self.input_channel.get()
            #debug_print(str(self.state['name']) + str(msg))

            if msg == "finalize":
                debug_print("Finalizing node " + str(self.state['name']))
                self.strategy['finalizer'].finalize(self.state)
                self.output_channel.put("ok")

            elif msg == "inference":
                debug_print("Doing inference on node " + str(self.state['name']))
                self.strategy['inference_maker'].inference(self.state)
                # debug_print("Node " + str(self.state['name']) + " output message: " +\
                #                 str(self.state['output_msg']))
                self.output_channel.put("ok")

            elif msg == "get_output":
                self.output_channel.put(self.state['output_msg'])
                
            elif msg == "clone_state":
                debug_print("Cloning node " + str(self.state['name']) + " state")
                self.output_channel.put(self.clone_state())

            elif msg == "reset_input":
                self.state['input_msg'] = []
                # debug_print(str(self.state['name']) + \
                #                 " input has been reset")

            elif msg[0] == "set_state":
                debug_print("Setting state on node " + str(self.state['name']))
                self.set_state(msg[1])
                                
            elif msg[0] == "train":
                # debug_print("Training node " + str(self.state['name']))

                self.strategy['trainer'].train(self.state, msg[1])
                
                debug_print("Node " + str(self.state['name']) + " coincidences: " + \
                                str(self.state['coincidences'].shape))
                                       

                # debug_print("Node " + str(self.state['name']) + \
                #                 " new coincidences:" + \
                #                 str(self.state['coincidences']))

                self.output_channel.put("ok")
            
            elif msg[0] == "set_input":
                self.state['input_msg'] = msg[1]

                # debug_print(str(self.state['name']) + \
                #                 " new input: " + \
                #                 str(self.state['input_msg']))

                self.output_channel.put("ok")

            elif msg[0] == "append_input":
                self.state['input_msg'].append(msg[1])

                # debug_print(str(self.state['name']) + \
                #                 " new input: " + \
                #                 str(self.state['input_msg']))
                
                self.output_channel.put("ok")

            else:
                debug_print(str(self.state['name']) + \
                                ": received unknown message")


## -----------------------------------------------------------------------------
## Layer Class
## -----------------------------------------------------------------------------
class Layer(object):
    def __init__(self, *args, **kwargs):
        self.nodes = [[]]
        self.node_sharing = False

    def train(self, uInputInfo):
        """Train the layer on the current input pattern."""
        if self.node_sharing:
            ## train just one node,
            self.nodes[0][0].input_channel.put(("train", uInputInfo))
            self.nodes[0][0].output_channel.get()
            
        else:
            ## start each node's training
            for i in range(len(self.nodes)):
                for j in range(len(self.nodes[i])):
                    self.nodes[i][j].input_channel.put(("train", uInputInfo))

            ## wait for the training to be finished
            for i in range(len(self.nodes)):
                for j in range(len(self.nodes[i])):
                    self.nodes[i][j].output_channel.get()
        
    def finalize(self):
        """Finalize training on each node."""
        if self.node_sharing:
            ## finalize just one node,
            self.nodes[0][0].input_channel.put("finalize")
            self.nodes[0][0].output_channel.get()

            ## then clone its state
            ## into all the other nodes
            self.nodes[0][0].input_channel.put("clone_state")
            state = self.nodes[0][0].output_channel.get()

            for i in range(len(self.nodes)):
                for j in range(len(self.nodes[i])):
                    if i == 0 and j == 0: continue
                    self.nodes[i][j].input_channel.put(("set_state", state))

        else:
            ## start each node's finalization procedure
            for i in range(len(self.nodes)):
                for j in range(len(self.nodes[i])):
                    self.nodes[i][j].input_channel.put("finalize")

            ## wait for the finalization to be completed
            for i in range(len(self.nodes)):
                for j in range(len(self.nodes[i])):
                    self.nodes[i][j].output_channel.get()

    def inference(self):
        """Perform inference on the current input."""
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes[i])):
                self.nodes[i][j].input_channel.put("inference")
                
        ## wait for the finalization to be completed
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes[i])):
                self.nodes[i][j].output_channel.get()


## -----------------------------------------------------------------------------
## Network Class
## -----------------------------------------------------------------------------
class Network(object):
    def __init__(self, *args, **kwargs):
        self.layers = []
        self.spec = []

    def start(self):
        for layer in self.layers:
            for i in range(len(layer.nodes)):
                for j in range(len(layer.nodes[i])):
                    layer.nodes[i][j].start()

    def train(self, uTrainingSequences):
        """Train the network on the given training sequences."""
        layers_type = [INTERMEDIATE for s in self.layers]
        layers_type[0] = ENTRY
        layers_type[-1] = OUTPUT
        
        ## for each layer
        for i in range(len(self.layers)): 

            ## for each training sequence
            for pattern in uTrainingSequences[layers_type[i]]:
                (input_raw, input_info) = pattern

                if i == ENTRY: self.expose(input_raw, uJustUpperLeftCorner=True)
                else: self.expose(input_raw)
                    
                for m in range(i):
                    self.layers[m].inference()
                    self.propagate(m, m + 1)

                self.layers[i].train(input_info)

            self.layers[i].finalize()

    def inference(self, uInput):
        """After the network has learned, make inference on a new input."""
        self.expose(uInput)
        for m in range(len(self.layers) - 1):
            self.layers[m].inference()
            self.propagate(m, m + 1)

        ## make inference on the output node
        self.layers[-1].nodes[0][0].input_channel.put("inference")
        self.layers[-1].nodes[0][0].output_channel.get()

        ## read its output
        self.layers[-1].nodes[0][0].input_channel.put("get_output")
        msg = self.layers[-1].nodes[0][0].output_channel.get()
        return msg

    def propagate(self, uFrom, uTo): 
        f = self.layers[uFrom]
        t = self.layers[uTo]
        
        for i in range(len(t.nodes)):
            for j in range(len(t.nodes[i])):
                t.nodes[i][j].input_channel.put("reset_input")

        if len(t.nodes[0]) == 1: ## if last node
            for i in range(len(f.nodes)):
                for j in range(len(f.nodes[i])):
                    f.nodes[i][j].input_channel.put("get_output")
                    msg = f.nodes[i][j].output_channel.get()
                    t.nodes[0][0].input_channel.put(("append_input", msg))
                    t.nodes[0][0].output_channel.get()

        else:
            for i in range(len(f.nodes)):
                upper_i = math.floor(i / float(len(t.nodes)))
            
                for j in range(len(f.nodes[i])):
                    upper_j = math.floor(j / float(len(t.nodes[0])))
                    f.nodes[i][j].input_channel.put("get_output")
                    msg = f.nodes[i][j].output_channel.get()
                    t.nodes[int(upper_i)][int(upper_j)].input_channel.put(("append_input", msg))
                    t.nodes[int(upper_i)][int(upper_j)].output_channel.get()
    
    def expose(self, uInput, uJustUpperLeftCorner=False):
        """Expose an input pattern to first layer's nodes."""
        (input_height, input_width) = uInput.shape
        (layer_height, layer_width) = (len(self.layers[0].nodes), 
                                       len(self.layers[0].nodes))
        
        patch_height = input_height / layer_height
        patch_width = input_width / layer_width
    
        starting_point_h = 0;
        starting_point_w = 0;
        
        for i in range(layer_height):
            for j in range(layer_width):
                node = self.layers[0].nodes[i][j]
                patch = uInput[starting_point_h : starting_point_h + patch_height,
                               starting_point_w : starting_point_w + patch_width]
                
                patch = np.reshape(patch, (1, patch.size))[0]
    
                node.input_channel.put(("set_input", patch))
                node.output_channel.get()
                
                if uJustUpperLeftCorner: return

                starting_point_w += patch_width
                
            starting_point_w = 0
            starting_point_h += patch_height


## -----------------------------------------------------------------------------
## OutputNodeFinalizer Class
## -----------------------------------------------------------------------------
class OutputNodeFinalizer(object):
    """Implements the algorithm for computing the PCW in the output node."""
    def finalize(self, uNodeState):
        PCW = uNodeState['PCW']

        s = PCW.sum(axis=0)
        total = PCW.sum()
        cls_prior_prob = s / float(total)
        
        ## normalize the PCW matrix
        uNodeState['PCW'] = utils.normalize_over_cols(PCW)
        uNodeState['cls_prior_prob'] = cls_prior_prob


## -----------------------------------------------------------------------------
## NodeStateHandler Class
## -----------------------------------------------------------------------------
class NodeStateHandler(object):
    def clone(self, uNodeState):
        """Clone a node's state. Despite of the method name it implements
        a shallow copy."""
        s = {}
        s['coincidences'] = uNodeState['coincidences']
        s['temporal_groups'] = uNodeState['temporal_groups']
        s['PCG'] = uNodeState['PCG']
        return s
        
    def set_state(self, uNodeState, uNewState):
        """Set the internal state of a node."""
        uNodeState['coincidences'] = uNewState['coincidences']
        uNodeState['temporal_groups'] = uNewState['temporal_groups']
        uNodeState['PCG'] = uNewState['PCG']
        

## -----------------------------------------------------------------------------
## OutputNodeStateHandler Class
## -----------------------------------------------------------------------------
class OutputNodeStateHandler(NodeStateHandler):
    def clone(self, uNodeState):
        """Clone a node's state. Despite of the method name it implements
        a shallow copy."""
        s = {}
        s['coincidences'] = uNodeState['coincidences']
        s['cls_prior_prob'] = uNodeState['cls_prior_prob']
        s['PCW'] = uNodeState['PCW']
        return s

    def set_state(self, uNodeState, uNewState):
        uNodeState['coincidences'] = uNewState['coincidences']
        uNodeState['cls_prior_prob'] = uNewState['cls_prior_prob']
        uNodeState['PCW'] = uNewState['PCW']


## -----------------------------------------------------------------------------
## NodeFactory Class
## -----------------------------------------------------------------------------
class NodeFactory(object):
    """A factory of HTM nodes."""
    def make_node(self, uName, uType, uNodeSpec):
        ## initialize the new node's state with common structs
        state = {'name'         : uName,
                 'input_msg'    : np.array([]),
                 'coincidences' : np.array([[]]),
                 'output_msg'   : np.array([]),
                 'k'            : None,
                 
                 'distance_thr' : uNodeSpec['distance_thr'],
                 }
        
        ## then, add type-dependent structs
        if uType == ENTRY  or uType == INTERMEDIATE:
            state['temporal_groups'] = None
            state['seen'] = np.array([])
            state['TAM'] = np.array([[]])
            state['PCG'] = np.array([[]])
            state['k_prev'] = []

            ## if one wishes to change clustering algorithms,
            ## these params should be created by a factory
            state['transition_memory_size'] = uNodeSpec['transition_memory_size']
            state['top_neighbours'] = uNodeSpec['top_neighbours']
            state['max_group_size'] = uNodeSpec['max_group_size']
            state['min_group_size'] = uNodeSpec['min_group_size']
            
            ## !FIXME better ask forgiveness than permission?
            if uType == ENTRY:
                state['input_msg'] = np.array([])
                state['sigma'] = uNodeSpec['sigma']
            
        else: ## uType == OUTPUT
            state['cls_prior_prob'] = np.array([])
            state['PCW'] = np.array([[]])

        ## last, add type-dependent algorithms
        strategy = {}
        if uType == ENTRY:
            strategy['trainer'] = EntrySpatialPooler()
            strategy['finalizer'] = TemporalPooler()
            strategy['inference_maker'] = EntryInferenceMaker()
            strategy['state_handler'] = NodeStateHandler()
            
        elif uType == INTERMEDIATE:
            strategy['trainer'] = IntermediateSpatialPooler()
            strategy['finalizer'] = TemporalPooler()
            strategy['inference_maker'] = IntermediateInferenceMaker()
            strategy['state_handler'] = NodeStateHandler()

        elif uType == OUTPUT:
            strategy['trainer'] = OutputSpatialPooler()
            strategy['finalizer'] = OutputNodeFinalizer()
            strategy['inference_maker'] = OutputInferenceMaker()
            strategy['state_handler'] = OutputNodeStateHandler()
            
        ## the state is ready, make the node
        node = Node(state, strategy)
        
        ## set the node to be a daemon
        node.daemon = True
        return node


## -----------------------------------------------------------------------------
## NetworkBuilder Class
## -----------------------------------------------------------------------------
class NetworkBuilder(object):
    """Encapsulates all the mechanisms needed to craft an HTM."""
    def __init__(self, uNetworkSpec):
        self.spec = uNetworkSpec
        self.node_factory = NodeFactory()
        
    def build(self):
        """Build the network."""
        layers_type = [INTERMEDIATE for s in self.spec]
        layers_type[0] = ENTRY
        layers_type[-1] = OUTPUT
        layers = []
        
        ## iterate over layers to create
        for i in range(len(self.spec)):
            layer = Layer()
            (height, width) = self.spec[i]['shape']
            
            layer.nodes = []

            try: layer.node_sharing = self.spec[i]['node_sharing']
            except: pass

            ## create nodes
            for k in range(height):
                layer.nodes.append([])

                for l in range(width):
                    node = self.node_factory.make_node((k,l), 
                                                       layers_type[i],
                                                       self.spec[i])
                    layer.nodes[k].append(node)
                    
            layers.append(layer)
            
        network = Network()
        network.layers = layers
        network.spec = self.spec
        
        return network


## main/tests ------------------------------------------------------------------
if __name__ == "__main__":
    import config
    import usps
    import time
    
    builder = NetworkBuilder(config.test_net)
    htm = builder.build()
    
    htm.start()
    t0 = time.time()

    ## train layer 0
    image = usps.read("data_sets/train100/0/1.bmp")
    htm.expose(image)
    htm.layers[0].train({'temporal_gap' : False})
    htm.layers[0].train({'temporal_gap' : False})

    image = usps.read("data_sets/train100/0/2.bmp")
    htm.expose(image)
    htm.layers[0].train({'temporal_gap' : False})

    image = usps.read("data_sets/train100/0/3.bmp")
    htm.expose(image)
    htm.layers[0].train({'temporal_gap' : False})

    htm.layers[0].finalize()

    ## train layer 1
    image = usps.read("data_sets/train100/0/1.bmp")
    htm.expose(image)
    htm.layers[0].inference()
    htm.propagate(0, 1)
    htm.layers[1].train({'temporal_gap' : False})

    image = usps.read("data_sets/train100/0/3.bmp")
    htm.expose(image)
    htm.layers[0].inference()
    htm.propagate(0, 1)
    htm.layers[1].train({'temporal_gap' : True})

    image = usps.read("data_sets/train100/0/2.bmp")
    htm.expose(image)
    htm.layers[0].inference()
    htm.propagate(0, 1)
    htm.layers[1].train({'temporal_gap' : False})
    
    htm.layers[1].finalize()
    
    ## train layer 2
    image = usps.read("data_sets/train100/0/1.bmp")
    htm.expose(image)
    htm.layers[0].inference()
    htm.propagate(0, 1)
    htm.layers[1].inference()
    htm.propagate(1, 2)
    htm.layers[2].train({'class' : 0})

    image = usps.read("data_sets/train100/1/1.bmp")
    htm.expose(image)
    htm.layers[0].inference()
    htm.propagate(0, 1)
    htm.layers[1].inference()
    htm.propagate(1, 2)
    htm.layers[2].train({'class' : 1})

    image = usps.read("data_sets/train100/2/1.bmp")
    htm.expose(image)
    htm.layers[0].inference()
    htm.propagate(0, 1)
    htm.layers[1].inference()
    htm.propagate(1, 2)
    htm.layers[2].train({'class' : 2})
    
    htm.layers[2].finalize()

    print "Network output is: ", htm.inference(image)
    print time.time() - t0, "seconds"
