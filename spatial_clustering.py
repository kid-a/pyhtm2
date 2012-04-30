## global import ---------------------------------------------------------------
import numpy as np


## local import ----------------------------------------------------------------
import utils


## -----------------------------------------------------------------------------
## compute_widx
## -----------------------------------------------------------------------------
def compute_widx(msg):
    """Compute a widx, out of an input message."""
    widx = []
    for m in msg: widx.append(np.argmax(m))
    return np.array(widx)


## -----------------------------------------------------------------------------
## widx_distance
## -----------------------------------------------------------------------------
def widx_distance(diff): return (diff != 0).sum()


## -----------------------------------------------------------------------------
## SpatialPooler abstract class
## -----------------------------------------------------------------------------
class SpatialPooler(object):
    """Implements the algorithms for clustering coincidences."""    
    def train(self, uNodeState, uInputInfo):
        """Train a node on the current input."""
        ## select the active threshold
        self.select_active_coinc(uNodeState)
        self.update_TAM(uNodeState, uInputInfo)
        
    def select_active_coinc(self, uNodeState): pass

    def update_TAM(self, uNodeState, uInputInfo):
        """Update the temporal activation matrix (TAM)."""
        TAM = uNodeState['TAM']
        k_prev = uNodeState['k_prev']
        k = uNodeState['k']
        transition_memory = uNodeState['transition_memory_size']
            
        ## then, update the temporal activation matrix (TAM)
        if not uInputInfo['temporal_gap']:
            for t in range(len(k_prev)):
                TAM[k_prev[t], k] = TAM[k_prev[t], k] + transition_memory - t

        ## last, add k to the k_prev list
        k_prev.insert(0, k)
        if len(k_prev) > transition_memory:
            k_prev = k_prev[:-1]        
    
        ## update the node's state
        uNodeState['TAM'] = TAM
        uNodeState['k_prev'] = k_prev
        


## -----------------------------------------------------------------------------
## SpatialPooler abstract class
## -----------------------------------------------------------------------------
class EntrySpatialPooler(SpatialPooler):
    def select_active_coinc(self, uNodeState):
        """Given a node, selects its current active coincidence."""
        distance_thr = uNodeState['distance_thr']
        coincidences = uNodeState['coincidences']
        input_msg = uNodeState['input_msg']
        k = uNodeState['k']
        seen = uNodeState['seen']
        TAM = uNodeState['TAM']
        
        ## if processing the first input pattern,
        ## immediately make a new coincidence and return
        if coincidences.size == 0:
            coincidences = np.array([input_msg])
            k = 0
            seen = np.array([1])
            TAM = np.array([[0]])
            
        else:
            ## compute the distance of each coincidence from the
            ## given input
            distances = np.sum(np.abs(coincidences - \
                                          input_msg) ** 2, axis=-1) ** (1./2)
            
            ## find the minimum
            k = np.argmin(distances)
            minimum = distances[k]
            
            ## if the closest coincidence is not close enough,
            ## make a new coincidence
            if minimum > distance_thr:
                coincidences = np.vstack((coincidences, input_msg))
                (k, _) = coincidences.shape
                k -= 1
                seen = np.hstack((seen, 0))

                ## resize TAM
                TAM = utils.inc_rows_cols(TAM)
                
            ## increment the seen vector
            seen[k] += 1
        
        uNodeState['coincidences'] = coincidences
        uNodeState['k'] = k
        uNodeState['seen'] = seen
        uNodeState['TAM'] = TAM


class IntermediateSpatialPooler(SpatialPooler):
    def select_active_coinc(self, uNode):
        """Given a node, selects its current active coincidence."""
        ## if processing the first input pattern,
        ## immediately make a new coincidence and return
        if uNode.coincidences.size == 0:
            uNode.coincidences = np.array([compute_widx(uNode.input_msg)])
            uNode.k = 0
            uNode.seen = np.array([1])
            uNode.TAM = np.array([[0]])
            
        else:
            ## compute the distance of each coincidence from the
            ## given input
            w = compute_widx(uNode.input_msg)
            distances = np.apply_along_axis(widx_distance, 1,
                                            (uNode.coincidences - w))
            uNode.k = np.argmin(distances)
            minimum = distances[uNode.k]
            
            ## if the closest coincidence is not close enough,
            ## make a new coincidence
            if minimum > self.threshold:
                uNode.coincidences = np.vstack((uNode.coincidences, w))
                (uNode.k, _) = uNode.coincidences.shape
                uNode.k -= 1
                uNode.seen = np.hstack((uNode.seen, 0))

                ## resize TAM
                uNode.TAM = utils.inc_rows_cols(uNode.TAM)
                
            ## increment the seen vector
            uNode.seen[uNode.k] += 1


class OutputSpatialPooler(SpatialPooler):
    def train_node(self, uNode, uClass, uTemporalGap=False):
        """Train a node on the current input."""
        ## select the active threshold
        self.select_active_coinc(uNode)
        
        ## update the PCW matrix
        try: uNode.PCW[uNode.k, uClass] += 1
        except:
            (rows, cols) = uNode.PCW.shape
            (delta_r, delta_c) = (uNode.k + 1 - rows, uClass + 1 - cols)
            if delta_r < 0: delta_r = 0
            if delta_c < 0: delta_c = 0
            
            uNode.PCW.resize((rows + delta_r, cols + delta_c), refcheck=False)
            uNode.PCW[uNode.k, uClass] = 1
        
            
    def update_TAM(self, uNodeState): pass

    def select_active_coinc(self, uNode):
        """Given a node, selects its current active coincidence."""
        ## if processing the first input pattern,
        ## immediately make a new coincidence and return
        if uNode.coincidences.size == 0:
            uNode.coincidences = np.array([compute_widx(uNode.input_msg)])
            uNode.k = 0
            
        else:
            ## compute the distance of each coincidence from the
            ## given input
            w = compute_widx(uNode.input_msg)
            distances = np.apply_along_axis(widx_distance, 1,
                                            (uNode.coincidences - w))
            uNode.k = np.argmin(distances)
            minimum = distances[uNode.k]
            
            ## if the closest coincidence is not close enough,
            ## make a new coincidence
            if minimum > self.threshold:
                uNode.coincidences = np.vstack((uNode.coincidences, w))
                (uNode.k, _) = uNode.coincidences.shape
                uNode.k -= 1
                

        

# if __name__ == "__main__":
#     # print widx_distance(np.array([1,2,3,4]), np.array([0,0,0,0]))
#     # print widx_distance(np.array([1,2,3,4]), np.array([1,2,3,4]))

#     ## train output node
#     n = network.OutputNode()
#     s = OutputSpatialPooler(0)

#     n.input_msg = [ np.array([1,2,3,4]),
#                     np.array([4,5,6]),
#                     np.array([2,3,0.1]) ]
    
#     s.train_node(n, 0)

#     print n.coincidences
#     print n.k
#     print n.PCW
#     print

#     s.train_node(n, 1)

#     print n.coincidences
#     print n.k
#     print n.PCW
#     print

#     n.input_msg = [ np.array([4,3,2,1]),
#                     np.array([6,5,4]),
#                     np.array([3,2,0.1]) ]

#     s.train_node(n, 1)

#     print n.coincidences
#     print n.k
#     print n.PCW
#     print

#     n.input_msg = [ np.array([0,0,0,0]),
#                     np.array([1,5,4]),
#                     np.array([3,10,0.1]) ]

#     s.train_node(n, 5)

#     print n.coincidences
#     print n.k
#     print n.PCW
#     print
    
#     ## train intermediate node
#     # n = network.Node()
#     # s = IntermediateSpatialPooler(0, 5)
#     # n.input_msg = [ np.array([1,2,3,4]),
#     #                 np.array([4,5,6]),
#     #                 np.array([2,3,0.1]) ]

#     # s.train_node(n)

#     # print n.coincidences
#     # print n.seen
#     # print n.TAM
#     # print n.k
#     # print n.k_prev
#     # print

#     # s.train_node(n)

#     # print n.coincidences
#     # print n.seen
#     # print n.TAM
#     # print n.k
#     # print n.k_prev
#     # print

#     # n.input_msg = [ np.array([4,3,2,1]),
#     #                 np.array([6,5,4]),
#     #                 np.array([3,2,0.1]) ]
#     # s.train_node(n)

#     # print n.coincidences
#     # print n.seen
#     # print n.TAM
#     # print n.k
#     # print n.k_prev
#     # print



#     ## train entry node
#     # n = network.Node()
#     # s = EntrySpatialPooler(0, 5)
#     # n.input_msg = np.array([1,2,3,4])
#     # s.train_node(n)
#     # print n.coincidences
#     # print n.seen
#     # print n.TAM
#     # print n.k
#     # print n.k_prev
#     # print

#     # s.train_node(n)

#     # print n.coincidences
#     # print n.seen
#     # print n.TAM
#     # print n.k
#     # print n.k_prev
#     # print

#     # n.input_msg = np.array([0,0,0,0])
#     # s.train_node(n)

#     # print n.coincidences
    # print n.seen
    # print n.TAM
    # print n.k
    # print n.k_prev
    # print

    # n.input_msg = np.array([1,1,1,1])
    # s.train_node(n)
    
    # print n.coincidences
    # print n.seen
    # print n.TAM
    # print n.k
    # print n.k_prev
    # print

    # n.input_msg = np.array([1,2,3,4])
    # s.train_node(n)
    
    # print n.coincidences
    # print n.seen
    # print n.TAM
    # print n.k
    # print n.k_prev
    # print

    
    
    
