import numpy as np
import utils
import network


class SpatialPooler(object):
    def __init__(self, uThreshold, uTransitionMemory, *args, **kwargs):
        self.threshold = uThreshold
        self.transition_memory = uTransitionMemory
    

    def train_node(self, uNode, uTemporalGap=False):
        """Train a node on the current input."""
        ## select the active threshold
        self.select_active_coinc(uNode)
    
        ## then, update the temporal activation matrix (TAM)
        if not uTemporalGap:
            for t in range(len(uNode.k_prev)):
                uNode.TAM[uNode.k_prev[t], uNode.k] = \
                    uNode.TAM[uNode.k_prev[t], uNode.k] + 1 + self.transition_memory - (t + 1)
        
        ## last, add k to the k_prev list
        uNode.k_prev.insert(0, uNode.k)
        if len(uNode.k_prev) > self.transition_memory:
            uNode.k_prev = uNode.k_prev[1:]        
        
    def select_active_coinc(self, uNode): pass


class EntrySpatialPooler(SpatialPooler):
    def select_active_coinc(self, uNode):
        """Given a node, selects its current active coincidence."""
        ## if processing the first input pattern,
        ## immediately make a new coincidence and return        
        if uNode.coincidences.size == 0:
            uNode.coincidences = np.array([uNode.input_msg])
            uNode.k = 0
            uNode.seen = [1]
            uNode.TAM = np.array([[0]])
            
        else:
            ## compute the distance of each coincidence from the
            ## given input
            distances = np.apply_along_axis(np.linalg.norm, 1, 
                                                (uNode.coincidences - uNode.input_msg))
                ## find the minimum
            uNode.k = np.argmin(distances)
            minimum = distances[uNode.k]
            
            ## if the closest coincidence is not close enough,
            ## make a new coincidence
            if minimum > self.threshold:
                uNode.coincidences = np.vstack((uNode.coincidences, uNode.input_msg))
                (uNode.k, _) = uNode.coincidences.shape
                uNode.k -= 1
                uNode.seen = np.hstack((uNode.seen, 0))
                ## resize TAM
                uNode.TAM = utils.inc_rows_cols(uNode.TAM)
                
            ## increment the seen vector
            uNode.seen[uNode.k] += 1


if __name__ == "__main__":
    n = network.Node()
    s = EntrySpatialPooler(0, 5)

    n.input_msg = np.array([1,2,3,4])
    s.train_node(n)
    
    print n.coincidences
    print n.seen
    print n.TAM
    print n.k
    print n.k_prev
    print

    s.train_node(n)

    print n.coincidences
    print n.seen
    print n.TAM
    print n.k
    print n.k_prev
    print

    n.input_msg = np.array([0,0,0,0])

    s.train_node(n)

    print n.coincidences
    print n.seen
    print n.TAM
    print n.k
    print n.k_prev
    print

    
    
    
