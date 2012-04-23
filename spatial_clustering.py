import numpy as np
import utils
import network


class SpatialPooler(object):
    def select_active_coinc(self, uNode): pass


class EntrySpatialPooler(SpatialPooler):
    def __init__(self, *args, **kwargs):
        self.threshold = None
        self.transition_memory = None

    def train_node(self, uNode, uTemporalGap=False):
        """Train a node on the current input."""
        ## select the active threshold
        self.select_active_coinc(uNode, self.threshold)
    
        ## then, update the temporal activation matrix (TAM)
        if not uTemporalGap:            
            for t in range(uNode.k_prev):
                uNode.TAM[uNode.k_prev[t], uNode.k] = \
                    uNode.TAM[uNode.k_prev[t], uNode.k] + 1 + self.transition_memory - (t + 1)
        
        ## last, add k to the k_prev list
        uNode.k_prev.insert(0, k)
        if len(uNode.k_prev) > self.transition_memory:
            uNode.k_prev = uNode.k_prev[1:]
        
    
    def select_active_coinc(self, uNode, uThreshold):
        """Given a node, selects its current active coincidence."""
        ## if this is the first input pattern,
        ## immediately make a new coincidence and return        
        if uNode.coincidences.size == 0:
            uNode.coincidences = uNode.input_msg
            uNode.k = 0
            uNode.seen = [1]
            uNode.TAM = [1]
            
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
            uNode.seen[k] += 1
