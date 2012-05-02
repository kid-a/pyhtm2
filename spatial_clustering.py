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
                
    def select_active_coinc(self, uNodeState):
        """Selects the active coincidence."""
        coincidences = uNodeState['coincidences']

        if coincidences.size == 0:
            k = self.make_new_coincidence(uNodeState, uFirstCoinc=True)

        else:
            distance_thr = uNodeState['distance_thr']
            input_msg = uNodeState['input_msg']
            (k, minimum) = self.closest_coincidence(coincidences, input_msg)
            
            if minimum > distance_thr:
                k = self.make_new_coincidence(uNodeState)
        
        ## set the current active coincidence
        uNodeState['k'] = k

        ## increment the seen vector
        uNodeState['seen'][k] += 1
                
    def make_new_coincidence(self, uNodeState, uFirstCoinc=False): pass
    def closest_coincidence(self, uCoincidences, uInputMsg): pass

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
## EntrySpatialPooler Class
## -----------------------------------------------------------------------------
class EntrySpatialPooler(SpatialPooler):
    def make_new_coincidence(self, uNodeState, uFirstCoinc=False):
        """Make a new coincidence and updates the size of the node's internal matrices."""
        input_msg = uNodeState['input_msg']

        if uFirstCoinc:
            coincidences = np.array([input_msg])
            k = 0
            seen = np.array([1])
            TAM = np.array([[0]])
            
        else:
            coincidences = uNodeState['coincidences']
            seen = uNodeState['seen']
            TAM = uNodeState['TAM']
            
            coincidences = np.vstack((coincidences, input_msg))
            (k, _) = coincidences.shape
            k -= 1
            seen = np.hstack((seen, 0))

            ## resize TAM
            TAM = utils.inc_rows_cols(TAM)

        ## update the node's state
        uNodeState['coincidences'] = coincidences
        uNodeState['seen'] = seen
        uNodeState['TAM'] = TAM

        return k
    
    def closest_coincidence(self, uCoincidences, uInputMsg):
        """Compute the distance of each coincidence from a given input."""
        distances = np.sum(np.abs(uCoincidences - \
                                      uInputMsg) ** 2, axis=-1) ** (1./2)
            
        ## find the minimum
        k = np.argmin(distances)
        minimum = distances[k]
        
        return (k, minimum)


## -----------------------------------------------------------------------------
## IntermediateSpatialPooler Class
## -----------------------------------------------------------------------------
class IntermediateSpatialPooler(SpatialPooler):
    def make_new_coincidence(self, uNodeState, uFirstCoinc=False):
        """Make a new coincidence and updates the size of the node's internal matrices."""
        input_msg = uNodeState['input_msg']

        if uFirstCoinc:
            print "****", input_msg
            coincidences = np.array([compute_widx(input_msg)])
            print "*****", coincidences
            k = 0
            seen = np.array([1])
            TAM = np.array([[0]])
            
        else:
            coincidences = uNodeState['coincidences']
            TAM = uNodeState['TAM']
            seen = uNodeState['seen']

            coincidences = np.vstack((coincidences, w))
            (k, _) = coincidences.shape
            k -= 1
            seen = np.hstack((seen, 0))
            
            ## resize TAM
            TAM = utils.inc_rows_cols(TAM)

        ## update the node's state
        uNodeState['coincidences'] = coincidences
        uNodeState['seen'] = seen
        uNodeState['TAM'] = TAM
        
    def closest_coincidence(self, uCoincidences, uInputMsg): 
        """Compute the distance of each coincidence from a given input."""       
        w = compute_widx(uInputMsg)
        distances = np.apply_along_axis(widx_distance, 1, (uCoincidences - w))
        k = np.argmin(distances)
        minimum = distances[k]
        
        return (k, minimum)        


## -----------------------------------------------------------------------------
## OutputSpatialPooler Class
## -----------------------------------------------------------------------------
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
