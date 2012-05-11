## global import ---------------------------------------------------------------
import numpy as np
import sys


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
## widx_distance1
## -----------------------------------------------------------------------------
def widx_distance1(row): 
    return (row != 0).sum()


## -----------------------------------------------------------------------------
## widx_distance2
## -----------------------------------------------------------------------------
def widx_distance2(array): 
    return (array != 0).sum(axis=1)


## -----------------------------------------------------------------------------
## vect_widx_distance
## -----------------------------------------------------------------------------
#vect_widx_distance = np.vectorize(widx_distance)


## -----------------------------------------------------------------------------
## SpatialPooler abstract class
## -----------------------------------------------------------------------------
class SpatialPooler(object):
    """Implements the algorithms for clustering coincidences."""    
    def train(self, uNodeState, uInputInfo):
        """Train a node on the current input."""
        ## select the active coincidence
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
        try: uNodeState['seen'][k] += 1
        except: pass
                
    def make_new_coincidence(self, uNodeState, uFirstCoinc=False):
        """Make a new coincidence and updates the size of the node's internal matrices."""
        input_msg = uNodeState['input_msg']

        if uFirstCoinc:
            coincidences = np.array([compute_widx(input_msg)])
            k = 0
            seen = np.array([1])
            TAM = np.array([[0]])
            
        else:
            coincidences = uNodeState['coincidences']
            TAM = uNodeState['TAM']
            seen = uNodeState['seen']

            (rows, cols) = coincidences.shape
            new_coincidences = np.zeros((rows + 1, cols))
            new_coincidences[:rows,:cols] = coincidences
            new_coincidences[rows,:] = compute_widx(input_msg)
            coincidences = new_coincidences

            (k, _) = coincidences.shape
            k -= 1

            (cols,) = seen.shape
            new_seen = np.zeros(cols + 1)
            new_seen[:cols] = seen
            seen = new_seen
            
            ## resize TAM
            TAM = utils.inc_rows_cols(TAM)
            
        ## update the node's state
        uNodeState['coincidences'] = coincidences
        uNodeState['seen'] = seen
        uNodeState['TAM'] = TAM
        
        return k

    def closest_coincidence(self, uCoincidences, uInputMsg): 
        """Compute the distance of each coincidence from a given input."""       
        w = compute_widx(uInputMsg)
        distances = widx_distance2(uCoincidences - w)
        k = np.argmin(distances)
        minimum = distances[k]
        
        return (k, minimum)

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
            
            (rows, cols) = coincidences.shape
            new_coincidences = np.zeros((rows + 1, cols))
            new_coincidences[:rows,:cols] = coincidences
            new_coincidences[rows,:] = input_msg
            coincidences = new_coincidences

            (k, _) = coincidences.shape
            k -= 1

            (cols,) = seen.shape
            new_seen = np.zeros(cols + 1)
            new_seen[:cols] = seen
            seen = new_seen

            ## resize TAM
            TAM = utils.inc_rows_cols(TAM)

        ## update the node's state
        uNodeState['coincidences'] = coincidences
        uNodeState['seen'] = seen
        uNodeState['TAM'] = TAM

        return k
    
    def closest_coincidence(self, uCoincidences, uInputMsg):
        """Compute the distance of each coincidence from a given input."""

        distances = np.sqrt(np.sum(np.power(uCoincidences - uInputMsg, 2), axis=1))
            
        ## find the minimum
        k = np.argmin(distances)
        minimum = distances[k]
        
        return (k, minimum)


## -----------------------------------------------------------------------------
## IntermediateSpatialPooler Class
## -----------------------------------------------------------------------------
class IntermediateSpatialPooler(SpatialPooler):
    pass


## -----------------------------------------------------------------------------
## OutputSpatialPooler Class
## -----------------------------------------------------------------------------
class OutputSpatialPooler(SpatialPooler):
    def train(self, uNodeState, uInputInfo):
        """Train the output node on the current input."""
        ## select the active coincidence
        self.select_active_coinc(uNodeState)
        self.update_PCW(uNodeState, uInputInfo)

    def make_new_coincidence(self, uNodeState, uFirstCoinc=False):
        """Make a new coincidence and updates the size of the node's
        internal matrices."""
        input_msg = uNodeState['input_msg']

        if uFirstCoinc:
            coincidences = np.array([compute_widx(input_msg)])
            k = 0
            
        else:
            coincidences = uNodeState['coincidences']

            (rows, cols) = coincidences.shape
            new_coincidences = np.zeros((rows + 1, cols))
            new_coincidences[:rows,:cols] = coincidences
            new_coincidences[rows,:] = compute_widx(input_msg)
            coincidences = new_coincidences
            
            (k, _) = coincidences.shape
            k -= 1
            
        ## update the node's state
        uNodeState['coincidences'] = coincidences
        
        return k
        
        
    def update_PCW(self, uNodeState, uInputInfo):
        """Update the PCW matrix."""
        PCW = uNodeState['PCW']
        k = uNodeState['k']
        cls = uInputInfo['class']
        
        try: 
            PCW[k, cls] += 1
        except:
            (rows, cols) = PCW.shape
            (delta_r, delta_c) = (k + 1 - rows, cls + 1 - cols)
            if delta_r < 0: delta_r = 0
            if delta_c < 0: delta_c = 0
            
            PCW = utils.resize(PCW, (rows + delta_r, cols + delta_c))
            PCW[k, cls] = 1
            
        uNodeState['PCW'] = PCW
            
    def update_TAM(self, uNodeState): pass


if __name__ == "__main__":
    ## profile inc_rows_cols
    import profile

    p = IntermediateSpatialPooler()
    
    def profile_clustering():
        coincidences = np.random.randint(0, 100, size=(12000, 4))
        input_msg = [np.random.random((1, 10)),
                     np.random.random((1, 10)),
                     np.random.random((1, 10)),
                     np.random.random((1, 10))]


        p.closest_coincidence(coincidences, input_msg)

    profile.run("profile_clustering()")
    
