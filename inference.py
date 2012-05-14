## global import ---------------------------------------------------------------
import numpy as np


## local import ----------------------------------------------------------------
import utils


def input_msg2array(uInputMsg):
    shape = (len(uInputMsg), max([len(x) for x in uInputMsg]))
    array = np.zeros(shape)
    
    for i in range(shape[0]):
        for j in range(len(uInputMsg[i])):
            array[i,j] = uInputMsg[i][j]
            
    return array

def make_mask(uCoincidence, uInput):
    mask = np.zeros_like(uInput)
    for i in range(len(uCoincidence)):
        mask[i, uCoincidence[i]] = 1

    return mask

def select_components(uInput, uMask):
    return uInput[uMask != 0]


## -----------------------------------------------------------------------------
## InferenceMaker Class
## -----------------------------------------------------------------------------
class InferenceMaker(object):
    """Abstract class representing an inference maker."""
    def dens_over_coinc(self, uCoincidences, uCurrentInput):
        #uCurrentInput = np.array(uCurrentInput)
        input_array = input_msg2array(uCurrentInput)
        
        (rows, cols) = uCoincidences.shape
        y = np.array([[]])

        for i in range(rows):
            
            # selected_coincidences = np.array([])

            # for j in range(cols):
            #     selected_coincidences = \
            #         np.append(selected_coincidences, 
            #                   uCurrentInput[j][uCoincidences[i,j]])
            mask = make_mask(uCoincidences[i], input_array)
            selected_components = select_components(input_array, mask)

            y = np.append(y, np.prod(selected_components))
            
        return np.array([y])    
    
    def dens_over_matrix(self, uY, uMatrix):
        return np.dot(uY, uMatrix)
    
    def inference(self, uNodeState): pass


## -----------------------------------------------------------------------------
## EntryInferenceMaker Class
## -----------------------------------------------------------------------------
class EntryInferenceMaker(InferenceMaker):
    """Implements inference algorithms for entry layer nodes."""
    def dens_over_coinc(self, uCoincidences, uCurrentInput, uSigma=1):
        coinc = np.array(uCoincidences, dtype=np.double)
        input_msg = np.array(uCurrentInput, dtype=np.double)
        
        y = np.sqrt(np.sum(np.power(coinc - input_msg, 2), axis=1))  
        y = np.exp(- np.power(y, 2) / np.power(uSigma, 2))
        return y

    def inference(self, uNodeState):
        coinc = uNodeState['coincidences']
        input_msg = uNodeState['input_msg']
        sigma = uNodeState['sigma']
        PCG = uNodeState['PCG']
        
        y = self.dens_over_coinc(coinc, input_msg, sigma)
        y = (utils.normalize_over_rows(np.array([y])))[0]
        z = self.dens_over_matrix(y, PCG)
        
        uNodeState['output_msg'] = z


## -----------------------------------------------------------------------------
## EntryInferenceMaker Class
## -----------------------------------------------------------------------------
class IntermediateInferenceMaker(InferenceMaker):
    """Implements inference algorithms for intermediate layer nodes."""
    def inference(self, uNodeState):
        coinc = uNodeState['coincidences']
        input_msg = uNodeState['input_msg']
        PCG = uNodeState['PCG']
        
        y = self.dens_over_coinc(coinc, input_msg)
        y = utils.normalize_over_rows(y)[0]
        z = self.dens_over_matrix(y, PCG)

        uNodeState['output_msg'] = z


## -----------------------------------------------------------------------------
## OutputInferenceMaker Class
## -----------------------------------------------------------------------------
class OutputInferenceMaker(InferenceMaker):
    """Implements inference algorithms for output nodes."""
    def class_post_prob(self, uZ, uClassPriorProb):
        total = np.dot(uZ, uClassPriorProb)
        return uZ * uClassPriorProb / total
    
    def inference(self, uNodeState):
        coinc = uNodeState['coincidences']
        input_msg = uNodeState['input_msg']
        cls_prior_prob = uNodeState['cls_prior_prob']
        PCW = uNodeState['PCW']
        
        y = self.dens_over_coinc(coinc, input_msg)
        y = utils.normalize_over_rows(y)
        z = self.dens_over_matrix(y, PCW)
        p = self.class_post_prob(z, cls_prior_prob)
        uNodeState['output_msg'] = p


if __name__ == "__main__":
    # from network import Node
    # from spatial_clustering import EntrySpatialPooler

    # pass

    import numpy.ma as ma
    c = np.array([[0,1],[1,1],[1,0]])
    #i = np.array([[1,2,3],[1,2,0]])
    i = input_msg2array(np.array([[1,2,3],[1,2]]))
    print i
    m = make_mask(c[1], i)
    print m
    print i[m!=0]
