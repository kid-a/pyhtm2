## global import ---------------------------------------------------------------
import numpy as np


## local import ----------------------------------------------------------------
import utils


RECORD_ACTIVATION = True


## -----------------------------------------------------------------------------
## InferenceMaker Class
## -----------------------------------------------------------------------------
class InferenceMaker(object):
    """Abstract class representing an inference maker."""
    def dens_over_coinc(self, uCoincidences, uCurrentInput):
        (rows, cols) = uCoincidences.shape
        y = np.array([[]])

        for i in range(rows):
            selected_coincidences = np.array([])

            for j in range(cols):
                selected_coincidences = \
                    np.append(selected_coincidences, 
                              uCurrentInput[j][uCoincidences[i,j]])

            y = np.append(y, np.prod(selected_coincidences))
            
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
        y = np.sqrt(np.sum(np.power(uCoincidences - uCurrentInput, 2), axis=1))  
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

## -----------------------------------------------------------------------------
## OutputInferenceMaker Class
## -----------------------------------------------------------------------------
class ActivationRecorder(object):
    def __init__(self, *args, **kwargs):
        self.activation = []

    def record(self, uCoincidences, uY, uSigma):
        if RECORD_ACTIVATION:
            (total_coinc, _) = uCoincidences.shape
            three_percent = np.ceil(total_coinc * 3 / 100.0)
            
            ordered_y = uY.tolist()
            ordered_y = sorted(ordered_y)[::-1]
            ordered_y = ordered_y[:int(three_percent)]
            
            three_percent_activation = sum(ordered_y)
            self.activation.append(three_percent_activation/np.sum(uY))
            
            ## save on file, constantly
            (np.array(self.activation)).tofile("activation-sigma-" + str(uSigma) + ".txt", sep=",")


if __name__ == "__main__":
    from network import Node
    from spatial_clustering import EntrySpatialPooler

    pass
