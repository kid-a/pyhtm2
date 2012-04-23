## the inference module

import network
import numpy as np
import utils


class InferenceMaker(object):
    def dens_over_coinc(self, uCoincidences, uCurrentInput, uSigma=None):
        (rows, cols) = uCoincidences.shape
        y = np.array([])
    
        for i in range(rows):
            selected_coincidences = np.array([])
            
            for j in range(cols):
            
                selected_coincidences = \
                    np.append(selected_coincidences, 
                              uCurrentInput[j][uCoincidences[i][j]])

            y = np.append(y, np.prod(selected_coincidences))
            
        return y

    def dens_over_groups(self, uY, uPCG): pass

    def inference(self, uNode, uSigma=None):
        y = utils.normalize_over_rows(self.dens_over_coinc(uNode.coincidences,
                                                           uNode.in_msg,
                                                           uSigma))
        z = self.dens_over_groups(y, uNode.PCG)
        uNode.out_msg = z
        

class EntryInferenceMaker(InferenceMaker):
    def dens_over_coinc(self, uCoincidences, uCurrentInput, uSigma=1):
        y = np.apply_along_axis(np.linalg.norm, 1, (uCoincidences - uCurrentInput))
        y = np.exp(- np.power(y, 2) / np.power(uSigma, 2))
        return y


class IntermediateInferenceMaker(InferenceMaker):
    pass
    

class OutputInferenceMaker(InferenceMaker):
    def dens_over_classes(self, uGroups):
        pass
