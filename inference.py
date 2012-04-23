## the inference module

import network
import numpy as np
import utils


class InferenceMaker(object):
    def dens_over_coinc(self, uCoincidences, uCurrentInput, uSigma=None):
        (rows, cols) = uCoincidences.shape
        y = np.array([[]])
    
        for i in range(rows):
            selected_coincidences = np.array([])
            
            for j in range(cols):
            
                selected_coincidences = \
                    np.append(selected_coincidences, 
                              uCurrentInput[j][uCoincidences[i][j]])

            y = np.append(y, np.prod(selected_coincidences))
            
        return y

    def dens_over_groups(self, uY, uPCG):
        return y * uPCG

    def inference(self, uNode, uSigma=None):
        y = utils.normalize_over_rows(self.dens_over_coinc(uNode.coincidences,
                                                           uNode.input_msg,
                                                           uSigma))
        z = self.dens_over_groups(y, uNode.PCG)
        uNode.output_msg = z
        

class EntryInferenceMaker(InferenceMaker):
    def dens_over_coinc(self, uCoincidences, uCurrentInput, uSigma=1):
        y = np.apply_along_axis(np.linalg.norm, 1, (uCoincidences - uCurrentInput))
        y = np.exp(- np.power(y, 2) / np.power(uSigma, 2))
        return np.array([y])


class IntermediateInferenceMaker(InferenceMaker):
    pass
    

class OutputInferenceMaker(InferenceMaker):
    def dens_over_classes(self, uGroups):
        pass


if __name__ == "__main__":
    from network import Node
    from spatial_clustering import EntrySpatialPooler
    
    n = Node()

    ## train entry node
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

    n.input_msg = np.array([1,1,1,1])
    s.train_node(n)
    
    print n.coincidences
    print n.seen
    print n.TAM
    print n.k
    print n.k_prev
    print

    n.input_msg = np.array([1,2,3,4])
    s.train_node(n)
    
    print n.coincidences
    print n.seen
    print n.TAM
    print n.k
    print n.k_prev
    print

    print "Now making inference"
    e = EntryInferenceMaker()
    e.inference(n, 25.0)

    print n.output_msg
    print n.coincidences
    print n.seen
    print n.TAM
    print n.k
    print n.k_prev
    

    
